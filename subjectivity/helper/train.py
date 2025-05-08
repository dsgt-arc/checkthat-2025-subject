import copy
import logging
import os
import random
import time
from typing import Callable

import numpy as np
import polars as pl
import torch
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import get_linear_schedule_with_warmup

from subjectivity.helper.run_config import RunConfig
from subjectivity.helper.metrics import compute_class_report
from subjectivity.helper.util import format_time
from subjectivity.helper.inference import predict_test_data

# Explicitly encode labels given mapping dict to avoid any issues with orginal data vs our data
LABEL_ENCODING = {"SUBJ": 0, "OBJ": 1}
LABEL_DECODING = {value: key for key, value in LABEL_ENCODING.items()}


def train_val_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split data into train and validation set."""
    split_ratio = RunConfig.train["train_test_split"]
    logging.info(f"Splitting data with ratio '{split_ratio}'.")
    # Shuffle the DataFrame
    df = df.sample(fraction=1.0, shuffle=True, seed=RunConfig.train["seed"])
    # Compute split index
    split_idx = int(split_ratio * df.height)
    # Train/Test split
    df_train = df.slice(0, split_idx)
    df_val = df.slice(split_idx, df.height - split_idx)
    return df_train, df_val


def encode_labels(
    df: pl.DataFrame, mapping: dict[str, str] = LABEL_ENCODING
) -> pl.DataFrame:
    """Encode labels."""

    logging.info("Encoding labels.")
    df = df.with_columns(pl.col("label").replace_strict(mapping).alias("label_encoded"))

    return df


def decode_labels(tensor: torch.Tensor, mapping: dict[str, str] = LABEL_DECODING) -> torch.Tensor:
    """Decode labels."""
    logging.info("Decoding labels.")
    array = tensor.cpu().numpy()
    encoded_strings = np.vectorize(LABEL_DECODING.get)(array)
    return encoded_strings


def tokenize_data(
    tokenizer: torch.nn.Module, df_train: pl.DataFrame, df_val: pl.DataFrame, df_test: pl.DataFrame, max_length: int = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize data."""

    logging.info("Tokenizing training and validation data.")
    if max_length is None:
        max_length = RunConfig.encoder_model.get("max_length")
    logging.info(f"Tokenizing with context window/max_legnth of {max_length}")

    if not RunConfig.encoder_model["TOKENIZERS_PARALLELISM"]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logging.info("Set TOKENIZERS_PARALLELISM to false.")

    train_tokens = tokenizer.tokenizer(
        df_train["sentence"].to_list(),
        padding=RunConfig.encoder_model["padding"],
        truncation=RunConfig.encoder_model["truncation"],
        return_tensors=RunConfig.encoder_model["return_tensors"],
        return_attention_mask=RunConfig.encoder_model["return_attention_mask"],
        max_length=max_length,
        add_special_tokens=RunConfig.encoder_model["add_special_tokens"],
    )
    val_tokens = tokenizer.tokenizer(
        df_val["sentence"].to_list(),
        padding=RunConfig.encoder_model["padding"],
        truncation=RunConfig.encoder_model["truncation"],
        return_tensors=RunConfig.encoder_model["return_tensors"],
        return_attention_mask=RunConfig.encoder_model["return_attention_mask"],
        max_length=max_length,
        add_special_tokens=RunConfig.encoder_model["add_special_tokens"],
    )
    test_tokens = tokenizer.tokenizer(
        df_test["sentence"].to_list(),
        padding=RunConfig.encoder_model["padding"],
        truncation=RunConfig.encoder_model["truncation"],
        return_tensors=RunConfig.encoder_model["return_tensors"],
        return_attention_mask=RunConfig.encoder_model["return_attention_mask"],
        max_length=max_length,
        add_special_tokens=RunConfig.encoder_model["add_special_tokens"],
    )

    train_input_ids = train_tokens["input_ids"]
    val_input_ids = val_tokens["input_ids"]
    test_input_ids = test_tokens["input_ids"]

    train_masks = train_tokens["attention_mask"]
    val_masks = val_tokens["attention_mask"]
    test_masks = test_tokens["attention_mask"]

    return train_input_ids, val_input_ids, test_input_ids, train_masks, val_masks, test_masks


def set_up_dataloaders(
    batch_size: int,
    seed: int,
    num_workers: int,
    train_input_ids: torch.Tensor,
    val_input_ids: torch.Tensor,
    test_input_ids: torch.Tensor,
    train_masks: torch.Tensor,
    val_masks: torch.Tensor,
    test_masks: torch.Tensor,
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Set up dataloaders."""

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    logging.info(f"Setting up dataloaders for training with '{num_workers}' num_workers.")

    train_labels = torch.tensor(df_train["label_encoded"])
    val_labels = torch.tensor(df_val["label_encoded"])
    test_labels = torch.tensor(df_test["label_encoded"])

    # Wrapping into TensorDataset so that each sample will be retrieved by indexing tensors along the first dimension
    train_dataset = TensorDataset(train_input_ids, train_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_masks, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_masks, test_labels)

    # Set a seed for reproducibility in random sampler generation
    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset, generator=generator),
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker if ((num_workers > 0) and (torch.cuda.is_available())) else None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),  # Venktesh: SequentialSampler
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker if ((num_workers > 0) and (torch.cuda.is_available())) else None,
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),  # Venktesh: SequentialSampler
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker if ((num_workers > 0) and (torch.cuda.is_available())) else None,
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_step_size(epoch: int, num_batchs: int, warmup_steps: int) -> int:
    """Compute step size."""
    return (epoch * num_batchs) + warmup_steps


def set_up_optimizer(
    model, learning_rate: float, eps: float, num_warmup_steps: int, step_size: int
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """Set up optimizer."""
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=eps)
    logging.info(
        f"Setting up optimizer '{optimizer.__class__.__name__}' with learning rate '{learning_rate}' and epsilon '{eps}'."
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=step_size
    )
    return optimizer, scheduler


def train(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    loss: torch.nn.CrossEntropyLoss,
    metric_fns: dict[str, Callable],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    early_stopping_patience: int = 5,
    log_fn: Callable[[dict], None] | None = None,
) -> tuple[torch.nn.Module, float]:
    """Train model with early stopping and validation macro-F1 tracking."""
    if not isinstance(metric_fns, dict):
        metric_fns = {"metric": metric_fns}

    best_macro_f1 = -1.0
    best_model_state_dict = None
    patience_counter = 0

    if device in ["cuda", "cpu"]:
        scaler = GradScaler()
        logging.info("Using GradScaler for mixed precision training.")

    logging.info("Start training.")
    start_training_time = time.time()

    for epoch in range(epochs):
        logging.info(f"  Epoch {epoch + 1}/{epochs}")
        logging.info("  Training...")
        model.train()
        total_train_loss = 0
        train_preds = []
        train_labels = []
        epoch_start_training_time = time.time()

        # Training loop
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()  # Reset gradients

            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with autocast(device.type):
                outputs = model(input_ids, attention_mask)
                loss_value = loss(outputs, labels)

            if device in ["cuda", "cpu"]:
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_value.backward()
                optimizer.step()

            total_train_loss += loss_value.item()
            train_preds.append(outputs)
            train_labels.append(labels)

            if RunConfig.train["step_per"] == "batch":
                scheduler.step()

            if step % 40 == 0 and step > 0:
                elapsed = format_time(time.time() - epoch_start_training_time)
                logging.info("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len(train_dataloader), elapsed))

        if RunConfig.train["step_per"] == "epoch":
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_metrics_value = {metric_name: metric_fn(train_preds, train_labels) for metric_name, metric_fn in metric_fns.items()}

        # Validation loop
        model.eval()
        total_val_loss = 0
        val_preds = []
        val_labels = []
        logging.info("  Running Validation...")
        epoch_start_validation_time = time.time()

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                with autocast(device.type):
                    outputs = model(input_ids, attention_mask)
                    loss_value = loss(outputs, labels)

                total_val_loss += loss_value.item()
                val_preds.append(outputs)
                val_labels.append(labels)

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_metrics_value = {metric_name: metric_fn(val_preds, val_labels) for metric_name, metric_fn in metric_fns.items()}

        validation_time = format_time(time.time() - epoch_start_validation_time)
        logging.info("  Validation took: {:}".format(validation_time))
        logging.info(
            f"  Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Metrics: {train_metrics_value} Val Metrics: {val_metrics_value}"
        )

        val_labels_decoded = decode_labels(val_labels)
        val_preds_decoded = decode_labels(torch.argmax(val_preds, dim=1))
        val_classification_report = compute_class_report(val_labels_decoded, val_preds_decoded)
        logging.info(f"Validation classification report:\n{val_classification_report}")

        # Early stopping based on macro-F1
        current_f1 = val_metrics_value["macro_f1"]
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            patience_counter = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info("  Early stopping triggered.")
                break

        # Logging with ML-experiment tracker (wandb)
        if log_fn:
            log_fn(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "train/macro_f1": train_metrics_value["macro_f1"],
                    "val/macro_f1": val_metrics_value["macro_f1"],
                    "learning_rate": scheduler.get_last_lr()[0],
                    "train/accuracy": train_metrics_value.get("accuracy"),
                    "val/accuracy": val_metrics_value.get("accuracy"),
                    "val/OBJ_class_f1": val_classification_report.filter(pl.col("category") == "OBJ")["f1-score"][0],
                    "val/SUBJ_class_f1": val_classification_report.filter(pl.col("category") == "SUBJ")["f1-score"][0],
                }
            )

    # Restore best model
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)

    elapsed_training_time = format_time(time.time() - start_training_time)
    logging.info(f"Training complete. Duration: {elapsed_training_time}. Best macro-F1 fine-tuning validation: {best_macro_f1:.4f}")

    # Predict dev test data given best model
    test_predictions, test_labels = predict_test_data(test_dataloader, model, device)
    test_metrics_value = {metric_name: metric_fn(test_predictions, test_labels) for metric_name, metric_fn in metric_fns.items()}

    test_labels_decoded = decode_labels(test_labels)
    test_predictions_decoded = decode_labels(torch.argmax(test_predictions, dim=1))
    test_classification_report = compute_class_report(test_predictions_decoded, test_labels_decoded)
    logging.info(f"Test classification report: {test_classification_report}")

    if log_fn:
        log_fn(
            {
                "test/OBJ_class_f1": test_classification_report.filter(pl.col("category") == "OBJ")["f1-score"][0],
                "test/SUBJ_class_f1": test_classification_report.filter(pl.col("category") == "SUBJ")["f1-score"][0],
            }
        )

    return model, test_metrics_value["macro_f1"], test_predictions_decoded


def compute_token_stats(train_input_ids: torch.Tensor, val_input_ids: torch.Tensor, test_input_ids: torch.Tensor) -> None:
    """Compute token statistics."""

    # ModernBERT: 50283 is [PAD] (https://discuss.huggingface.co/t/modernbert-maskedlm-nan-training-loss/133951/5)
    tokenized_train_stats = pl.DataFrame(train_input_ids.numpy()).select(pl.all().ne(50283)).sum_horizontal().describe()
    logging.info(f"Tokenized train sequence length statistics: {tokenized_train_stats}")
    tokenized_val_stats = pl.DataFrame(val_input_ids.numpy()).select(pl.all().ne(50283)).sum_horizontal().describe()
    logging.info(f"Tokenized validation sequence length statistics: {tokenized_val_stats}")
    tokenized_test_stats = pl.DataFrame(test_input_ids.numpy()).select(pl.all().ne(50283)).sum_horizontal().describe()
    logging.info(f"Tokenized test sequence length statistics: {tokenized_test_stats}")