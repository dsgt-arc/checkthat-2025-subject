import polars as pl
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import logging
import argparse

from helper.logger import set_up_log
from helper.data_store import DataStore
from helper.run_config import RunConfig
from models.encoder_classifier import Classifier
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch import optim
import torch


def init_args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-Tune Encoder")

    parser.add_argument("--config_path", type=str, default="config/run_config.yml", help="Config name")
    parser.add_argument(
        "--force",
        "-f",
        default=False,
        action="store_true",
        help="Force recomputation of embedding",
    )

    return parser.parse_args()


# TODO: length for tokenizer
# max_train_features_length = max([len(f) for f in train_features])


def check_data_availability(path: Path) -> bool:
    """Check if data is already computed."""
    return path.exists()


def read_train_and_val_data(rc: RunConfig) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Read train and validation data."""
    logging.info(f"Reading train and validation data from '{rc.data['train_en']}' and '{rc.data['val_en']}'.")

    ds = DataStore(location=rc.data["dir"])
    ds.read_csv_data(rc.data["train_en"], separator="\t")
    df_train = ds.get_data()

    if rc.data["val_en"] is None:
        logging.info("No validation data available. Need to split train.")
        df_val = None
    else:
        ds.read_csv_data(rc.data["val_en"], separator="\t")
        df_val = ds.get_data()

    return df_train, df_val


def train_val_split(df: pl.DataFrame, rc: RunConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split data into train and validation set."""
    split_ratio = rc.train["train_test_split"]
    logging.info(f"Splitting data with ratio '{split_ratio}'.")
    # Shuffle the DataFrame
    df = df.sample(fraction=1.0, shuffle=True, seed=rc.train["seed"])
    # Compute split index
    split_idx = int(split_ratio * df.height)
    # Train/Test split
    df_train = df.slice(0, split_idx)
    df_val = df.slice(split_idx, df.height - split_idx)
    return df_train, df_val


def encode_labels(df_train: pl.DataFrame, df_val: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Encode labels."""

    logging.info("Encoding labels.")
    LE = LabelEncoder()

    train_label_encoded = LE.fit_transform(df_train["label"].to_list())
    val_label_encoded = LE.transform(df_val["label"].to_list())

    df_train = df_train.with_columns(pl.Series(name="label_encoded", values=train_label_encoded))
    df_val = df_val.with_columns(pl.Series(name="label_encoded", values=val_label_encoded))

    return df_train, df_val


def tokenize_data(
    model: torch.nn.Module, df_train: pl.DataFrame, df_val: pl.DataFrame, rc: RunConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize data."""

    logging.info("Tokenizing data.")

    # TODO: max_length:   max_length=min([8192, max_train_features_length, sequence_length]),
    # TODO: split tokens and attention masks
    train_tokens = model.tokenizer(
        df_train["sentence"].to_list(),
        padding=rc.encoder_model["padding"],
        truncation=rc.encoder_model["truncation"],
        return_tensors=rc.encoder_model["return_tensors"],
        return_attention_mask=rc.encoder_model["return_attention_mask"],
        max_length=rc.encoder_model["max_length"],
    )
    val_tokens = model.tokenizer(
        df_val["sentence"].to_list(), padding=True, truncation=True, return_tensors="pt", return_attention_mask=True
    )

    train_input_ids = train_tokens["input_ids"]
    val_input_ids = val_tokens["input_ids"]
    train_masks = train_tokens["attention_mask"]
    val_masks = val_tokens["attention_mask"]

    return train_input_ids, val_input_ids, train_masks, val_masks


def set_up_dataloaders(
    batch_size: int,
    seed: int,
    train_input_ids: torch.Tensor,
    val_input_ids: torch.Tensor,
    train_masks: torch.Tensor,
    val_masks: torch.Tensor,
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
) -> tuple[DataLoader, DataLoader]:
    """Set up dataloaders."""

    logging.info("Setting up dataloaders for training.")

    train_labels = torch.tensor(df_train["label_encoded"])
    val_labels = torch.tensor(df_val["label_encoded"])

    # Wrapping into TensorDataset so that each sample will be retrieved by indexing tensors along the first dimension
    train_dataset = TensorDataset(train_input_ids, train_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_masks, val_labels)

    # Set a seed for reproducibility in random sampler generation
    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, generator=generator), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset, generator=generator), batch_size=batch_size)

    return train_dataloader, val_dataloader


def set_up_optimizer(
    model, learning_rate: float, eps: float, epochs: int
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """Set up optimizer."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=eps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / epochs)
    return optimizer, scheduler


def accuracy_fn(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(torch.nn.functional.softmax(preds, dim=1), dim=1)
    return (preds == labels).float().mean().item()


def train(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    loss: torch.nn.CrossEntropyLoss,
    metric_fn,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int = 100,
    early_stopping_patience: int = 5,
):
    """Train model with logging, early stopping, and configurable accuracy metrics."""

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_samples = 0

        # Training loop
        for batch in train_dataloader:
            optimizer.zero_grad()  # Reset gradients

            input_ids = batch[0].to(model.device)
            attention_mask = batch[1].to(model.device)
            labels = batch[2].to(model.device)

            outputs = model(input_ids, attention_mask)
            loss_value = loss(outputs, labels)

            loss_value.backward()
            optimizer.step()

            total_train_loss += loss_value.item()
            total_train_samples += input_ids.size(0)

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch[0].to(model.device)
                attention_mask = batch[1].to(model.device)
                labels = batch[2].to(model.device)

                outputs = model(input_ids, attention_mask)
                loss_value = loss(outputs, labels)

                total_val_loss += loss_value.item()
                total_val_samples += input_ids.size(0)

                if metric_fn:  # Store outputs for metric evaluation
                    all_preds.append(outputs)
                    all_labels.append(labels)

        avg_val_loss = total_val_loss / len(val_dataloader)

        # Compute metric if a function is provided
        metric_value = None
        if metric_fn:
            metric_value = metric_fn(torch.cat(all_preds), torch.cat(all_labels))
            logging.info(
                f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Metric: {metric_value:.4f}"
            )
        else:
            logging.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info("Early stopping triggered.")
                break  # Stop training

    logging.info("Training complete.")

    return model


def main() -> int:
    set_up_log()
    logging.info("Start Fine-Tuning Encoder")
    try:
        args = init_args_parser()
        logging.info(f"Reading config {args.config_path}")
        rc = RunConfig(Path(args.config_path))
        rc.load_config()

        df_train, df_val = read_train_and_val_data(rc)
        if df_val is None:
            # Split train data
            split_ratio = rc.train["train_test_split"]
            logging.info(f"Splitting train data with ratio '{split_ratio}'.")
            df_train, df_val = train_val_split(df_train, rc)

        df_train, df_val = encode_labels(df_train, df_val)
        num_classes = len(df_train["label"].unique())

        # Load encoder classifer to fine-tune if not already fine-tuned
        # TODO: Do we need this?
        encoder_model_name = rc.encoder_model["name"]
        model_path = Path(rc.data['dir']) / Path(rc.data['fine_tuned_model_path'] + encoder_model_name.replace("/", "-") + '-classifier' + '.pth') 
        if check_data_availability(model_path) & (not args.force):
            logging.info(f"Weights of model '{encoder_model_name}' already exist in '{model_path}'. Loading.")
            model = Classifier.load(model_path)
        else:
            model = Classifier(
                model_name=encoder_model_name,
                labels_count=num_classes,
                freeze_encoder=rc.encoder_model["freeze_encoder"],
                dropout_ratio=rc.encoder_model["dropout_ratio"],
                hidden_dim=rc.encoder_model["hidden_dim"],
                mlp_dim=rc.encoder_model["mlp_dim"],
            )

        # Use tokenizer of model to prepare data furhter and load into dataloaders
        train_input_ids, val_input_ids, train_masks, val_masks = tokenize_data(model, df_train, df_val, rc)
        train_dataloader, val_dataloader = set_up_dataloaders(
            rc.train["batch_size"], rc.train["seed"], train_input_ids, val_input_ids, train_masks, val_masks, df_train, df_val
        )

        # Set up optimizer
        optimizer, scheduler = set_up_optimizer(model, rc.train["learning_rate"], rc.train["eps"], rc.train["epochs"])

        # Set up loss function
        loss = torch.nn.CrossEntropyLoss()

        # Train model
        best_model = train(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            metric_fn=accuracy_fn,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=rc.train["epochs"],
            early_stopping_patience=rc.train["early_stopping_patience"],
        )

        # Save weights of fine-tuned model
        best_model.save(model_path)

        logging.info("Finished fine-tuning.")
        return 0
    except Exception:
        logging.exception("Fine-Tuning failed", stack_info=True)
        return 1


main()
