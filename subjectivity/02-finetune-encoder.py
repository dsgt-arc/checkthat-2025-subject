import polars as pl
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import logging
import argparse

from helper.logger import set_up_log
from helper.data_store import DataStore
from helper.run_config import RunConfig
from models.encoder_classifier import Classifier
from torch.utils.data import DataLoader, RandomSampler
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


max_train_features_length = max([len(f) for f in train_features])


def check_data_availability(path: Path) -> bool:
    """Check if data is already computed."""
    return path.exists()


def read_train_and_val_data(rc: RunConfig) -> pl.DataFrame:
    """Read train and validation data."""
    logging.info(f"Reading train and validation data from '{rc.data['train_en']}' and '{rc.data['val_en']}'.")

    ds = DataStore(location=rc.data["dir"])
    ds.read_csv_data(rc.data["train_en"], separator="\t")
    df_train = ds.get_data()

    ds.read_csv_data(rc.data["val_en"], separator="\t")
    df_val = ds.get_data()

    return df_train, df_val


def encode_labels(df_train: pl.DataFrame, df_val: pl.DataFrame) -> pl.DataFrame:
    """Encode labels."""
    LE = LabelEncoder()

    train_label_encoded = LE.fit_transform(df_train["label"].to_list())
    val_label_encoded = LE.transform(df_val["label"].to_list())

    df_train = df_train.with_columns(pl.Series(name="label_encoded", values=train_label_encoded))
    df_val = df_val.with_columns(pl.Series(name="label_encoded", values=val_label_encoded))

    return df_train, df_val


def tokenize_data(model, df_train, df_val):
    """Tokenize data."""

    # TODO: max_length:   max_length=min([8192, max_train_features_length, sequence_length]),
    # TODO: split tokens and attention masks
    train_tokens = model.tokenizer(
        df_train["sentence"].to_list(),
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    val_tokens = model.tokenizer(df_val["sentence"].to_list(), padding=True, truncation=True, return_tensors="pt", return_attention_mask=True)

    train_masks = train_tokens["attention_mask"]
    val_masks = val_tokens["attention_mask"]

    return train_tokens, val_tokens, train_masks, val_masks


def set_up_dataloaders(batch_size, seed, train_tokens, val_tokens, train_masks, val_masks, df_train, df_val):
    """Set up dataloaders."""
    train_dataset = pl.TensorDataset(train_tokens["input_ids"], train_masks, df_train["label_encoded"])
    val_dataset = pl.TensorDataset(val_tokens["input_ids"], val_masks, df_val["label_encoded"])

    # Set a seed for reproducibility in random sampler generation
    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, generator=generator), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset, generator=generator), batch_size=batch_size)

    return train_dataloader, val_dataloader

def set_up_optimizer(model, learning_rate, eps, epochs):
    """Set up optimizer."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=eps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / epochs)
    return optimizer, scheduler

def train(model, optimizer, scheduler, loss, epochs, train_dataloader, val_dataloader):
    """Train model."""
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            # TODO: do i need both?
            optimizer.zero_grad()
            model.zero_grad()
            input_ids = batch[0].to(model.device)
            attention_mask = batch[1].to(model.device)
            labels = batch[2].to(model.device)
            outputs = model(input_ids, attention_mask)
            loss_value = loss(outputs, labels)
            loss_value.backward()
            optimizer.step()
        scheduler.step()
        logging.info(f"Epoch {epoch} finished. Loss: {loss_value.item()}")

        model.eval()
        for batch in val_dataloader:
            input_ids = batch[0].to(model.device)
            attention_mask = batch[1].to(model.device)
            labels = batch[2].to(model.device)
            outputs = model(input_ids, attention_mask)
            loss_value = loss(outputs, labels)
        logging.info(f"Validation loss: {loss_value.item()}")


def main() -> int:
    set_up_log()
    logging.info("Start Fine-Tuning Encoder")
    try:
        args = init_args_parser()
        logging.info(f"Reading config {args.config_path}")
        rc = RunConfig(Path(args.config_path))
        rc.load_config()

        df_train, df_val = read_train_and_val_data(rc)
        df_train, df_val = encode_labels(df_train, df_val)
        num_classes = len(df_train["label"].unique())

        # Load encoder classifer to fine-tune if not already fine-tuned
        encoder_model_name = rc.encoder_model["name"]
        # TODO: check weight path
        """weights_path = Path(ds_train.location) / Path(rc.data['train_en_embedding'] + '_' + encoder_model_name.replace("/", "-") + '.npy') 
        if check_data_availability(weights_path) & (not args.force):
            logging.info(f"Weights of model '{encoder_model_name}' already exist in '{weights_path}'. Loading matrix.")
            #model =
        else:
        """
        model = Classifier(model_name=encoder_model_name, labels_count=num_classes)

        # Use tokenizer of model to prepare data furhter and load into dataloaders
        train_tokens, val_tokens, train_masks, val_masks = tokenize_data(model, df_train, df_val)
        train_dataloader, val_dataloader = set_up_dataloaders(
            rc.train["batch_size"], rc.train["seed"], train_tokens, val_tokens, train_masks, val_masks, df_train, df_val
        )

        # Set up optimizer
        optimizer, scheduler = set_up_optimizer(model, rc.train["learning_rate"], rc.train["eps"], rc.train["epochs"])

        # Set up loss function
        loss = torch.nn.CrossEntropyLoss()

        # Train model
        train(model, optimizer, loss, train_dataloader, val_dataloader)

        # Save weeights of fine-tuned model

        logging.info("Finished fine-tuning.")
        return 0
    except Exception:
        logging.exception("Fine-Tuning failed", stack_info=True)
        return 1


main()
