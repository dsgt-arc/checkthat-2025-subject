import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    ModernBertForSequenceClassification,
    ModernBertConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import logging
from pathlib import Path
from tokenizers import pre_tokenizers, Regex
from peft import LoraConfig, TaskType, get_peft_model


class Tokenizer:
    def __init__(self, model_name: str, r2l: bool = False) -> None:
        if model_name == "uf-aice-lab/math-roberta":
            self.tokenzier_name = "roberta-large"
        else:
            self.tokenzier_name = model_name

        # Tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenzier_name)
        logging.info(f"Initialized '{model_name}' tokenizer.")
        if r2l:
            self.set_R2L_tokenizer()
            logging.info("Set tokenizer to split by R2L digits.")

    def set_R2L_tokenizer(self) -> None:
        """Set the tokenizer to split by R2L digits."""

        # Get existing pre-tokenizer steps
        existing_pre_tokenizer = self.tokenizer._tokenizer.pre_tokenizer

        # Add an extra step to the existing pre-tokenizer steps,
        self.tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                # Added step: split by R2L digits, Source: https://huggingface.co/spaces/huggingface/number-tokenization-blog
                pre_tokenizers.Split(pattern=Regex(r"\d{1,3}(?=(\d{3})*\b)"), behavior="isolated", invert=False),
                # Below: Existing steps from tokenizer
                existing_pre_tokenizer,
            ]
        )

    def save(self, save_path: str):
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        logging.info(f"{self.tokenzier_name} saved successfully at {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "Tokenizer":
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        logging.info(f"Tokenizer loeaded successfully from {load_path}")
        return tokenizer


class BaseClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
    ) -> None:
        super().__init__()
        self.model_name = model_name

    def to_device(self) -> None:
        # Setup common device
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.to(self.device)
        logging.info(f"Using device: '{self.device}'.")

    @classmethod
    def create(
        cls,
        model_name: str,
        labels_count: int = 2,
        hidden_dim: int = 768,
        mlp_dim: int = 500,
        dropout_ratio: float = 0.1,
        freeze_encoder: str = False,
        lora_rank: int | None = None,
        lora_alpha: float | None = None,
        **kwargs,
    ):
        """Factory method to create an instance of ClassifierPEFT if lora_rank and lora_alpha
        are provided, otherwise returns an instance of Classifier.
        """
        if lora_rank is not None and lora_alpha is not None:
            return ClassifierPEFT(
                model_name=model_name,
                labels_count=labels_count,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout_ratio=dropout_ratio,
                freeze_encoder=freeze_encoder,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                **kwargs,
            )
        else:
            return Classifier(
                model_name=model_name,
                labels_count=labels_count,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout_ratio=dropout_ratio,
                freeze_encoder=freeze_encoder,
                **kwargs,
            )

    def save(self, save_path: str):
        """Save the model, tokenizer, and configuration."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.state_dict(), save_path / "model.pth")

        # Save model config
        torch.save(
            {
                "model_name": self.encoder.config._name_or_path,
                "labels_count": self.mlp[-1].out_features,
                "hidden_dim": self.mlp[0].in_features,
                "mlp_dim": self.mlp[0].out_features,
                "dropout_ratio": self.dropout.p if hasattr(self, "dropout") else 0.0,
                "freeze_encoder": all(not param.requires_grad for param in self.encoder.parameters()),
            },
            save_path / "config.pth",
        )

        logging.info(f"{self.model_name} saved successfully at {save_path}")

    @classmethod
    def load(cls, load_path: str, peft: bool):
        """Load the model and configuration."""
        load_path = Path(load_path)

        # Load config
        config = torch.load(load_path / "config.pth")

        # Initialize model
        model = cls.create(
            model_name=config["model_name"],
            labels_count=config["labels_count"],
            hidden_dim=config["hidden_dim"],
            mlp_dim=config["mlp_dim"],
            dropout_ratio=config["dropout_ratio"],
            freeze_encoder=config.get("freeze_encoder"),  # Optional
            lora_alpha=config.get("lora_alpha"),  # Optional
            lora_rank=config.get("lora_rank"),  # Optional
        )

        # Load model weights
        model.load_state_dict(torch.load(load_path / "model.pth", map_location=model.device))

        logging.info(f"Model loaded successfully from {load_path}")

        return model


class Classifier(BaseClassifier):
    """Non-PEFT version: Regular classifier using an AutoModel encoder."""

    def __init__(
        self,
        model_name: str,
        labels_count: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout_ratio: float,
        freeze_encoder: str,
    ) -> None:
        super().__init__(model_name)

        # Encoder
        # no reference_compile, otherwise error when model.evaluation() using DataParallel see
        # https://huggingface.co/docs/transformers/v4.51.0/en/model_doc/modernbert#transformers.ModernBertConfig
        if "modern" in self.model_name:
            kwargs = {"reference_compile": False}
        else:
            kwargs = {}

        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True, **kwargs)
        logging.info(f"Initialzed '{model_name}' encoder backbone.")

        # Freeze encoder backbone if requested
        if freeze_encoder == "whole":
            logging.info(f"Freezing {model_name} Encoder layers")
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif freeze_encoder == "first_5_layers":
            logging.info("Freezing the first 5 layers of the encoder backbone.")
            for param in self.encoder.encoder.layer[0:5].parameters():
                param.requires_grad = False
        elif freeze_encoder is None:
            logging.info("No freezing of encoder backbone.")
        else:
            logging.error(f"Invalid value for 'freeze_encoder': {freeze_encoder}.")
            raise ValueError("Invalid value for 'freeze_encoder'.")

        # Classifier head with dropout and MLP
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, labels_count))

        # Move model to device
        self.to_device()

    def forward(self, tokens, masks):
        output = self.encoder(tokens, attention_mask=masks)

        if "modern" in self.model_name:
            # ModernBERT has no output["pooler_output"]
            cls_embedding = output.last_hidden_state[:, 0, :]
        else:
            # Replicating Venktesh code
            # Not: cls_embedding = cls_embedding = output.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_dim]
            cls_embedding = output["pooler_output"]

        dropout_output = self.dropout(cls_embedding)
        mlp_output = self.mlp(dropout_output)
        return mlp_output


class ClassifierPEFT(BaseClassifier):
    """PEFT-enabled classifier with optional LoRa modifications"""

    def __init__(
        self,
        model_name: str,
        labels_count: int = 2,
        hidden_dim: int = 768,
        mlp_dim: int = 500,  # Not used in standard ModernBertForSequenceClassification
        dropout_ratio: float = 0.1,
        freeze_encoder: str = False,
        lora_rank: int | None = None,
        lora_alpha: float | None = None,
    ) -> None:
        super().__init__(model_name)

        # Encoder given ModernBert configuration
        # no reference_compile, otherwise error when model.evaluation() using DataParallel see
        # https://huggingface.co/docs/transformers/v4.51.0/en/model_doc/modernbert#transformers.ModernBertConfig
        config = ModernBertConfig.from_pretrained(model_name, reference_compile=False)
        config.hidden_size = hidden_dim
        config.num_labels = labels_count
        config.classifier_activation = "relu"
        config.classifier_dropout = dropout_ratio

        # Load ModernBert with ClassificationHead given config
        self.mbc = ModernBertForSequenceClassification(config)
        logging.info(f"Initialzed '{model_name}'.")

        # PEFT/LoRa, see https://huggingface.co/docs/peft/en/quicktour
        if lora_alpha and lora_rank:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=dropout_ratio,
                target_modules="all-linear",
            )
            self.mbc = get_peft_model(self.mbc, peft_config)
            n_trainable_params, n_all_params = self.mbc.get_nb_trainable_parameters()
            logging.info(
                f"Using PEFT/LoRa with: trainable params {n_trainable_params:,} "
                + f"|| all params {n_all_params:,} || trainable% {round(n_trainable_params / n_all_params * 100, 4)}"
            )

        # Move model to device
        self.to_device()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        output = self.mbc(input_ids, attention_mask)
        return output.logits


class CustomModernBertForSequenceClassificationPEFT(ModernBertForSequenceClassification):
    """TODO: ModernBERT with custom classification head instead of standard classification head. Currently not used."""

    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.mlp_dim), nn.ReLU(), nn.Linear(config.mlp_dim, config.num_labels)
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Modify forward function to include your custom layers
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = outputs[0]  # last-layer hidden-state
        pooled_output = hidden_state[:, 0]  # take pooled output from the first token

        dropout_output = self.dropout(pooled_output)
        logits = self.mlp(dropout_output)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
