import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging
from pathlib import Path

class Classifier(nn.Module):
    def __init__(self,
                 model_name: str,
                 labels_count: int = 2,
                 hidden_dim: int = 768,
                 mlp_dim: int = 500,
                 dropout_ratio: float = 0.1,
                 freeze_encoder: bool = False,
                ) -> None:
        super().__init__()

        self.model_name = model_name + '-classifier'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=False, output_attentions=False)

        """during the backward pass (during gradient computation), 
        the framework will manage memory more efficiently by recomputing 
        certain activations as needed rather than storing them all.
        """
        self.encoder.gradient_checkpointing_enable()

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, labels_count)
        )

        if freeze_encoder:
            logging.info(f"Freezing {self.encoder.__class__.__name__} Encoder layers")
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)
        logging.info(f"Using device: '{self.device}'.")

    # passing data and training
    def forward(self, tokens, masks):
        output = self.encoder(tokens, attention_mask=masks)
        cls_embedding = output.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_dim]
        dropout_output = self.dropout(cls_embedding)
        mlp_output = self.mlp(dropout_output)
        return mlp_output
    
    def save(self, save_path: str, ):
        """Save the model, tokenizer, and configuration."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.state_dict(), save_path / "model.pth")

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        # Save model config
        torch.save({
            "model_name": self.encoder.config._name_or_path,
            "labels_count": self.mlp[-1].out_features,
            "hidden_dim": self.mlp[0].in_features,
            "mlp_dim": self.mlp[0].out_features,
            "dropout_ratio": self.dropout.p if hasattr(self, "dropout") else 0.0,
            "freeze_encoder": all(not param.requires_grad for param in self.encoder.parameters()),
        }, save_path / "config.pth")

        logging.info(f"{self.model_name} saved successfully at {save_path}")

    @classmethod
    def load(cls, load_path: str):
        """Load the model, tokenizer, and configuration."""
        load_path = Path(load_path)

        # Load config
        config = torch.load(load_path / "config.pth")

        # Initialize model
        model = cls(
            model_name=config["model_name"],
            labels_count=config["labels_count"],
            hidden_dim=config["hidden_dim"],
            mlp_dim=config["mlp_dim"],
            dropout_ratio=config["dropout_ratio"],
            freeze_encoder=config["freeze_encoder"]
        )

        # Load model weights
        model.load_state_dict(torch.load(load_path / "model.pth", map_location=model.device))

        # Load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(load_path)

        logging.info(f"Model loaded successfully from {load_path}")

        return model
