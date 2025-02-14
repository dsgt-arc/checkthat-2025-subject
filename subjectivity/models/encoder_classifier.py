import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import logging


class Classifier(nn.Module):
    def __init__(self,
                 model_name: str,
                 labels_count: int = 2,
                 hidden_dim: int = 768,
                 mlp_dim: int = 500,
                 extras_dim: int = 100,
                 dropout_ratio: float = 0.1,
                 freeze_bert: bool = False,
                ) -> None:
        super().__init__()

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

        if freeze_bert:
            logging.info(f"Freezing {self.encoder.__class__.name} Encoder layers")
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)
        logging.info(f"Using device: '{self.device}'.")

    def forward(self, tokens, masks):
        output = self.encoder(tokens, attention_mask=masks)
        cls_embedding = output.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_dim]
        dropout_output = self.dropout(cls_embedding)
        mlp_output = self.mlp(dropout_output)
        return mlp_output
    
    def save(self, save_path):
        # Save your model state
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        # Load a pre-trained model
        self.model.load_state_dict(torch.load(load_path))
