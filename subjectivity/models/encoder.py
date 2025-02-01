from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Any, Collection, Optional
import logging
import torch

class EncoderModel:
    def __init__(self, model_name: str, device_name: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device_name = device_name

        self.device = None
        self.batch_size: int = None

    def set_up_device(self) -> None:
        """Set up device to use for model."""
        if self.device_name is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else :
            self.device = torch.device(self.device_name)
        logging.info(f"Using device: '{self.device}'.")

    def init(self, context_window: int = 512, tokenizer_config: Optional[dict[str, Any]] = None, ) -> None:
        """Initialize Hugging Face encoder and tokenizer.
        Model is loaded to device."""

        if tokenizer_config is None:
            tokenizer_config = {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_config)
        self.model = AutoModel.from_pretrained(self.model_name)
        if self.device:
            self.model.to(self.device)
            logging.info(f"Model '{self.model_name}' loaded to device '{self.device}'.")
        self.context_window = context_window

        logging.info(f"Context window set to '{self.context_window}'.")

    def encode(self, sequence_collection: Collection, encode_settings: dict[str, Any]) -> None:
        """Encode a collection of sequences."""
        logging.info(f"Encoding text with '{self.tokenizer.__class__.__name__}'.")

        encoded_sequence = list()
        for sequence in sequence_collection:
            encoded_sequence.append(self.tokenizer.encode_plus(sequence, **encode_settings).to(self.device))

        return encoded_sequence

    def load_dataloader(self, dataset, batch_size: int = 20):
        """Load dataset into dataloader."""
        self.batch_size = batch_size
        self.data = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
        )

    def compute_embeddings(self) -> None:
        """Compute embeddings for dataset."""
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_dataloader() first.")
        if self.model is None:
            raise ValueError("Model not initialized yet. Call init() first.")

        # Ensure model is in evaluation mode
        self.model.eval()
        last_hidden_states = []
        forward_passes = len(self.data)
        logging.info(f"With batchsize '{self.batch_size}' a total of '{forward_passes}' number of forward passes are necessary.")

        for i, batch in enumerate(self.data):
            logging.info(f"Embedding batch {i}/{forward_passes}.")
            b_input_ids = batch["input_ids"].reshape(-1, self.context_window)
            b_input_mask = batch["attention_mask"].reshape(-1, self.context_window)

            with torch.no_grad():
                outputs = self.model(b_input_ids, b_input_mask)
                last_hidden_states.append(outputs.last_hidden_state.to("cpu").numpy())
        np_last_hidden_states = np.concatenate(last_hidden_states)
        logging.info(f"Embeddings computed with shape {np_last_hidden_states.shape}.")
        self.last_hidden_states = np_last_hidden_states

    def get_cls_embeddings(self) -> np.ndarray:
        """Get CLS embeddings."""
        return self.last_hidden_states[:, 0, :]
    
    def get_embeddings(self) -> np.ndarray:
        """Get all embeddings."""
        return self.last_hidden_states