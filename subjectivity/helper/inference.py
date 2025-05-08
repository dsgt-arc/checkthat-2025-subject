import logging
import torch
import polars as pl


def predict_test_data(
    val_dataloader: torch.utils.data.DataLoader,
    best_model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Predict val data given best model
    # This is only possible because I use sequential sampler for val_dataloader
    logging.info("Predicting val data with best model.")
    predicted_labels = []
    true_labels = []
    for batch in val_dataloader:
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)
        # Inference
        with torch.no_grad():
            outputs = best_model(input_ids, attention_masks)
        predicted_labels.append(outputs)
        true_labels.append(labels)

    predicted_labels = torch.cat(predicted_labels)
    true_labels = torch.cat(true_labels)


    return predicted_labels, true_labels
