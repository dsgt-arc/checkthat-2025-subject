import polars as pl
import pandas as pd
import numpy as np
import torch
from .run_config import RunConfig
from sklearn.metrics import classification_report, f1_score


def compute_class_report(labels: np.ndarray, predictions: np.ndarray) -> pl.DataFrame:
    """Compute classification metrics."""

    # Create stats of the class distribution in the labels and predictions
    unique_labels = np.unique(labels)
    count_labels = {}
    count_preds = {}
    for label in unique_labels:
        count_labels[str(label)] = np.sum(labels == label)
        count_preds[str(label)] = np.sum(predictions == label)
    count_labels["Total obs"] = len(labels)
    count_preds["Total obs"] = len(predictions)
    # Create a DataFrame from the counts
    counts_df = pl.concat(
        [
            pl.DataFrame(count_labels),
            pl.DataFrame(count_preds),
        ]
    ).transpose(column_names=["count-labels", "count-predictions"], include_header=True)

    metrics = classification_report(
        labels,
        predictions,
        output_dict=True,
        zero_division=np.nan,
    )
    # pandas necessary as polars transpose broken -> pyo3_runtime.PanicException
    metrics = pl.DataFrame(pd.DataFrame.from_dict(metrics).T.reset_index())
    metrics = metrics.rename({"index": "category"})
    metrics = metrics.with_columns(
        pl.col("precision").round(RunConfig.train["rounding_metrics"]),
        pl.col("recall").round(RunConfig.train["rounding_metrics"]),
        pl.col("f1-score").round(RunConfig.train["rounding_metrics"]),
        pl.col("support").round(RunConfig.train["rounding_metrics"]),
    )

    # Merge the counts DataFrame with the metrics DataFrame
    metrics = metrics.join(counts_df, left_on="category", right_on="column", how="left")
    return metrics


def accuracy_fn(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(preds, dim=1)
    return round((preds == labels).float().mean().item(), RunConfig.train["rounding_metrics"])


def macro_f1_fn(preds: torch.Tensor, labels: torch.Tensor) -> float:
    # measures macro-f1 score = 2 * ((precision * recall)/(precision + recall))
    preds = torch.argmax(preds, dim=1)
    return round(f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro"), RunConfig.train["rounding_metrics"])


def f1_macro(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    preds = torch.argmax(torch.nn.functional.softmax(preds, dim=1), dim=1)

    # Initialize variables to store true positives, false positives, and false negatives for each class
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    # Calculate true positives, false positives, and false negatives
    for pred, true_label in zip(preds, labels):
        if pred == true_label:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[true_label] += 1

    # Calculate precision, recall, and F1 score for each class
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)

    for i in range(num_classes):
        precision[i] = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        recall[i] = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    # Macro-average F1 score
    macro_f1 = torch.mean(f1)

    return macro_f1.item()


def f1_classwise(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[int, float]:
    preds = torch.argmax(torch.nn.functional.softmax(preds, dim=1), dim=1)

    # Initialize variables to store true positives, false positives, and false negatives for each class
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    # Calculate true positives, false positives, and false negatives
    for pred, true_label in zip(preds, labels):
        if pred == true_label:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[true_label] += 1

    # Calculate precision, recall, and F1 score for each class
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)

    classwise_f1 = {}

    for i in range(num_classes):
        precision[i] = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        recall[i] = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

        classwise_f1[i] = f1[i].item()

    return classwise_f1


def reweight(cls_num_tensor, beta=0.9999):
    """
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None

    cls_num_tensor = torch.tensor(cls_num_tensor)

    # Assuming Ni ≈ N for all classes
    beta_i = beta * torch.ones_like(cls_num_tensor)

    # Calculate effective number of samples for each class
    effective_num = (1.0 - torch.pow(beta_i, cls_num_tensor)) / (1.0 - beta_i)

    # Weight for each class is inversely proportional to effective number of samples
    per_cls_weights = 1.0 / effective_num

    # Normalize the weights so that sum equals the number of classes
    per_cls_weights = per_cls_weights / torch.sum(per_cls_weights) * len(cls_num_tensor)

    return per_cls_weights


class FocalLoss(torch.nn.Module):
    def __init__(self, weight: list[float] = None, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        pass
