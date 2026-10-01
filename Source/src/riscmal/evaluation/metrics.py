"""Classification metrics computation and multi-task evaluation engine.

Computes overall Accuracy, Macro/Weighted/Micro Precision, Recall, F1 score,
and Worst-Class Recall (min_c Recall_c) to quantify minority malware family retention.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor, Sequence[int]],
    y_pred: Union[np.ndarray, torch.Tensor, Sequence[int]],
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive multi-class classification metrics.

    Args:
        y_true: Ground truth target labels (integer IDs).
        y_pred: Predicted class labels (integer IDs).
        labels: Optional explicit sequence of class labels to evaluate. If None,
            inferred from the unique set of targets present in y_true.

    Returns:
        Dictionary containing overall accuracy, macro/weighted/micro precision,
        recall, f1, worst_class_recall, per_class breakdowns, and prediction arrays.
    """
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_true)

    if isinstance(y_pred, torch.Tensor):
        y_pred_np = y_pred.detach().cpu().numpy()
    else:
        y_pred_np = np.asarray(y_pred)

    if len(y_true_np) == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "worst_class_recall": 0.0,
            "per_class": {},
            "num_classes_seen": 0,
            "all_true": y_true_np,
            "all_pred": y_pred_np,
        }

    unique_classes = np.unique(y_true_np) if labels is None else np.asarray(labels)
    overall_acc = accuracy_score(y_true_np, y_pred_np)

    report_dict = classification_report(
        y_true_np,
        y_pred_np,
        labels=unique_classes,
        output_dict=True,
        zero_division=0,
    )

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=unique_classes,
        average="micro",
        zero_division=0,
    )

    # Extract individual per-class recall values
    per_class_recalls: List[float] = [
        float(report_dict[str(cls)]["recall"])
        for cls in unique_classes
        if str(cls) in report_dict and isinstance(report_dict[str(cls)], dict)
    ]

    # Worst-Class Recall: min_c Recall_c across all evaluated classes
    worst_class_recall = min(per_class_recalls) if per_class_recalls else 0.0

    return {
        "accuracy": float(overall_acc),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_precision": float(report_dict["weighted avg"]["precision"]),
        "weighted_recall": float(report_dict["weighted avg"]["recall"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "worst_class_recall": float(worst_class_recall),
        "per_class": {
            str(k): v
            for k, v in report_dict.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "num_classes_seen": int(len(unique_classes)),
        "all_true": y_true_np,
        "all_pred": y_pred_np,
    }


@torch.no_grad()
def evaluate_all_seen_tasks(
    model: torch.nn.Module,
    seen_task_loaders_dict: Dict[Union[str, int], Any],
    device: Union[torch.device, str],
    scenario_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluates the model across all tasks encountered up to the current incremental step.

    Handles both dictionary-based and tuple-based dataloaders, as well as unified
    and dynamic multi-column classifiers.

    Args:
        model: Evaluated PyTorch neural network.
        seen_task_loaders_dict: Dict mapping task identifiers to test DataLoader instances.
        device: Target execution device.
        scenario_name: Optional name of the continual strategy (e.g. for DER routing).

    Returns:
        Comprehensive metric dictionary produced by compute_classification_metrics.
    """
    model.eval()
    target_device = torch.device(device) if isinstance(device, str) else device
    all_preds: List[int] = []
    all_targets: List[int] = []

    for task_id, dataloader in seen_task_loaders_dict.items():
        for batch in dataloader:
            if isinstance(batch, dict):
                h = batch["header"].to(target_device)
                imp = batch["imports"].to(target_device)
                i1d = batch["img1d"].to(target_device)
                i2d = batch["img2d"].to(target_device)
                api = batch["apis"].to(target_device)
                y = batch["label"].to(target_device)
            elif isinstance(batch, (list, tuple)):
                h, imp, i1d, i2d, api, y = [item.to(target_device) for item in batch]
            else:
                raise TypeError(f"Unsupported batch type: {type(batch).__name__}")

            # Routing through model
            if scenario_name is not None and scenario_name.startswith("DER"):
                outputs, _ = model(h, imp, i1d, i2d, api)
            elif hasattr(model, "backbone") and hasattr(model, "classifier_head"):
                features = model.backbone(h, imp, i1d, i2d, api)
                outputs = model.classifier_head(features)
            else:
                out = model(h, imp, i1d, i2d, api)
                outputs = out[0] if isinstance(out, tuple) else out

            predicted = outputs.argmax(dim=1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    return compute_classification_metrics(all_targets, all_preds)
