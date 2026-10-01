"""Continual learning metric tracker and lifelong evaluation matrix.

Computes catastrophic forgetting, backward transfer (BWT), forward transfer (FWT),
average accuracy across seen tasks, old-vs-new F1 trade-offs, and harmonic mean.
"""

from typing import Any, Dict, List, Optional, Tuple


class ContinualMetricsTracker:
    """Tracks class recall peaks, incremental task matrices, and continual transfer metrics."""

    def __init__(self) -> None:
        """Initializes empty tracking caches."""
        self.best_historical_recall: Dict[int, float] = {}
        # Performance matrix R[i][j] = metric on task j after training on task i (1-indexed)
        self.performance_matrix: Dict[str, Dict[int, Dict[int, float]]] = {
            "accuracy": {},
            "macro_f1": {},
        }
        self.task_summaries: List[Dict[str, Any]] = []

    def calculate_continual_metrics(
        self,
        overall_report: Dict[str, Any],
        current_task_id: int,
        classes_per_task: int = 2,
    ) -> Tuple[float, float, float, float]:
        """Calculates Average Forgetting, Old F1, New F1, and Harmonic Mean.

        Also updates the provided overall_report dictionary in-place with:
        'avg_forgetting', 'old_f1', 'new_f1', and 'harmonic_mean'.

        Args:
            overall_report: Dictionary returned by compute_classification_metrics.
            current_task_id: Current continual task index (1-based).
            classes_per_task: Number of novel classes introduced in each task.

        Returns:
            Tuple of (avg_forgetting, old_f1_avg, new_f1_avg, harmonic_mean).
        """
        current_num_classes = int(overall_report.get("num_classes_seen", 0))
        per_class = overall_report.get("per_class", {})

        # Old classes are those introduced before the current task
        num_old_classes = max(0, current_num_classes - classes_per_task)
        old_classes = list(range(num_old_classes))
        new_classes = list(range(num_old_classes, current_num_classes))

        # 1. Update historical recall and compute Average Forgetting for old classes
        forgetting_list: List[float] = []
        for c in range(current_num_classes):
            c_str = str(c)
            current_recall = (
                float(per_class[c_str]["recall"])
                if c_str in per_class and isinstance(per_class[c_str], dict)
                else 0.0
            )

            # Update historical best recall
            if c not in self.best_historical_recall:
                self.best_historical_recall[c] = current_recall
            else:
                self.best_historical_recall[c] = max(self.best_historical_recall[c], current_recall)

            # Forgetting only applies to old classes
            if c in old_classes:
                forgetting = self.best_historical_recall[c] - current_recall
                forgetting_list.append(max(0.0, forgetting))

        avg_forgetting = (
            float(sum(forgetting_list) / len(forgetting_list))
            if forgetting_list
            else 0.0
        )

        # 2. Compute Old F1, New F1 and Harmonic Mean
        old_f1s = [
            float(per_class[str(c)]["f1-score"])
            for c in old_classes
            if str(c) in per_class and isinstance(per_class[str(c)], dict)
        ]
        new_f1s = [
            float(per_class[str(c)]["f1-score"])
            for c in new_classes
            if str(c) in per_class and isinstance(per_class[str(c)], dict)
        ]

        old_f1_avg = float(sum(old_f1s) / len(old_f1s)) if old_f1s else 0.0
        new_f1_avg = float(sum(new_f1s) / len(new_f1s)) if new_f1s else 0.0

        harmonic_mean = float(
            (2.0 * old_f1_avg * new_f1_avg) / (old_f1_avg + new_f1_avg + 1e-8)
        )

        # Update report dictionary
        overall_report["avg_forgetting"] = avg_forgetting
        overall_report["old_f1"] = old_f1_avg
        overall_report["new_f1"] = new_f1_avg
        overall_report["harmonic_mean"] = harmonic_mean

        return avg_forgetting, old_f1_avg, new_f1_avg, harmonic_mean

    def record_task_metric(
        self,
        trained_task: int,
        evaluated_task: int,
        value: float,
        metric_name: str = "accuracy",
    ) -> None:
        """Records an element in the continual evaluation matrix R[trained_task][evaluated_task].

        Args:
            trained_task: Task index up to which the model has been trained (1-based).
            evaluated_task: Task dataset index on which the model was evaluated (1-based).
            value: Numerical performance score [0.0, 1.0].
            metric_name: Target metric identifier ('accuracy', 'macro_f1', etc.).
        """
        if metric_name not in self.performance_matrix:
            self.performance_matrix[metric_name] = {}
        if trained_task not in self.performance_matrix[metric_name]:
            self.performance_matrix[metric_name][trained_task] = {}

        self.performance_matrix[metric_name][trained_task][evaluated_task] = float(value)

    def compute_matrix_metrics(
        self,
        metric_name: str = "accuracy",
        num_tasks: Optional[int] = None,
    ) -> Dict[str, float]:
        """Computes Average Performance (A_T), Backward Transfer (BWT), and Forgetting Measure (FM).

        Formulations:
            A_T = (1 / T) * sum_{j=1}^T R_{T, j}
            BWT = (1 / (T - 1)) * sum_{j=1}^{T-1} (R_{T, j} - R_{j, j})
            FM  = (1 / (T - 1)) * sum_{j=1}^{T-1} max_{l in {j,...,T-1}} (R_{l, j} - R_{T, j})

        Args:
            metric_name: Metric name to evaluate ('accuracy', 'macro_f1').
            num_tasks: Total number of tasks T. If None, derived from matrix keys.

        Returns:
            Dictionary containing average_performance, bwt, and forgetting_measure.
        """
        mat = self.performance_matrix.get(metric_name, {})
        if not mat:
            return {"average_performance": 0.0, "bwt": 0.0, "forgetting_measure": 0.0}

        T = num_tasks if num_tasks is not None else max(mat.keys())
        if T not in mat:
            return {"average_performance": 0.0, "bwt": 0.0, "forgetting_measure": 0.0}

        # 1. Average performance A_T
        seen_evals = [mat[T][j] for j in range(1, T + 1) if j in mat[T]]
        a_t = sum(seen_evals) / len(seen_evals) if seen_evals else 0.0

        if T <= 1:
            return {"average_performance": a_t, "bwt": 0.0, "forgetting_measure": 0.0}

        # 2. Backward Transfer (BWT)
        bwt_diffs: List[float] = []
        for j in range(1, T):
            if j in mat[T] and j in mat.get(j, {}) and j in mat[j]:
                bwt_diffs.append(mat[T][j] - mat[j][j])
        bwt = sum(bwt_diffs) / len(bwt_diffs) if bwt_diffs else 0.0

        # 3. Forgetting Measure (FM)
        fm_diffs: List[float] = []
        for j in range(1, T):
            historical_max = max(mat[l][j] for l in range(j, T) if l in mat and j in mat[l])
            current_val = mat[T].get(j, 0.0)
            fm_diffs.append(max(0.0, historical_max - current_val))
        fm = sum(fm_diffs) / len(fm_diffs) if fm_diffs else 0.0

        return {
            "average_performance": float(a_t),
            "bwt": float(bwt),
            "forgetting_measure": float(fm),
        }

    def reset(self) -> None:
        """Resets all tracking states."""
        self.best_historical_recall.clear()
        for metric in self.performance_matrix:
            self.performance_matrix[metric].clear()
        self.task_summaries.clear()
