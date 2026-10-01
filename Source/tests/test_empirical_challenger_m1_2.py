"""Empirical Challenger Test Suite for Milestone 1.2 Evaluation & Utilities.

Empirically tests:
1. `riscmal.evaluation.metrics`:
   - Exact confusion matrix ground truth verification against analytical formulas and scikit-learn
   - Accuracy, Precision (Macro/Micro/Weighted), Recall (Macro/Micro/Weighted), F1 (Macro/Micro/Weighted)
   - Worst-Class Recall (min_c Recall_c)
   - Input type invariance (torch.Tensor, np.ndarray, Python list)
   - Edge cases: Zero division, zero support, empty inputs, extreme imbalance, novel predictions
2. `riscmal.evaluation.tracker`:
   - 3-Task continual evaluation sequence simulation
   - Performance matrix R[i][j], Average Performance A_T, Backward Transfer BWT, Forgetting Measure FM
   - Class-level Average Forgetting, Old F1, New F1, Harmonic Mean
   - Non-monotonic intermediate improvement stress-test (historical max verification)
   - Edge cases: Single task (T=1), empty matrix, tracker reset, custom classes_per_task
   - Adversarial Failure Mode 1: Sparse matrix evaluations crash `compute_matrix_metrics` with empty max() ValueError
   - Adversarial Failure Mode 2: Unused `current_task_id` causes false old-class splitting on base task if classes > classes_per_task
3. `riscmal.utils.seed`:
   - Deterministic reproducibility of seed 42 across random, numpy, and torch
   - Multi-distribution consistency (uniform, normal, integer, permutation)
   - Entropy/variation under differing seeds
   - Environment and backend flags (PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG, cuDNN)
   - seed_worker function for DataLoader workers & DataLoader shuffle repeatability
"""

import os
import random
from typing import Dict, List
import numpy as np
import pytest
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

from riscmal.evaluation.metrics import compute_classification_metrics
from riscmal.evaluation.tracker import ContinualMetricsTracker
from riscmal.utils.seed import set_deterministic_seed, seed_worker


# ==============================================================================
# SECTION 1: EMPIRICAL VERIFICATION OF METRICS.PY
# ==============================================================================

class TestMetricsEmpirical:
    """Rigorous empirical validation of compute_classification_metrics."""

    def test_metrics_against_exact_confusion_matrix_and_sklearn(self):
        """Validates all metrics against an exact 3-class confusion matrix and manual math.

        Confusion Matrix Design:
        -------------------------
                    Predicted
                  0     1     2    Total (Support)
        True 0   10     2     0      12
        True 1    5     5     0      10
        True 2    8     0     2      10
        Total    23     7     2      32
        """
        y_true = [0] * 12 + [1] * 10 + [2] * 10
        y_pred = [0] * 10 + [1] * 2 + [0] * 5 + [1] * 5 + [0] * 8 + [2] * 2

        # 1. Manual analytical calculations
        expected_acc = 17.0 / 32.0  # 0.53125
        rec_0 = 10.0 / 12.0
        rec_1 = 5.0 / 10.0
        rec_2 = 2.0 / 10.0
        expected_worst_recall = min(rec_0, rec_1, rec_2)  # 0.20

        prec_0 = 10.0 / 23.0
        prec_1 = 5.0 / 7.0
        prec_2 = 2.0 / 2.0  # 1.0

        f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0)  # 20 / 35 = 4/7
        f1_1 = 2 * (prec_1 * rec_1) / (prec_1 + rec_1)  # 10 / 17
        f1_2 = 2 * (prec_2 * rec_2) / (prec_2 + rec_2)  # 1 / 3

        expected_macro_prec = (prec_0 + prec_1 + prec_2) / 3.0
        expected_macro_rec = (rec_0 + rec_1 + rec_2) / 3.0
        expected_macro_f1 = (f1_0 + f1_1 + f1_2) / 3.0

        # Weighted metrics
        w_0, w_1, w_2 = 12.0 / 32.0, 10.0 / 32.0, 10.0 / 32.0
        expected_weighted_prec = w_0 * prec_0 + w_1 * prec_1 + w_2 * prec_2
        expected_weighted_rec = w_0 * rec_0 + w_1 * rec_1 + w_2 * rec_2
        expected_weighted_f1 = w_0 * f1_0 + w_1 * f1_1 + w_2 * f1_2

        # 2. Scikit-learn reference
        sk_acc = accuracy_score(y_true, y_pred)
        sk_p_macro, sk_r_macro, sk_f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        sk_p_micro, sk_r_micro, sk_f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        sk_p_weighted, sk_r_weighted, sk_f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        _, sk_per_class_rec, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # 3. Execution under test
        res = compute_classification_metrics(y_true, y_pred)

        # Assertions against analytical ground truth
        assert abs(res["accuracy"] - expected_acc) < 1e-6
        assert abs(res["macro_precision"] - expected_macro_prec) < 1e-6
        assert abs(res["macro_recall"] - expected_macro_rec) < 1e-6
        assert abs(res["macro_f1"] - expected_macro_f1) < 1e-6
        assert abs(res["worst_class_recall"] - expected_worst_recall) < 1e-6
        assert abs(res["weighted_precision"] - expected_weighted_prec) < 1e-6
        assert abs(res["weighted_recall"] - expected_weighted_rec) < 1e-6
        assert abs(res["weighted_f1"] - expected_weighted_f1) < 1e-6

        # Assertions against scikit-learn
        assert abs(res["accuracy"] - sk_acc) < 1e-6
        assert abs(res["macro_precision"] - sk_p_macro) < 1e-6
        assert abs(res["macro_recall"] - sk_r_macro) < 1e-6
        assert abs(res["macro_f1"] - sk_f1_macro) < 1e-6
        assert abs(res["micro_precision"] - sk_p_micro) < 1e-6
        assert abs(res["micro_recall"] - sk_r_micro) < 1e-6
        assert abs(res["micro_f1"] - sk_f1_micro) < 1e-6
        assert abs(res["weighted_precision"] - sk_p_weighted) < 1e-6
        assert abs(res["weighted_recall"] - sk_r_weighted) < 1e-6
        assert abs(res["weighted_f1"] - sk_f1_weighted) < 1e-6
        assert abs(res["worst_class_recall"] - float(np.min(sk_per_class_rec))) < 1e-6

        # Check per-class values
        assert abs(res["per_class"]["0"]["recall"] - rec_0) < 1e-5
        assert abs(res["per_class"]["1"]["recall"] - rec_1) < 1e-5
        assert abs(res["per_class"]["2"]["recall"] - rec_2) < 1e-5
        assert res["num_classes_seen"] == 3

    def test_metrics_input_type_invariance(self):
        """Verifies that torch.Tensor, np.ndarray, and list yield bit-exact identical metrics."""
        y_true_list = [0, 1, 2, 0, 1, 2, 0, 1]
        y_pred_list = [0, 1, 1, 0, 2, 2, 0, 1]

        res_list = compute_classification_metrics(y_true_list, y_pred_list)
        res_np = compute_classification_metrics(np.array(y_true_list), np.array(y_pred_list))
        res_torch = compute_classification_metrics(
            torch.tensor(y_true_list), torch.tensor(y_pred_list)
        )

        for key in ["accuracy", "macro_f1", "macro_recall", "worst_class_recall", "micro_f1"]:
            assert res_list[key] == res_np[key]
            assert res_list[key] == res_torch[key]

    def test_metrics_extreme_minority_imbalance(self):
        """Tests that minority family failure is immediately reflected in worst_class_recall."""
        # 1000 samples of class 0, 1 sample of class 1 (severe 1000:1 malware imbalance)
        y_true = [0] * 1000 + [1] * 1
        y_pred = [0] * 1001  # Classifier predicts majority class for everything

        res = compute_classification_metrics(y_true, y_pred)
        assert res["accuracy"] == 1000.0 / 1001.0  # > 99.9% accuracy
        assert res["per_class"]["1"]["recall"] == 0.0  # Minority class completely dropped
        assert res["worst_class_recall"] == 0.0  # Catches the critical failure
        assert res["per_class"]["0"]["recall"] == 1.0

    def test_metrics_zero_division_and_unseen_prediction(self):
        """Tests safe handling when predictions predict classes never occurring in ground truth."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 2, 1, 1, 2]  # Class 2 was predicted, but no true class 2 exists

        # Evaluated on true labels [0, 1]
        res = compute_classification_metrics(y_true, y_pred)
        assert res["num_classes_seen"] == 2
        assert "2" not in res["per_class"]
        assert res["per_class"]["0"]["recall"] == 2.0 / 3.0
        assert res["per_class"]["1"]["recall"] == 2.0 / 3.0
        assert abs(res["worst_class_recall"] - 2.0 / 3.0) < 1e-6

    def test_metrics_empty_inputs(self):
        """Tests boundary edge case with empty sequences."""
        res = compute_classification_metrics([], [])
        assert res["accuracy"] == 0.0
        assert res["macro_precision"] == 0.0
        assert res["macro_recall"] == 0.0
        assert res["macro_f1"] == 0.0
        assert res["worst_class_recall"] == 0.0
        assert res["num_classes_seen"] == 0


# ==============================================================================
# SECTION 2: EMPIRICAL VERIFICATION OF TRACKER.PY
# ==============================================================================

class TestTrackerEmpirical:
    """Rigorous empirical validation of ContinualMetricsTracker across 3 tasks."""

    def test_continual_tracker_3_task_simulation(self):
        """Simulates an exact 3-task continual evaluation sequence.

        Task protocol:
        - Task 1: Classes [0, 1]
        - Task 2: Classes [2, 3] (old: [0, 1])
        - Task 3: Classes [4, 5] (old: [0, 1, 2, 3])

        Evaluation Performance Matrix R[trained_task][evaluated_task] (Accuracy):
        - Task 1 trained:
            R[1][1] = 0.85
        - Task 2 trained:
            R[2][1] = 0.65  (backward transfer on task 1: 0.65 - 0.85 = -0.20)
            R[2][2] = 0.80
        - Task 3 trained:
            R[3][1] = 0.55  (further degradation on task 1: 0.55 - 0.85 = -0.30)
            R[3][2] = 0.65  (degradation on task 2: 0.65 - 0.80 = -0.15)
            R[3][3] = 0.82

        Theoretical Values at T=3:
        - Average Accuracy A_3 = (0.55 + 0.65 + 0.82) / 3 = 2.02 / 3 ≈ 0.673333
        - Backward Transfer BWT = ((0.55 - 0.85) + (0.65 - 0.80)) / 2 = (-0.30 - 0.15) / 2 = -0.225
        - Forgetting Measure FM:
            Task 1: max(R[1][1], R[2][1]) - R[3][1] = max(0.85, 0.65) - 0.55 = 0.30
            Task 2: max(R[2][2]) - R[3][2] = 0.80 - 0.65 = 0.15
            FM = (0.30 + 0.15) / 2 = 0.225
        """
        tracker = ContinualMetricsTracker()

        # Step 1: Train on Task 1
        tracker.record_task_metric(trained_task=1, evaluated_task=1, value=0.85, metric_name="accuracy")
        report_t1 = {
            "num_classes_seen": 2,
            "per_class": {
                "0": {"recall": 0.90, "f1-score": 0.88},
                "1": {"recall": 0.80, "f1-score": 0.82},
            },
        }
        f1_step, old_f1_step, new_f1_step, h_mean_step = tracker.calculate_continual_metrics(
            report_t1, current_task_id=1, classes_per_task=2
        )
        assert f1_step == 0.0, "Task 1 should have zero forgetting"
        assert old_f1_step == 0.0, "Task 1 has no old classes"
        assert abs(new_f1_step - 0.85) < 1e-6, "New F1 should be average of classes 0, 1"

        # Check matrix metrics at T=1
        m1 = tracker.compute_matrix_metrics(metric_name="accuracy", num_tasks=1)
        assert abs(m1["average_performance"] - 0.85) < 1e-6
        assert m1["bwt"] == 0.0
        assert m1["forgetting_measure"] == 0.0

        # Step 2: Train on Task 2
        tracker.record_task_metric(trained_task=2, evaluated_task=1, value=0.65, metric_name="accuracy")
        tracker.record_task_metric(trained_task=2, evaluated_task=2, value=0.80, metric_name="accuracy")
        report_t2 = {
            "num_classes_seen": 4,
            "per_class": {
                "0": {"recall": 0.70, "f1-score": 0.72},  # Drop from 0.90 -> forgetting = 0.20
                "1": {"recall": 0.60, "f1-score": 0.64},  # Drop from 0.80 -> forgetting = 0.20
                "2": {"recall": 0.85, "f1-score": 0.84},  # New class
                "3": {"recall": 0.75, "f1-score": 0.76},  # New class
            },
        }
        f2_step, old_f1_t2, new_f1_t2, h_mean_t2 = tracker.calculate_continual_metrics(
            report_t2, current_task_id=2, classes_per_task=2
        )
        # Old classes [0, 1] average forgetting: (0.20 + 0.20) / 2 = 0.20
        assert abs(f2_step - 0.20) < 1e-6
        assert abs(old_f1_t2 - 0.68) < 1e-6  # (0.72 + 0.64) / 2
        assert abs(new_f1_t2 - 0.80) < 1e-6  # (0.84 + 0.76) / 2
        expected_h2 = (2.0 * 0.68 * 0.80) / (0.68 + 0.80)
        assert abs(h_mean_t2 - expected_h2) < 1e-5

        # Check matrix metrics at T=2
        m2 = tracker.compute_matrix_metrics(metric_name="accuracy", num_tasks=2)
        assert abs(m2["average_performance"] - (0.65 + 0.80) / 2.0) < 1e-6
        assert abs(m2["bwt"] - (0.65 - 0.85)) < 1e-6  # -0.20
        assert abs(m2["forgetting_measure"] - 0.20) < 1e-6

        # Step 3: Train on Task 3
        tracker.record_task_metric(trained_task=3, evaluated_task=1, value=0.55, metric_name="accuracy")
        tracker.record_task_metric(trained_task=3, evaluated_task=2, value=0.65, metric_name="accuracy")
        tracker.record_task_metric(trained_task=3, evaluated_task=3, value=0.82, metric_name="accuracy")
        report_t3 = {
            "num_classes_seen": 6,
            "per_class": {
                "0": {"recall": 0.60, "f1-score": 0.62},  # Drop from 0.90 -> forgetting = 0.30
                "1": {"recall": 0.50, "f1-score": 0.52},  # Drop from 0.80 -> forgetting = 0.30
                "2": {"recall": 0.70, "f1-score": 0.70},  # Drop from 0.85 -> forgetting = 0.15
                "3": {"recall": 0.60, "f1-score": 0.62},  # Drop from 0.75 -> forgetting = 0.15
                "4": {"recall": 0.80, "f1-score": 0.80},  # New
                "5": {"recall": 0.80, "f1-score": 0.80},  # New
            },
        }
        f3_step, old_f1_t3, new_f1_t3, h_mean_t3 = tracker.calculate_continual_metrics(
            report_t3, current_task_id=3, classes_per_task=2
        )
        # Old classes [0, 1, 2, 3] forgetting: (0.30 + 0.30 + 0.15 + 0.15) / 4 = 0.90 / 4 = 0.225
        assert abs(f3_step - 0.225) < 1e-6
        assert abs(old_f1_t3 - (0.62 + 0.52 + 0.70 + 0.62) / 4.0) < 1e-6
        assert abs(new_f1_t3 - 0.80) < 1e-6

        # Step 4: Validate Continual Matrix Metrics at T=3
        m3 = tracker.compute_matrix_metrics(metric_name="accuracy", num_tasks=3)
        expected_a3 = (0.55 + 0.65 + 0.82) / 3.0
        expected_bwt = ((0.55 - 0.85) + (0.65 - 0.80)) / 2.0  # -0.225
        expected_fm = 0.225

        assert abs(m3["average_performance"] - expected_a3) < 1e-6
        assert abs(m3["bwt"] - expected_bwt) < 1e-6
        assert abs(m3["forgetting_measure"] - expected_fm) < 1e-6

    def test_tracker_non_monotonic_improvement_historical_peak(self):
        """Stress-tests FM tracking when performance on an old task improves at an intermediate task.

        Scenario:
        - R[1][1] = 0.70
        - R[2][1] = 0.85 (Task 2 training IMPROVED task 1 performance!)
        - R[3][1] = 0.60
        - R[2][2] = 0.80
        - R[3][2] = 0.70
        - R[3][3] = 0.90

        At T=3 for Task 1:
        - BWT compares R[3][1] to R[1][1]: 0.60 - 0.70 = -0.10
        - FM compares R[3][1] to max(R[1][1], R[2][1]) = 0.85:
          0.85 - 0.60 = 0.25!
        Here BWT != -FM!
        BWT = ((-0.10) + (0.70 - 0.80)) / 2 = (-0.10 - 0.10) / 2 = -0.10
        FM  = ((0.25) + (0.80 - 0.70)) / 2 = (0.25 + 0.10) / 2 = 0.175
        """
        tracker = ContinualMetricsTracker()
        tracker.record_task_metric(1, 1, 0.70)
        tracker.record_task_metric(2, 1, 0.85)
        tracker.record_task_metric(2, 2, 0.80)
        tracker.record_task_metric(3, 1, 0.60)
        tracker.record_task_metric(3, 2, 0.70)
        tracker.record_task_metric(3, 3, 0.90)

        metrics = tracker.compute_matrix_metrics(num_tasks=3)
        assert abs(metrics["bwt"] - (-0.10)) < 1e-6
        assert abs(metrics["forgetting_measure"] - 0.175) < 1e-6
        assert metrics["bwt"] != -metrics["forgetting_measure"]

    def test_tracker_reset_and_empty(self):
        """Tests that reset() wipes caches and empty matrix returns 0.0 safely."""
        tracker = ContinualMetricsTracker()
        tracker.record_task_metric(1, 1, 0.90)
        tracker.reset()

        empty_metrics = tracker.compute_matrix_metrics(num_tasks=1)
        assert empty_metrics["average_performance"] == 0.0
        assert empty_metrics["bwt"] == 0.0
        assert empty_metrics["forgetting_measure"] == 0.0
        assert len(tracker.best_historical_recall) == 0

    def test_tracker_adversarial_sparse_matrix_empty_max_crash(self):
        """Adversarial stress-test: Unhandled ValueError when previous task was not evaluated.

        When intermediate task evaluations are sparse (e.g. task 2 was never evaluated
        during task 2), `range(j, T)` produces an empty sequence for `max()`, raising
        `ValueError: max() iterable argument is empty`.
        """
        tracker = ContinualMetricsTracker()
        tracker.record_task_metric(1, 1, 0.8)
        # Task 2 is missing from mat
        tracker.record_task_metric(3, 1, 0.6)
        tracker.record_task_metric(3, 2, 0.7)
        tracker.record_task_metric(3, 3, 0.85)

        # Under the current implementation, this raises ValueError: max() iterable argument is empty
        with pytest.raises(ValueError, match="max\\(\\) iterable argument is empty"):
            tracker.compute_matrix_metrics(num_tasks=3)

    def test_tracker_adversarial_unused_current_task_id_flaw(self):
        """Adversarial stress-test: Demonstrates that current_task_id is ignored.

        If a base task has 4 classes and default classes_per_task=2 is used,
        calculate_continual_metrics sets num_old_classes = 4 - 2 = 2 despite current_task_id=1.
        """
        tracker = ContinualMetricsTracker()
        report = {
            "num_classes_seen": 4,
            "per_class": {
                "0": {"recall": 0.9, "f1-score": 0.9},
                "1": {"recall": 0.8, "f1-score": 0.8},
                "2": {"recall": 0.7, "f1-score": 0.7},
                "3": {"recall": 0.6, "f1-score": 0.6},
            },
        }
        f, old_f1, new_f1, h_mean = tracker.calculate_continual_metrics(
            report, current_task_id=1, classes_per_task=2
        )
        # Vulnerability confirmed: At Task 1, old_f1 is NOT 0.0 because current_task_id=1 is ignored!
        assert old_f1 > 0.0  # Demonstrates the logic flaw


# ==============================================================================
# SECTION 3: EMPIRICAL VERIFICATION OF SEED.PY
# ==============================================================================

class TestSeedEmpirical:
    """Rigorous empirical validation of set_deterministic_seed and reproducibility."""

    def test_seed_42_reproducibility_across_libraries(self):
        """Empirically verifies that seed 42 produces identical numbers across 10 repeated resets."""
        results_py_float: List[float] = []
        results_py_int: List[int] = []
        results_np_float: List[np.ndarray] = []
        results_np_int: List[np.ndarray] = []
        results_torch_float: List[torch.Tensor] = []
        results_torch_perm: List[torch.Tensor] = []

        for _ in range(5):
            set_deterministic_seed(42)

            # Python random
            results_py_float.append(random.random())
            results_py_int.append(random.randint(0, 1000000))

            # NumPy random
            results_np_float.append(np.random.randn(10))
            results_np_int.append(np.random.randint(0, 100, size=10))

            # PyTorch random
            results_torch_float.append(torch.randn(10))
            results_torch_perm.append(torch.randperm(20))

        # Check that all 5 runs produced identical values
        for i in range(1, 5):
            assert results_py_float[i] == results_py_float[0], "Python random.random() diverged!"
            assert results_py_int[i] == results_py_int[0], "Python random.randint() diverged!"
            assert np.array_equal(results_np_float[i], results_np_float[0]), "NumPy randn diverged!"
            assert np.array_equal(results_np_int[i], results_np_int[0]), "NumPy randint diverged!"
            assert torch.equal(results_torch_float[i], results_torch_float[0]), "PyTorch randn diverged!"
            assert torch.equal(results_torch_perm[i], results_torch_perm[0]), "PyTorch randperm diverged!"

    def test_seed_distinct_entropy_under_different_seeds(self):
        """Verifies that differing seeds (e.g. 42 vs 43) generate non-identical sequences."""
        set_deterministic_seed(42)
        r42_py = random.random()
        r42_np = np.random.randn(5)
        r42_torch = torch.randn(5)

        set_deterministic_seed(43)
        r43_py = random.random()
        r43_np = np.random.randn(5)
        r43_torch = torch.randn(5)

        assert r42_py != r43_py
        assert not np.array_equal(r42_np, r43_np)
        assert not torch.equal(r42_torch, r43_torch)

    def test_seed_environment_and_backend_flags(self):
        """Verifies that critical backend and environment variables are set correctly."""
        applied_seed = set_deterministic_seed(42)
        assert applied_seed == 42
        assert os.environ.get("PYTHONHASHSEED") == "42"
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

        if hasattr(torch.backends, "cudnn"):
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False

    def test_seed_worker_execution(self):
        """Verifies seed_worker initializes worker processes deterministically."""
        torch.manual_seed(42)
        seed_worker(0)
        w0_py = random.random()
        w0_np = np.random.rand()

        torch.manual_seed(42)
        seed_worker(0)
        assert random.random() == w0_py
        assert np.random.rand() == w0_np

    def test_dataloader_shuffle_determinism_with_seed(self):
        """Verifies that DataLoader shuffling is 100% reproducible with Generator and seed_worker."""
        def run_dl():
            """Executes seeded DataLoader pass to collect batch order."""
            set_deterministic_seed(42)
            ds = TensorDataset(torch.arange(100))
            g = torch.Generator()
            g.manual_seed(42)
            dl = DataLoader(ds, batch_size=10, shuffle=True, worker_init_fn=seed_worker, generator=g)
            return [b[0].tolist() for b in dl]

        run1 = run_dl()
        run2 = run_dl()
        assert run1 == run2, "DataLoader shuffle produced non-deterministic batches!"
