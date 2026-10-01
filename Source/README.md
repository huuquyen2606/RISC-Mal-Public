# RISC-Mal: Risk-Informed Selective Class-Incremental Learning for Imbalanced Windows Malware Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-100%2F100%20passed-brightgreen.svg)](tests/)

Official PyTorch implementation of the research paper:
> **"Risk-Informed Selective Class-Incremental Learning for Imbalanced Windows Malware Detection"** (RISC-Mal)

---

## 🏛️ System Architecture

```
                               ┌────────────────────────┐
                               │ Raw PE Windows Malware │
                               └───────────┬────────────┘
                                           │
       ┌───────────────────┬───────────────┼───────────────┬───────────────────┐
       ▼                   ▼               ▼               ▼                   ▼
┌──────────────┐   ┌──────────────┐ ┌─────────────┐ ┌──────────────┐   ┌──────────────┐
│  PE-Header   │   │  PE-Imports  │ │  Conv1D 1D  │ │  Texture 2D  │   │ API Sequence │
│    [B, 4]    │   │  [B, 1000]   │ │  [B, 1024]  │ │[B,3,224,224] │   │   [B, 100]   │
└──────┬───────┘   └──────┬───────┘ └──────┬──────┘ └──────┬───────┘   └──────┬───────┘
       ▼                   ▼               ▼               ▼                   ▼
┌──────────────┐   ┌──────────────┐ ┌─────────────┐ ┌──────────────┐   ┌──────────────┐
│ 5-Layer MLP  │   │ 5-Layer MLP  │ │3x Conv1D+BN │ │EfficientNet  │   │ 3x LSTM +    │
│  [B -> 64]   │   │  [B -> 64]   │ │  [B -> 64]  │ │ Ensemble GAP │   │ MultiheadAttn│
└──────┬───────┘   └──────┬───────┘ └──────┬──────┘ └──────┬───────┘   └──────┬───────┘
       └───────────────────┴───────────────┼───────────────┴───────────────────┘
                                           │ (Concat 5 x 64 = 320-d)
                                           ▼
                               ┌────────────────────────┐
                               │ Multi-View Fusion MLP  │
                               │ [320->256->128->64-d]  │
                               └───────────┬────────────┘
                                           │ Latent Representation h in R^64
                                           ▼
                               ┌────────────────────────┐
                               │ Unified Classifier     │
                               │ Output-Only Expansion  │
                               │   (Task 1 -> 2 -> 3)   │
                               └────────────────────────┘
```

---

## 📂 Repository Structure

```
├── configs/
│   ├── protocol.yaml               # Benchmark protocol
│   ├── task_map.json               # Task-to-class schedule
│   └── methods/                    # Continual learning method configs
│       ├── baseline.yaml
│       ├── ewc.yaml
│       ├── lwf.yaml
│       ├── lgr.yaml
│       ├── bir.yaml
│       ├── gc.yaml
│       ├── der.yaml
│       ├── foster.yaml
│       └── riscmal.yaml
├── scripts/
│   ├── prepare_data.py             # Feature extraction & preprocessing
│   ├── train_incremental.py        # Incremental training pipeline
│   ├── evaluate.py                 # Checkpoint evaluation & metrics
│   ├── run_all.py                  # Benchmark runner
│   └── count_parameters.py         # Parameter profiling
├── src/
│   └── riscmal/                    # Core library package
│       ├── __init__.py
│       ├── backbones/              # Multi-view feature extractors
│       ├── models/                 # Model architectures
│       ├── continual/              # Continual learning strategies
│       ├── memory/                 # Rehearsal buffer
│       ├── data/                   # Dataset & data loaders
│       ├── evaluation/             # Evaluation & metrics
│       └── utils/                  # Utilities
├── tests/                          # Automated test suite
├── CITATION.cff
├── environment.yml
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── info.txt
```

---

## 💾 Datasets & Pretrained Checkpoints (Google Drive)

In accordance with academic repository hygiene, raw malware datasets and large PyTorch model checkpoints (`.pth`) are archived separately on Google Drive to maintain a clean, lightweight distribution package:

- 🔗 **Google Drive Archive:** [Download Dataset, Checkpoints & Source](https://drive.google.com/drive/folders/1Arlo3AgeNlxjbZZ6KBsHDaLME-gUg2mR)

The Google Drive archive contains three primary directories:
- **`Dataset/`**: Contains **`Data_RISC.zip`** (804.7 MB), which extracts into **`incremental_data_v2/`** containing 3 incremental task splits (`task1/`, `task2/`, `task3/` with 12 `.npy` feature files each) and global vocabulary/scaler assets (`global_api_map.pkl`, `global_imp_map.pkl`, `global_scaler.pkl`):
  - **Task 1**: `Benign`, `Locker` (classes 0, 1)
  - **Task 2**: `Mediyes`, `Winwebsec` (classes 2, 3)
  - **Task 3**: `Zbot`, `Zeroaccess` (classes 4, 5)
  *(Note: `incremental_data_v2` is generated by the offline feature extraction script `trichxuat2.py`. It stores raw 3-task split arrays without on-disk deduplication. In accordance with Section 3.1 of the paper, in-memory SHA-256 byte deduplication via `purge_data_leakage()` is executed dynamically at runtime during training to purge 2,444 train-test duplicate leaks).*
- **`Check_point/`**: Pretrained PyTorch model checkpoints (`.pth`) across all 9 benchmarked CL strategies (RISC-Mal, DER, FOSTER, GC, etc.).
- **`Source/`**: Complete, clean, and self-contained source code package of the RISC-Mal framework.

---

## ⚡ Quickstart

### 1. Installation

Set up a Python virtual environment and install dependencies:

```bash
# Create and activate virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Installation & Test Suite

Run the full 100-test verification suite:
```bash
pytest tests/ -v
```
All 100 tests execute deterministically on CPU with 100% pass rate.

### 3. Evaluating Checkpoints

Evaluate pretrained checkpoints on cumulative test sets:
```bash
python scripts/evaluate.py --checkpoint results/RISCMAL_S42_task1_checkpoint.pth --method riscmal
```

### 4. Training an Incremental Strategy

Train a continual learning strategy sequentially across tasks:
```bash
python scripts/train_incremental.py --method riscmal --seed 42
```

### 5. Running Full Benchmark Suite (All 9 Methods)

```bash
python scripts/run_all.py --methods baseline ewc lwf lgr bir gc der foster riscmal --seeds 42
```

### 6. Parameter Profiling

Audit the exact parameter counts across methods and tasks:
```bash
python scripts/count_parameters.py --method all --task 3
```

---

## 👥 Authors & Affiliation

- **Vo Dang Khoa** (`25520883@gm.uit.edu.vn`) - [ORCID](https://orcid.org/0009-0007-2796-6920)
- **Nguyen Minh Tri** (`25521912@gm.uit.edu.vn`) - [ORCID](https://orcid.org/0009-0007-7402-096X)
- **Nguyen Huu Quyen** (`quyennh@uit.edu.vn`) - [ORCID](https://orcid.org/0000-0002-0065-9919)
- **Pham Van-Hau** (`haupv@uit.edu.vn`) - [ORCID](https://orcid.org/0000-0003-3147-3356)

*Information Security Laboratory (InSecLab), University of Information Technology (UIT), Vietnam National University Ho Chi Minh City (VNU-HCM), Vietnam.*

---

## 📜 Citation

```bibtex
@article{riscmal2026,
  title   = {Risk-Informed Selective Class-Incremental Learning for Imbalanced Windows Malware Detection},
  author  = {Vo, Dang Khoa and Nguyen, Minh Tri and Nguyen, Huu Quyen and Pham, Van-Hau},
  year    = {2026}
}
```