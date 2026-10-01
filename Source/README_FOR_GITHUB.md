# RISC-Mal: Risk-Informed Selective Class-Incremental Learning for Imbalanced Windows Malware Detection

Official repository for the research paper:
> **"Risk-Informed Selective Class-Incremental Learning for Imbalanced Windows Malware Detection"** (RISC-Mal)

---

## 🚀 Google Drive Archive Access

To maintain repository hygiene and accommodate storage limits, the full experimental dataset, pretrained PyTorch model checkpoints, and verified source code package are hosted on Google Drive:

🔗 **Direct Download Link:**  
**[Download RISC-Mal Artifacts (Google Drive Archive)](https://drive.google.com/drive/folders/1Arlo3AgeNlxjbZZ6KBsHDaLME-gUg2mR)**

The Google Drive archive contains three primary directories:
- **`Dataset/`**: Contains **`Data_RISC.zip`** (804.7 MB), which extracts into **`incremental_data_v2/`** containing 3 incremental task splits (`task1/`, `task2/`, `task3/` with 12 `.npy` feature files each) and global vocabulary/scaler assets (`global_api_map.pkl`, `global_imp_map.pkl`, `global_scaler.pkl`):
  - **Task 1**: `Benign`, `Locker` (classes 0, 1)
  - **Task 2**: `Mediyes`, `Winwebsec` (classes 2, 3)
  - **Task 3**: `Zbot`, `Zeroaccess` (classes 4, 5)
  *(Note: `incremental_data_v2` is the output of the feature extraction pipeline `trichxuat2.py`. It stores raw 3-task split arrays; cryptographic SHA-256 cross-split duplicate purging is executed dynamically at runtime during training).*
- **`Check_point/`**: Pretrained PyTorch model checkpoints (`.pth`) across all evaluated incremental tasks and continual learning baselines.
- **`Source/`**: Complete, clean, and self-contained source code package of the RISC-Mal framework.

---

## 🛠️ Usage Instructions

To inspect, evaluate, or reproduce the experimental results:
1. Download `Source/` together with `Dataset/Data_RISC.zip` and `Check_point/` from the [Google Drive Archive](https://drive.google.com/drive/folders/1Arlo3AgeNlxjbZZ6KBsHDaLME-gUg2mR).
2. Extract `Data_RISC.zip` to obtain the `incremental_data_v2/` directory and place it alongside or inside `Source/`.
3. Model training directly consumes `incremental_data_v2/` (faithfully following `New_RISC-Mal.ipynb`), with in-memory SHA-256 deduplication automatically performed during data loading.
4. Inside the `Source/` directory, refer to `README.md` for environment setup (`requirements.txt`) and execution commands for model training and evaluation.

---

## 👥 Authors & Affiliations

- **Vo Dang Khoa** (`25520883@gm.uit.edu.vn`) — [ORCID](https://orcid.org/0009-0007-2796-6920)
- **Nguyen Minh Tri** (`25521912@gm.uit.edu.vn`) — [ORCID](https://orcid.org/0009-0007-7402-096X)
- **Nguyen Huu Quyen** (`quyennh@uit.edu.vn`) — [ORCID](https://orcid.org/0000-0002-0065-9919)
- **Pham Van-Hau** (`haupv@uit.edu.vn`) — [ORCID](https://orcid.org/0000-0003-3147-3356)

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
