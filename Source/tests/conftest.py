"""Pytest fixtures and test doubles for the RISC-Mal E2E test suite.

Provides synthetic multi-view data generators, mock PE file builders,
declarative configuration fixtures, and contract-compliant model doubles.
"""

from pathlib import Path
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure src/ is resolvable in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from torch.utils.data import DataLoader
from riscmal.data.dataset import MalwareMultiViewDataset, MultiViewBatch, multiview_collate_fn
from riscmal.memory.buffer import RehearsalMemoryManager
from riscmal.utils.seed import set_deterministic_seed


# ---------------------------------------------------------------------------
# Contract-Compliant Reference Architecture
# ---------------------------------------------------------------------------

class ReferenceBackbone(nn.Module):
    """Reference multi-view feature extractor adhering to PROJECT.md interface contract.
    
    Inputs:
        header: [B, 4]
        imports: [B, 1000]
        img1d: [B, 1024]
        img2d: [B, 3, 224, 224]
        apis: [B, 100]
    Output:
        features: [B, 64]
    """

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.fc_header = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, out_dim))
        self.fc_imports = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, out_dim))
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, out_dim),
        )
        self.texture = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, out_dim),
        )
        self.api_embed = nn.Embedding(101, 32)
        self.api_lstm = nn.LSTM(32, out_dim, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 5, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(
        self,
        header: torch.Tensor,
        imports: torch.Tensor,
        img1d: torch.Tensor,
        img2d: torch.Tensor,
        apis: torch.Tensor,
    ) -> torch.Tensor:
        """Extracts and fuses 5 PE modalities into unified 64-dimensional feature vector."""
        b = header.shape[0]
        h_feat = self.fc_header(header)
        imp_feat = self.fc_imports(imports)
        
        # img1d: [B, 1024] -> [B, 1, 1024]
        img1d_feat = self.conv1d(img1d.unsqueeze(1))
        
        # img2d: [B, 3, 224, 224]
        img2d_feat = self.texture(img2d)
        
        # apis: [B, 100] -> [B, 100, 32] -> LSTM -> last hidden [B, 64]
        emb = self.api_embed(apis.clamp(0, 100))
        _, (hn, _) = self.api_lstm(emb)
        api_feat = hn[-1]

        # Fusion: concat 5x64 = 320 -> 64
        concat = torch.cat([h_feat, imp_feat, img1d_feat, img2d_feat, api_feat], dim=-1)
        return self.fusion(concat)


class ReferenceMultiViewClassifier(nn.Module):
    """Reference classifier head with norm-matched output expansion.
    
    Formula from PROJECT.md:
        w_new = r_bar * (u_k / ||u_k||_2)
        b_new = 0.0
    where r_bar = (1/C_old) * sum(||w_c||_2).
    """

    def __init__(self, num_classes: int = 4, feat_dim: int = 64) -> None:
        super().__init__()
        self.backbone = ReferenceBackbone(out_dim=feat_dim)
        self.classifier_head = nn.Linear(feat_dim, num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        x_dict: Union[Dict[str, torch.Tensor], MultiViewBatch, Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through reference classifier returning (logits, features)."""
        if isinstance(x_dict, (dict, MultiViewBatch)):
            features = self.backbone(
                x_dict["header"],
                x_dict["imports"],
                x_dict["img1d"],
                x_dict["img2d"],
                x_dict["apis"],
            )
        else:
            h, imp, i1d, i2d, api = x_dict[:5]
            features = self.backbone(h, imp, i1d, i2d, api)
        logits = self.classifier_head(features)
        return logits, features

    def expand_classes(self, num_new_classes: int, device: Union[torch.device, str] = "cpu") -> None:
        """Expands output classification layer using norm-matched weight initialization."""
        old_classes = self.classifier_head.out_features
        new_total_classes = old_classes + num_new_classes
        feat_dim = self.classifier_head.in_features

        old_weights = self.classifier_head.weight.data
        old_bias = self.classifier_head.bias.data

        # Average norm of existing class weight vectors
        r_bar = float(torch.norm(old_weights, dim=1).mean().item())

        # Initialize novel class weights with unit Gaussian normalized to unit sphere, scaled by r_bar
        u_k = torch.randn(num_new_classes, feat_dim, device=old_weights.device)
        u_k_norm = u_k / torch.norm(u_k, dim=1, keepdim=True).clamp(min=1e-8)
        new_weights = r_bar * u_k_norm
        new_biases = torch.zeros(num_new_classes, device=old_bias.device)

        # Build expanded linear head
        expanded_head = nn.Linear(feat_dim, new_total_classes).to(device)
        expanded_head.weight.data[:old_classes] = old_weights.to(device)
        expanded_head.weight.data[old_classes:] = new_weights.to(device)
        expanded_head.bias.data[:old_classes] = old_bias.to(device)
        expanded_head.bias.data[old_classes:] = new_biases.to(device)

        self.classifier_head = expanded_head
        self.num_classes = new_total_classes


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_seed():
    """Ensures deterministic random states across all tests."""
    set_deterministic_seed(42)


@pytest.fixture
def make_synthetic_batch():
    """Factory fixture generating valid MultiViewBatch dictionaries."""
    def _generator(
        batch_size: int = 32,
        num_classes: int = 4,
        class_offset: int = 0,
        device: str = "cpu",
    ) -> MultiViewBatch:
        header = torch.randn(batch_size, 4, device=device)
        imports = torch.randint(0, 2, (batch_size, 1000), dtype=torch.float32, device=device)
        img1d = torch.rand(batch_size, 1024, device=device)
        img2d = torch.rand(batch_size, 3, 224, 224, device=device)
        apis = torch.randint(0, 101, (batch_size, 100), dtype=torch.long, device=device)
        labels = torch.randint(
            class_offset,
            class_offset + num_classes,
            (batch_size,),
            dtype=torch.long,
            device=device,
        )

        batch = MultiViewBatch({
            "header": header,
            "imports": imports,
            "img1d": img1d,
            "img2d": img2d,
            "apis": apis,
            "label": labels,
        })
        return batch

    return _generator


@pytest.fixture
def synthetic_batch_32(make_synthetic_batch) -> MultiViewBatch:
    """Standard 32-sample synthetic batch."""
    return make_synthetic_batch(batch_size=32, num_classes=4, class_offset=0)


@pytest.fixture
def make_synthetic_dataset():
    """Factory fixture generating MalwareMultiViewDataset instances."""
    def _dataset_builder(
        num_samples: int = 60,
        num_classes: int = 4,
        class_offset: int = 0,
    ) -> MalwareMultiViewDataset:
        header = np.random.randn(num_samples, 4).astype(np.float32)
        imports = np.random.randint(0, 2, size=(num_samples, 1000)).astype(np.float32)
        img1d = np.random.rand(num_samples, 1024).astype(np.float32)
        img2d = np.random.rand(num_samples, 3, 224, 224).astype(np.float32)
        apis = np.random.randint(0, 101, size=(num_samples, 100)).astype(np.int64)
        labels = np.random.randint(class_offset, class_offset + num_classes, size=(num_samples,)).astype(np.int64)

        ds = MalwareMultiViewDataset(
            header=header,
            imports=imports,
            img1d=img1d,
            img2d=img2d,
            apis=apis,
            labels=labels,
        )
        return ds

    return _dataset_builder


@pytest.fixture
def make_synthetic_loader(make_synthetic_dataset):
    """Factory fixture generating DataLoader yielding batched MultiViewBatch instances."""
    def _loader_builder(
        num_samples: int = 60,
        num_classes: int = 4,
        class_offset: int = 0,
        batch_size: int = 16,
    ) -> DataLoader:
        ds = make_synthetic_dataset(num_samples=num_samples, num_classes=num_classes, class_offset=class_offset)
        return DataLoader(ds, batch_size=batch_size, collate_fn=multiview_collate_fn, shuffle=False)

    return _loader_builder


@pytest.fixture
def memory_manager() -> RehearsalMemoryManager:
    """Provides a freshly instantiated RehearsalMemoryManager."""
    return RehearsalMemoryManager(replay_ratio=0.10, min_per_class=1)


@pytest.fixture
def reference_model() -> ReferenceMultiViewClassifier:
    """Provides a fresh ReferenceMultiViewClassifier instance."""
    return ReferenceMultiViewClassifier(num_classes=4, feat_dim=64)


@pytest.fixture
def mock_pe_file(tmp_path: Path) -> Path:
    """Generates a minimal syntactically parseable PE binary for testing."""
    pe_file = tmp_path / "sample.exe"
    
    # DOS Header (64 bytes)
    dos_header = bytearray(64)
    dos_header[0:2] = b"MZ"
    # Offset to PE signature at e_lfanew (0x3C) -> point to offset 64
    dos_header[0x3C:0x40] = (64).to_bytes(4, byteorder="little")

    # PE Signature (4 bytes)
    pe_sig = b"PE\x00\x00"

    # COFF File Header (20 bytes)
    # Machine: i386 (0x014c), NumberOfSections: 1 (0x0001)
    coff_header = bytearray(20)
    coff_header[0:2] = (0x014C).to_bytes(2, byteorder="little")
    coff_header[2:4] = (1).to_bytes(2, byteorder="little")
    # SizeOfOptionalHeader: 224 (0x00e0)
    coff_header[16:18] = (224).to_bytes(2, byteorder="little")
    # Characteristics: 0x0102 (Executable, 32-bit word machine)
    coff_header[18:20] = (0x0102).to_bytes(2, byteorder="little")

    # Optional Header Standard Fields (28 bytes for 32-bit PE)
    opt_header = bytearray(224)
    opt_header[0:2] = (0x010B).to_bytes(2, byteorder="little") # Magic PE32
    opt_header[16:20] = (0x1000).to_bytes(4, byteorder="little") # AddressOfEntryPoint
    opt_header[28:32] = (0x00400000).to_bytes(4, byteorder="little") # ImageBase
    opt_header[32:36] = (0x1000).to_bytes(4, byteorder="little") # SectionAlignment
    opt_header[36:40] = (0x200).to_bytes(4, byteorder="little") # FileAlignment
    opt_header[56:60] = (0x3000).to_bytes(4, byteorder="little") # SizeOfImage
    opt_header[60:64] = (0x400).to_bytes(4, byteorder="little") # SizeOfHeaders

    # Section Header (40 bytes): .text
    sec_header = bytearray(40)
    sec_header[0:5] = b".text"
    sec_header[8:12] = (0x1000).to_bytes(4, byteorder="little") # VirtualSize
    sec_header[12:16] = (0x1000).to_bytes(4, byteorder="little") # VirtualAddress
    sec_header[16:20] = (0x200).to_bytes(4, byteorder="little") # SizeOfRawData
    sec_header[20:24] = (0x400).to_bytes(4, byteorder="little") # PointerToRawData
    sec_header[36:40] = (0x60000020).to_bytes(4, byteorder="little") # Characteristics: Code, Executable, Readable

    # Padding up to PointerToRawData (0x400 = 1024 bytes)
    header_bytes = bytes(dos_header + pe_sig + coff_header + opt_header + sec_header)
    padding = b"\x00" * (1024 - len(header_bytes))

    # Section Raw Data (.text code)
    code_data = b"\x90" * 512 # NOP instructions

    pe_content = header_bytes + padding + code_data
    pe_file.write_bytes(pe_content)
    return pe_file


@pytest.fixture
def mock_cuckoo_json(tmp_path: Path) -> Path:
    """Generates a mock Cuckoo dynamic analysis JSON report."""
    json_path = tmp_path / "analysis.json"
    report = {
        "target": {
            "file": {
                "name": "sample.exe",
                "size": 1536,
                "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            }
        },
        "static": {
            "pe_imported_symbols": ["CreateFileA", "VirtualAlloc", "WriteProcessMemory", "ExitProcess"],
            "pe_header": {
                "size_of_data": 512,
                "virtual_size": 4096,
                "entropy": 5.42,
                "characteristics": 258,
            }
        },
        "behavior": {
            "processes": [
                {
                    "process_name": "sample.exe",
                    "calls": [
                        {"api": "NtAllocateVirtualMemory", "category": "memory"},
                        {"api": "NtWriteVirtualMemory", "category": "memory"},
                        {"api": "NtCreateThreadEx", "category": "process"},
                    ]
                }
            ]
        }
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return json_path


@pytest.fixture
def synthetic_configs() -> Dict[str, Any]:
    """Provides experiment configurations conforming to protocol.yaml and method configs."""
    return {
        "protocol": {
            "dataset_root": "data/sample",
            "seed": 42,
            "tasks": [
                {"task_id": 1, "classes": [0, 1, 2, 3], "epochs": 5, "lr": 0.001},
                {"task_id": 2, "classes": [4, 5], "epochs": 5, "lr": 0.0005},
                {"task_id": 3, "classes": [6, 7], "epochs": 5, "lr": 0.0005},
            ],
            "batch_size": 32,
            "replay_batch_size": 32,
        },
        "riscmal": {
            "alpha": 0.5,
            "tau_p": 0.8,
            "T_p": 0.2,
            "lambda_sp": 0.1,
            "lambda_kd": 0.1,
            "T_kd": 2.0,
            "replay_ratio": 0.10,
            "protection_strategy": "risc_soft",
        },
    }
