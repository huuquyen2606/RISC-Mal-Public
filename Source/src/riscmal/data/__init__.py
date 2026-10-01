"""Multi-view data loading, raw feature extraction, and leakage prevention pipeline."""

from riscmal.data.dataset import (
    MalwareMultiViewDataset,
    MultiViewBatch,
    multiview_collate_fn,
)
from riscmal.data.leakage import (
    audit_internal_duplicates,
    audit_leakage,
    purge_data_leakage,
)
from riscmal.data.pe_features import (
    clean_numeric,
    extract_apis_from_json,
    extract_byte_images,
    extract_header_from_json,
    extract_header_from_pe,
    extract_imports_from_json,
    extract_imports_from_pe,
)
from riscmal.data.pipeline import (
    GlobalAssetBuilder,
    TaskDataExtractor,
    build_incremental_dataloaders,
)

__all__ = [
    "MalwareMultiViewDataset",
    "MultiViewBatch",
    "multiview_collate_fn",
    "clean_numeric",
    "extract_byte_images",
    "extract_header_from_json",
    "extract_header_from_pe",
    "extract_imports_from_json",
    "extract_imports_from_pe",
    "extract_apis_from_json",
    "purge_data_leakage",
    "audit_leakage",
    "audit_internal_duplicates",
    "GlobalAssetBuilder",
    "TaskDataExtractor",
    "build_incremental_dataloaders",
]
