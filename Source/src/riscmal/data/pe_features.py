"""Atomic feature extractors for Portable Executable (PE) binaries and Cuckoo JSON reports.

Implements feature extraction across all five heterogeneous modalities:
1. PE Header attributes (size of data, virtual size, entropy, characteristics)
2. PE Imports multi-hot binary vectors (Top-1000 DLL symbols)
3. 1D Byte plots (1024-dimensional spatial layout)
4. 2D Multi-scale texture images (224x224 RGB byte visualizations)
5. Dynamic API call execution traces (Top-100 API sequence)

Supports Cuckoo analysis JSON input with robust static pefile fallback.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import cv2
import numpy as np
import pefile

logger = logging.getLogger(__name__)


def clean_numeric(val: Any) -> float:
    """Sanitizes raw values into floating point numbers, handling hexadecimal strings.

    Args:
        val: Input object (int, float, hex string, None, etc.).

    Returns:
        Float value, defaulting to 0.0 on missing or unparseable input.
    """
    if val is None:
        return 0.0
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            s = val.strip().lower()
            if not s:
                return 0.0
            if s.startswith("0x"):
                return float(int(s, 16))
            return float(s)
        return float(val)
    except Exception:
        return 0.0


def extract_byte_images(
    exe_path: Union[str, Path],
    max_buffer_bytes: int = 50 * 1024 * 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates 1D and 2D byte visualizations from an executable binary.

    1D Image: Raw binary stream padded to square, resized to 32x32 using nearest-neighbor
        interpolation, flattened to 1024 floats normalized to [0.0, 1.0].
    2D Image: Resizes the 32x32 byte grid to 224x224 bilinear and replicates across
        3 RGB channels, normalized to [0.0, 1.0].

    Args:
        exe_path: Path to executable binary file.
        max_buffer_bytes: Maximum number of raw bytes to read into RAM to prevent OOM.

    Returns:
        Tuple of (img1d, img2d) where:
            img1d: np.ndarray of shape (1024,), dtype float32.
            img2d: np.ndarray of shape (224, 224, 3), dtype float32.
    """
    empty_1d = np.zeros(1024, dtype=np.float32)
    empty_2d = np.zeros((224, 224, 3), dtype=np.float32)

    path = Path(exe_path)
    if not path.is_file():
        return empty_1d, empty_2d

    try:
        with open(path, "rb") as f:
            raw_bytes = np.frombuffer(f.read(max_buffer_bytes), dtype=np.uint8)

        if len(raw_bytes) == 0:
            return empty_1d, empty_2d

        # Calculate square size and pad
        size = int(np.ceil(np.sqrt(len(raw_bytes))))
        pad_size = size * size - len(raw_bytes)
        padded = np.pad(raw_bytes, (0, pad_size), "constant") if pad_size > 0 else raw_bytes

        # 32x32 Nearest Neighbor layout
        img_32 = cv2.resize(
            padded.reshape((size, size)),
            (32, 32),
            interpolation=cv2.INTER_NEAREST,
        )

        # 1D flattened representation: (1024,)
        img_1d = (img_32.flatten().astype(np.float32)) / 255.0

        # 2D RGB multi-scale texture: (224, 224, 3)
        img_resized = cv2.resize(img_32, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_2d = np.stack([img_resized] * 3, axis=-1).astype(np.float32) / 255.0

        return img_1d, img_2d
    except Exception as exc:
        logger.debug("Failed to extract byte images from %s: %s", exe_path, exc)
        return empty_1d, empty_2d


def extract_header_from_json(report: Dict[str, Any]) -> List[float]:
    """Extracts 4-D header attributes from a Cuckoo analysis JSON report.

    Attributes:
        1. size_of_data (SizeOfRawData)
        2. virtual_size (Misc_VirtualSize)
        3. entropy (Shannon entropy)
        4. characteristics (IMAGE_SCN_* flags)

    Searches primarily for .text or .code sections, with fallback to any
    executable section (IMAGE_SCN_MEM_EXECUTE = 0x20000000).

    Args:
        report: Parsed Cuckoo JSON dictionary.

    Returns:
        List of 4 float values [size_of_data, virtual_size, entropy, characteristics].
    """
    sections = report.get("static", {}).get("pe_sections", [])
    if not sections:
        return [0.0, 0.0, 0.0, 0.0]

    # Primary search: .text or .code
    for sec in sections:
        name = str(sec.get("name", "")).strip().lower()
        if name in (".text", ".code"):
            return [
                clean_numeric(sec.get("size_of_data", 0)),
                clean_numeric(sec.get("virtual_size", 0)),
                clean_numeric(sec.get("entropy", 0)),
                clean_numeric(sec.get("characteristics", 0)),
            ]

    # Fallback search: executable section bitmask
    for sec in sections:
        chars = int(clean_numeric(sec.get("characteristics", 0)))
        if chars & 0x20000000:
            return [
                clean_numeric(sec.get("size_of_data", 0)),
                clean_numeric(sec.get("virtual_size", 0)),
                clean_numeric(sec.get("entropy", 0)),
                float(chars),
            ]

    # Fallback to first section if available
    first_sec = sections[0]
    return [
        clean_numeric(first_sec.get("size_of_data", 0)),
        clean_numeric(first_sec.get("virtual_size", 0)),
        clean_numeric(first_sec.get("entropy", 0)),
        clean_numeric(first_sec.get("characteristics", 0)),
    ]


def extract_header_from_pe(pe_source: Union[str, Path, bytes]) -> List[float]:
    """Static fallback extractor: parses raw PE binary headers using pefile.

    Args:
        pe_source: Path to binary or raw bytes.

    Returns:
        List of 4 float values [size_of_data, virtual_size, entropy, characteristics].
    """
    try:
        if isinstance(pe_source, bytes):
            pe = pefile.PE(data=pe_source, fast_load=True)
        else:
            pe = pefile.PE(str(pe_source), fast_load=True)

        for sec in pe.sections:
            name = sec.Name.decode("latin-1", errors="ignore").strip("\x00").lower()
            if name in (".text", ".code") or (sec.Characteristics & 0x20000000):
                size_of_data = float(sec.SizeOfRawData)
                virtual_size = float(sec.Misc_VirtualSize)
                entropy = float(sec.get_entropy())
                chars = float(sec.Characteristics)
                pe.close()
                return [size_of_data, virtual_size, entropy, chars]

        if pe.sections:
            sec = pe.sections[0]
            res = [
                float(sec.SizeOfRawData),
                float(sec.Misc_VirtualSize),
                float(sec.get_entropy()),
                float(sec.Characteristics),
            ]
            pe.close()
            return res
        pe.close()
    except Exception as exc:
        logger.debug("pefile extraction failed: %s", exc)

    return [0.0, 0.0, 0.0, 0.0]


def extract_imports_from_json(
    report: Dict[str, Any],
    imp_map: Dict[str, int],
    top_k: int = 1000,
) -> np.ndarray:
    """Builds a 1000-D multi-hot binary indicator vector over imported DLL functions.

    Args:
        report: Parsed Cuckoo JSON dictionary.
        imp_map: Dictionary mapping imported function names to feature indices (0 to top_k-1).
        top_k: Dimensionality of the multi-hot feature vector (default: 1000).

    Returns:
        np.ndarray of shape (top_k,), dtype float32 with values in {0.0, 1.0}.
    """
    iv = np.zeros(top_k, dtype=np.float32)
    entries = report.get("static", {}).get("pe_imports", [])
    if not isinstance(entries, list):
        return iv

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        imports = entry.get("imports", [])
        if not isinstance(imports, list):
            continue
        for imp in imports:
            if isinstance(imp, dict):
                name = imp.get("name")
                if name and name in imp_map:
                    idx = imp_map[name]
                    if 0 <= idx < top_k:
                        iv[idx] = 1.0
    return iv


def extract_imports_from_pe(
    pe_source: Union[str, Path, bytes],
    imp_map: Dict[str, int],
    top_k: int = 1000,
) -> np.ndarray:
    """Static fallback extractor: builds multi-hot import vector from raw PE using pefile.

    Args:
        pe_source: Path to binary or raw bytes.
        imp_map: Dictionary mapping function names to indices.
        top_k: Length of output vector.

    Returns:
        np.ndarray of shape (top_k,), dtype float32.
    """
    iv = np.zeros(top_k, dtype=np.float32)
    try:
        if isinstance(pe_source, bytes):
            pe = pefile.PE(data=pe_source)
        else:
            pe = pefile.PE(str(pe_source))

        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        name = imp.name.decode("latin-1", errors="ignore")
                        if name in imp_map:
                            idx = imp_map[name]
                            if 0 <= idx < top_k:
                                iv[idx] = 1.0
        pe.close()
    except Exception as exc:
        logger.debug("pefile import parsing failed: %s", exc)

    return iv


def extract_apis_from_json(
    report: Dict[str, Any],
    api_map: Dict[str, int],
    max_seq: int = 100,
) -> np.ndarray:
    """Extracts dynamic API call execution sequence, mapped to integer token IDs.

    Token IDs range from 1 to max_seq (0 is strictly reserved for padding).

    Args:
        report: Parsed Cuckoo JSON dictionary.
        api_map: Dictionary mapping API names to token IDs (1 to 100).
        max_seq: Sequence length limit (default: 100).

    Returns:
        np.ndarray of shape (max_seq,), dtype int64.
    """
    aseq: List[int] = []
    processes = report.get("behavior", {}).get("processes", [])
    if isinstance(processes, list):
        for proc in processes:
            if not isinstance(proc, dict):
                continue
            calls = proc.get("calls", [])
            if not isinstance(calls, list):
                continue
            for call in calls:
                if isinstance(call, dict):
                    api_name = call.get("api")
                    if api_name and api_name in api_map:
                        aseq.append(api_map[api_name])
                        if len(aseq) >= max_seq:
                            break
            if len(aseq) >= max_seq:
                break

    # Right-pad with 0 (PAD_TOKEN) to ensure fixed length
    padded_seq = (aseq + [0] * max_seq)[:max_seq]
    return np.array(padded_seq, dtype=np.int64)
