"""PTB SD-format loader for reproduction scripts (minimal excerpt)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PennTreebankLoader:
    """Load Penn Treebank Stanford Dependencies .txt files."""

    @staticmethod
    def load_from_txt(txt_path: Path) -> List[Dict]:
        samples: List[Dict] = []
        current: Dict = {"words": [], "heads": [], "relns": []}

        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current["words"]:
                        samples.append(current)
                    current = {"words": [], "heads": [], "relns": []}
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                label = parts[1]
                if "-" not in label:
                    continue
                head_str, reln = label.split("-", 1)
                try:
                    head = int(head_str)
                except ValueError:
                    continue
                current["words"].append(word)
                current["heads"].append(head)
                current["relns"].append(reln)

        if current["words"]:
            samples.append(current)
        return samples


def load_ptb_dataset(
    split: str = "dev",
    num_samples: Optional[int] = None,
    base_dir: Path | str = Path("data/depparse"),
    data_format: str = "txt",
) -> List[Dict]:
    """Load PTB SD .txt split (dev/test/train)."""
    del data_format
    base = Path(base_dir)
    txt_path = base / f"{split}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(
            f"PTB file not found: {txt_path}. "
            "Set PTB_DEPPARSE_DIR to your LDC Treebank-3 depparse directory."
        )
    samples = PennTreebankLoader.load_from_txt(txt_path)
    if num_samples is not None and num_samples > 0:
        samples = samples[:num_samples]
    logger.info("Loaded %s samples from %s", len(samples), txt_path)
    return samples


def require_ptb_depparse_dir() -> Path:
    import os

    path = Path(os.environ.get("PTB_DEPPARSE_DIR", "data/depparse"))
    if not (path / "dev.txt").exists():
        raise FileNotFoundError(
            f"PTB depparse not found under {path}. "
            "Obtain Treebank-3 (LDC99T42) and set PTB_DEPPARSE_DIR."
        )
    return path


def ptb_cache_root() -> Path:
    import os

    return Path(os.environ.get("PTB_CACHE_ROOT", "cache/ptb_ig_analysis"))
