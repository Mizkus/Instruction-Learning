from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from datasets import load_dataset
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

def _resolve_config_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("configs directory not found relative to download.py")


CONFIG_DIR = _resolve_config_dir()


def _load_default_cfg() -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")
    return cfg


def _choose_splits(dataset, candidates: Iterable[str]) -> Iterable[str]:
    available = set(dataset.keys())
    chosen = [split for split in candidates if split in available]
    if not chosen:
        chosen = list(available)
    return chosen


def download_data(target_dir: str | Path | None = None, cfg: Optional[DictConfig] = None) -> Path:
    """Download NYTClustering/topic subset from Hugging Face to `topic.csv`.

    Dataset reference: https://huggingface.co/datasets/BrandonZYW/NYTClustering
    """

    cfg = cfg or _load_default_cfg()
    data_cfg = cfg.data.dataset
    target_path = Path(target_dir or Path(cfg.paths.data_root) / "raw")
    target_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(data_cfg.huggingface_id, data_cfg.subset)
    splits = _choose_splits(dataset, data_cfg.get("splits", []))
    frames = [dataset[split].to_pandas() for split in splits]
    df = pd.concat(frames, ignore_index=True)

    csv_path = target_path / data_cfg.topic_csv
    df.to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    cfg = _load_default_cfg()
    download_data(cfg=cfg)


if __name__ == "__main__":
    main()
