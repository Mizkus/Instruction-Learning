from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from .data import download_data
from .pipelines.nyt_preprocess import preprocess_nyt_dataset
from .train import run_training


def _resolve_config_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("configs directory not found relative to cli.py")


CONFIG_DIR = _resolve_config_dir()


def _load_config(config_name: str, overrides: Optional[Iterable[str]] = None) -> DictConfig:
    overrides = list(overrides or [])
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def train_cli(config_name: str = "config", overrides: Optional[Iterable[str]] = None) -> None:
    cfg = _load_config(config_name, overrides)
    run_training(cfg)


def download_cli(target_dir: str = "data/raw") -> None:
    cfg = _load_config("config")
    download_data(target_dir, cfg)


def preprocess_cli(config_name: str = "config", overrides: Optional[Iterable[str]] = None) -> None:
    cfg = _load_config(config_name, overrides)
    preprocess_nyt_dataset(cfg)


def main() -> None:
    fire.Fire({"train": train_cli, "download-data": download_cli, "preprocess": preprocess_cli})


if __name__ == "__main__":
    main()
