from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from .data import download_data
from .pipelines.csv_infer import run_csv_inference
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


def infer_cli(
    csv_path: Optional[str] = None,
    text_column: str = "text",
    output_dir: Optional[str] = None,
    limit: Optional[int] = None,
    config_name: str = "config",
    overrides: Optional[Iterable[str]] = None,
) -> None:
    cfg = _load_config(config_name, overrides)
    if csv_path:
        cfg.inference.dataset.csv_path = csv_path
    if text_column:
        cfg.inference.dataset.text_column = text_column
    if output_dir:
        cfg.inference.output.dir = output_dir
    if limit is not None:
        cfg.inference.dataset.limit = limit
    run_csv_inference(cfg, csv_path=csv_path, text_column=text_column, output_dir=output_dir, limit=limit)

def main() -> None:
    fire.Fire(
        {
            "train": train_cli,
            "download-data": download_cli,
            "preprocess": preprocess_cli,
            "infer": infer_cli,
        }
    )


if __name__ == "__main__":
    main()
