from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import lightning as L
import torch
from hydra import main as hydra_main
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from .data import EmbeddingDataModule
from .models import InstructionAdapterModule


def run_training(cfg: DictConfig) -> None:
    """Entry point shared between Hydra CLI and Python API."""
    L.seed_everything(cfg.project.seed, workers=True)

    datamodule = EmbeddingDataModule(cfg)
    model = InstructionAdapterModule(cfg.model, cfg.training, cfg.training.monitoring, cfg.project.seed)

    logger = _build_mlflow_logger(cfg)
    callbacks = _build_callbacks(cfg)

    trainer_kwargs = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(Path(cfg.paths.artifacts_root)),
        **trainer_kwargs,
    )
    trainer.fit(model=model, datamodule=datamodule)

    teacher_latency = _load_teacher_latency(cfg)
    adapter_latency = _measure_adapter_latency(model, datamodule)
    if logger and (teacher_latency or adapter_latency):
        experiment = logger.experiment
        run_id = logger.run_id
        if teacher_latency:
            experiment.log_metric(run_id, "teacher_instruct_latency_sec", teacher_latency)
        if adapter_latency:
            total, per_sample = adapter_latency
            experiment.log_metric(run_id, "adapter_latency_sec", total)
            if per_sample is not None:
                experiment.log_metric(run_id, "adapter_latency_ms_per_sample", per_sample * 1e3)
            if teacher_latency:
                experiment.log_metric(run_id, "latency_speedup", teacher_latency / total if total else None)

    best_ckpt = _best_checkpoint_path(callbacks)
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=best_ckpt if best_ckpt else None,
    )


def _build_mlflow_logger(cfg: DictConfig) -> Optional[MLFlowLogger]:
    mlflow_cfg = cfg.logging.mlflow
    tracking_uri = mlflow_cfg.tracking_uri
    if not tracking_uri:
        return None
    logger = MLFlowLogger(
        experiment_name=mlflow_cfg.experiment_name,
        run_name=mlflow_cfg.run_name,
        tracking_uri=tracking_uri,
    )
    logger.log_hyperparams(
        {
            "project": cfg.project.name,
            "seed": cfg.project.seed,
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "training": OmegaConf.to_container(cfg.training, resolve=True),
        }
    )
    commit = _current_git_commit()
    if commit and cfg.logging.capture_git:
        logger.experiment.log_param(logger.run_id, "git_commit", commit)
    return logger


def _build_callbacks(cfg: DictConfig) -> List[Callback]:
    ckpt_cfg = cfg.model.checkpoint
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(Path(ckpt_cfg.dirpath)),
        filename=ckpt_cfg.filename,
        monitor=cfg.training.monitoring.val_metric,
        mode=cfg.training.monitoring.mode,
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_cb, lr_monitor]


def _current_git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _load_teacher_latency(cfg: DictConfig) -> Optional[float]:
    metadata_path = Path(cfg.paths.data_root) / "embeddings" / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    return metadata.get("instruct_encode_seconds")


def _measure_adapter_latency(
    model: InstructionAdapterModule, datamodule: EmbeddingDataModule
) -> Optional[Tuple[float, Optional[float]]]:
    datamodule.setup("validate")
    dataloader = datamodule.val_dataloader()
    device = model.device
    model.eval()
    total_samples = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            base_embeddings = batch[0].to(device)
            _ = model(base_embeddings)
            total_samples += base_embeddings.size(0)
    total_time = time.perf_counter() - start
    if total_samples == 0:
        return None
    return total_time, total_time / total_samples


def _best_checkpoint_path(callbacks: List[Callback]) -> Optional[str]:
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            if callback.best_model_path:
                return callback.best_model_path
    return None


@hydra_main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
