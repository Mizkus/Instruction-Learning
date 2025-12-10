from __future__ import annotations

from typing import Any, Dict, List, Optional

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from torch import nn


class LinearAdapter(nn.Module):
    """Single-layer affine projection between embedding spaces."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.projection = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.dropout(x)
        return self.projection(x)


class InstructionAdapterModule(L.LightningModule):
    """Lightning wrapper that trains the linear adapter with MSE loss and reports V-measure."""

    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        monitoring_cfg: DictConfig,
        project_seed: int,
    ) -> None:
        super().__init__()
        model_cfg_resolved = OmegaConf.to_container(model_cfg, resolve=True)
        optimizer_cfg_resolved = OmegaConf.to_container(optimizer_cfg, resolve=True)
        monitoring_cfg_resolved = OmegaConf.to_container(monitoring_cfg, resolve=True)
        self.save_hyperparameters(
            {
                "model": model_cfg_resolved,
                "optimizer": optimizer_cfg_resolved.get("optimizer"),
                "scheduler": optimizer_cfg_resolved.get("scheduler"),
                "monitoring": monitoring_cfg_resolved,
            }
        )

        self.adapter = LinearAdapter(
            input_dim=model_cfg.input_dim,
            output_dim=model_cfg.output_dim,
            bias=model_cfg.get("bias", True),
            dropout=model_cfg.get("dropout", 0.0),
        )
        self.criterion = nn.MSELoss()
        self.optimizer_cfg = optimizer_cfg
        self.monitoring_cfg = monitoring_cfg
        self.project_seed = project_seed

        self.eval_storage = self._init_eval_storage()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.adapter(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        base_embeddings, target_embeddings, _ = batch
        predictions = self(base_embeddings)
        loss = self.criterion(predictions, target_embeddings)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_eval_step(batch, stage="val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_eval_step(batch, stage="test")

    def on_validation_epoch_end(self) -> None:
        self._finalize_eval_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._finalize_eval_metrics("test")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer_cfg = OmegaConf.to_container(self.optimizer_cfg.optimizer, resolve=True)
        scheduler_cfg: Optional[Dict[str, Any]] = None
        if "scheduler" in self.optimizer_cfg:
            scheduler_cfg = OmegaConf.to_container(self.optimizer_cfg.scheduler, resolve=True)

        optimizer = self._build_optimizer(optimizer_cfg)
        if scheduler_cfg:
            scheduler = self._build_scheduler(optimizer, scheduler_cfg)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.optimizer_cfg.monitoring.val_metric,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def _shared_eval_step(self, batch: Any, stage: str) -> None:
        base_embeddings, target_embeddings, labels = batch
        predictions = self(base_embeddings)
        loss = self.criterion(predictions, target_embeddings)
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"), on_step=False, on_epoch=True)
        self._store_eval_outputs(stage, predictions, target_embeddings, labels)

    def _init_eval_storage(self) -> Dict[str, Dict[str, List[np.ndarray]]]:
        return {
            "val": {"preds": [], "targets": [], "labels": []},
            "test": {"preds": [], "targets": [], "labels": []},
        }

    def _store_eval_outputs(
        self, stage: str, preds: torch.Tensor, targets: torch.Tensor, labels: torch.Tensor
    ) -> None:
        storage = self.eval_storage[stage]
        storage["preds"].append(preds.detach().cpu().numpy())
        storage["targets"].append(targets.detach().cpu().numpy())
        storage["labels"].append(labels.detach().cpu().numpy())

    def _finalize_eval_metrics(self, stage: str) -> None:
        storage = self.eval_storage[stage]
        if not storage["preds"]:
            return
        preds = np.concatenate(storage["preds"], axis=0)
        teachers = np.concatenate(storage["targets"], axis=0)
        labels = np.concatenate(storage["labels"], axis=0).astype(int)
        num_clusters = len(np.unique(labels))
        if num_clusters > 1 and len(labels) >= num_clusters:
            pred_score = self._compute_v_measure(preds, labels, num_clusters)
            teacher_score = self._compute_v_measure(teachers, labels, num_clusters)
            self.log(f"{stage}_v_measure_pred", pred_score, prog_bar=(stage == "val"), sync_dist=True)
            self.log(f"{stage}_v_measure_teacher", teacher_score, prog_bar=False, sync_dist=True)
        storage["preds"].clear()
        storage["targets"].clear()
        storage["labels"].clear()

    def _compute_v_measure(self, embeddings: np.ndarray, labels: np.ndarray, num_clusters: int) -> float:
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=self.project_seed)
        assignments = kmeans.fit_predict(embeddings)
        return float(v_measure_score(labels, assignments))

    def _build_optimizer(self, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        name = cfg.pop("name").lower()
        if name == "adamw":
            return torch.optim.AdamW(self.parameters(), **cfg)
        if name == "adam":
            return torch.optim.Adam(self.parameters(), **cfg)
        if name == "sgd":
            return torch.optim.SGD(self.parameters(), **cfg)
        raise ValueError(f"Unsupported optimizer: {name}")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
        name = cfg.pop("name").lower()
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cfg)
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, **cfg)
        raise ValueError(f"Unsupported scheduler: {name}")
