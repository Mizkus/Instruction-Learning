from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..models import InstructionAdapterModule


class CSVInferenceRunner:
    """Encode raw texts and run the trained adapter on demand."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.dataset_cfg = cfg.inference.dataset
        self.output_cfg = cfg.inference.output
        self.adapter_cfg = cfg.inference.adapter
        self.teacher_cfg = cfg.inference.teacher
        self.embedding_cfg = cfg.embedding_model
        self.instructions_cfg = cfg.instructions
        self.paths_cfg = cfg.paths
        self.logging_cfg = cfg.logging

        self.instruction_text = self.dataset_cfg.get("instruction") or self.instructions_cfg.topic or None

        self.embedding_model = SentenceTransformer(self.embedding_cfg.name, device=self.embedding_cfg.device)
        if self.embedding_cfg.get("max_seq_length"):
            self.embedding_model.max_seq_length = int(self.embedding_cfg.max_seq_length)
        encode_signature = inspect.signature(self.embedding_model.encode)
        self._supports_prompt = "prompt" in encode_signature.parameters
        self._supports_prompts = "prompts" in encode_signature.parameters

        adapter_ckpt = Path(self.adapter_cfg.checkpoint_path)
        if not adapter_ckpt.exists():
            raise FileNotFoundError(f"Adapter checkpoint not found at {adapter_ckpt}")

        adapter_device = self._resolve_device(self.adapter_cfg.get("device", "auto"))
        self.adapter = InstructionAdapterModule.load_from_checkpoint(adapter_ckpt)
        self.adapter.eval()
        self.adapter.to(adapter_device)
        self.adapter_device = adapter_device

    def run(
        self,
        csv_path: Optional[str] = None,
        text_column: Optional[str] = None,
        output_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Path:
        csv_path = Path(csv_path or self.dataset_cfg.csv_path)
        output_dir = Path(output_dir or self.output_cfg.dir)
        text_column = text_column or self.dataset_cfg.text_column
        limit = limit if limit is not None else self.dataset_cfg.get("limit")

        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        texts = self._load_texts(csv_path, text_column, limit)
        file_hash = self._hash_file(csv_path)
        metadata_path = output_dir / self.output_cfg.metadata_filename
        existing_metadata = self._load_metadata(metadata_path)

        if self._is_cache_valid(existing_metadata, file_hash, len(texts)):
            print("[infer] Cached embeddings detected; skipping recomputation.")
            return output_dir

        base_embeddings, base_seconds = self._encode(texts, prompt=None)
        adapter_embeddings, adapter_seconds = self._run_adapter(base_embeddings)
        teacher_embeddings = None
        teacher_seconds = None
        if self.teacher_cfg.get("compute_embeddings", True) and self.instruction_text:
            teacher_embeddings, teacher_seconds = self._encode(texts, prompt=self.instruction_text)

        self._persist_outputs(output_dir, base_embeddings, adapter_embeddings, teacher_embeddings)
        metadata = self._build_metadata(
            csv_path,
            text_column,
            file_hash,
            len(texts),
            base_seconds,
            adapter_seconds,
            teacher_seconds,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if self.adapter_cfg.get("log_to_mlflow", False):
            self._log_mlflow(metadata)

        return output_dir

    # ------------------------------------------------------------------ helpers
    def _load_texts(self, csv_path: Path, text_column: str, limit: Optional[int]) -> list[str]:
        df = pd.read_csv(csv_path)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not in CSV {csv_path}")
        series = df[text_column]
        if self.dataset_cfg.get("dropna", True):
            series = series.dropna()
        if self.dataset_cfg.get("deduplicate", False):
            series = series.drop_duplicates()
        if limit is not None:
            series = series.head(int(limit))
        return series.astype(str).tolist()

    def _hash_file(self, csv_path: Path) -> str:
        digest = hashlib.md5()
        with csv_path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_metadata(self, metadata_path: Path) -> Optional[Dict]:
        if not metadata_path.exists():
            return None
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _is_cache_valid(self, metadata: Optional[Dict], file_hash: str, num_rows: int) -> bool:
        if not metadata:
            return False
        return (
            metadata.get("file_hash") == file_hash
            and metadata.get("num_samples") == num_rows
            and metadata.get("instruction") == (self.instruction_text or "")
            and metadata.get("adapter_checkpoint") == str(Path(self.adapter_cfg.checkpoint_path))
        )

    def _encode(self, texts: list[str], prompt: Optional[str]) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        encode_kwargs = {}
        if prompt:
            if self._supports_prompt:
                encode_kwargs["prompt"] = prompt
            elif self._supports_prompts:
                encode_kwargs["prompts"] = [prompt] * len(texts)
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.embedding_cfg.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.embedding_cfg.normalize,
            convert_to_numpy=True,
            **encode_kwargs,
        ).astype(np.float32)
        elapsed = time.perf_counter() - start
        return embeddings, elapsed

    def _run_adapter(self, base_embeddings: np.ndarray) -> Tuple[np.ndarray, float]:
        batch_size = int(self.adapter_cfg.get("batch_size", 128))
        data = torch.from_numpy(base_embeddings).float()
        outputs = []
        start = time.perf_counter()
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(data), batch_size), desc="Adapter", unit="batch"):
                batch = data[start_idx : start_idx + batch_size].to(self.adapter_device)
                preds = self.adapter(batch).cpu().numpy()
                outputs.append(preds)
        elapsed = time.perf_counter() - start
        return np.concatenate(outputs, axis=0), elapsed

    def _persist_outputs(
        self,
        output_dir: Path,
        base_embeddings: np.ndarray,
        adapter_embeddings: np.ndarray,
        teacher_embeddings: Optional[np.ndarray],
    ) -> None:
        np.save(output_dir / self.output_cfg.base_filename, base_embeddings)
        np.save(output_dir / self.output_cfg.adapter_filename, adapter_embeddings)
        if teacher_embeddings is not None:
            np.save(output_dir / self.output_cfg.teacher_filename, teacher_embeddings)

    def _build_metadata(
        self,
        csv_path: Path,
        text_column: str,
        file_hash: str,
        num_rows: int,
        base_seconds: float,
        adapter_seconds: float,
        teacher_seconds: Optional[float],
    ) -> Dict:
        metadata = {
            "source_csv": str(csv_path),
            "file_hash": file_hash,
            "text_column": text_column,
            "num_samples": num_rows,
            "instruction": self.instruction_text or "",
            "embedding_model": self.embedding_cfg.name,
            "adapter_checkpoint": str(Path(self.adapter_cfg.checkpoint_path)),
            "adapter_latency_sec": adapter_seconds,
            "base_encode_seconds": base_seconds,
        }
        if teacher_seconds is not None:
            metadata["teacher_encode_seconds"] = teacher_seconds
            if adapter_seconds > 0:
                metadata["latency_speedup"] = teacher_seconds / adapter_seconds
        return metadata

    def _log_mlflow(self, metadata: Dict) -> None:
        tracking_uri = self.logging_cfg.mlflow.tracking_uri
        if not tracking_uri:
            return
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = self.logging_cfg.mlflow.experiment_name or self.cfg.project.name
        mlflow.set_experiment(experiment_name)
        run_name = f"inference-{Path(metadata['source_csv']).stem}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "instruction": metadata["instruction"],
                    "adapter_checkpoint": metadata["adapter_checkpoint"],
                    "num_samples": metadata["num_samples"],
                }
            )
            mlflow.log_metrics(
                {
                    "adapter_latency_sec": metadata["adapter_latency_sec"],
                    "base_encode_seconds": metadata["base_encode_seconds"],
                }
            )
            if "teacher_encode_seconds" in metadata:
                mlflow.log_metric("teacher_encode_seconds", metadata["teacher_encode_seconds"])
            if "latency_speedup" in metadata:
                mlflow.log_metric("latency_speedup", metadata["latency_speedup"])

    @staticmethod
    def _resolve_device(device_cfg: str) -> torch.device:
        if device_cfg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_cfg)


def run_csv_inference(
    cfg: DictConfig,
    csv_path: Optional[str] = None,
    text_column: Optional[str] = None,
    output_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> Path:
    runner = CSVInferenceRunner(cfg)
    return runner.run(csv_path=csv_path, text_column=text_column, output_dir=output_dir, limit=limit)
