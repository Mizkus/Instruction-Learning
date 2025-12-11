# Instruction Learning

## Project Description

Instruction Learning builds a lightweight linear adapter that maps base embeddings from Qwen/Qwen3-0.6B-Embedding into instruction-following space. The goal is to keep cluster quality (V-measure) close to the teacher model while avoiding repeated heavy inference. Training relies on the first 2 000 samples of the NYTClustering dataset (topics) with a 70/20/10 train/val/test split, and both teacher vs. adapter quality plus latency speedup are evaluated on that split.

## Technical Section

### Setup

1. Install Poetry and Python 3.10+.
2. Clone the repo and install all dependencies:
   ```bash
   git clone <repo-url>
   cd Instruction-Learning
   poetry install
   poetry run pre-commit install
   ```
3. (Optional) launch MLflow locally for metric tracking:
   ```bash
   poetry run mlflow ui --backend-store-uri mlruns --port 8080
   ```

### Train

1. Prepare data and embeddings via DVC (downloads HuggingFace dataset BrandonZYW/NYTClustering and precomputes base+instruct embeddings with SentenceTransformers):
   ```bash
   poetry run python -m dvc repro
   # or step-by-step
   poetry run python -m instruction_learning.cli download-data
   poetry run python -m instruction_learning.cli preprocess
   ```
2. Train the adapter with PyTorch Lightning + Hydra:
   ```bash
   poetry run python -m instruction_learning.cli train
   ```
   The command logs metrics (`train_loss`, `val_v_measure_pred`, latency numbers) into MLflow and saves the top checkpoint at `artifacts/checkpoints/linear-adapter.ckpt`. After fitting, `trainer.test` evaluates the best checkpoint on the held-out split and logs `test_*` metrics.

### Production preparation

- Checkpoint export: trained weights live at `artifacts/checkpoints/linear-adapter.ckpt` and can be loaded by `InstructionAdapterModule` for downstream serving/inference.
- Data/embedding artifacts are versioned with DVC under `data/` (raw, processed, embeddings) to guarantee reproducibility.
- MLflow keeps experiment history (default tracking URI `http://127.0.0.1:8080`).
- To share trained weights through DVC:
  1. After training, add the checkpoint: `dvc add artifacts/checkpoints/linear-adapter.ckpt` and `git add artifacts/checkpoints/linear-adapter.ckpt.dvc`.
  2. Configure a DVC remote (local path or S3/GDrive/etc.): `dvc remote add -d checkpoints /path/to/storage`.
  3. Run `dvc push` to upload the weights; other machines can call `dvc pull` to download them into `artifacts/checkpoints/linear-adapter.ckpt`.

### Infer

1. Prepare any CSV file that contains a `text` column (rename or pass `--text_column` if needed). You can limit rows via `--limit` and optionally deduplicate/drop missing values by editing `configs/inference/base.yaml`.
2. Run the inference CLI; it will encode the requested texts, reuse cached embeddings if the same file/instruction combo was processed before, run the trained adapter checkpoint, and optionally measure the teacher (instructional) encoder for latency comparison:
   ```bash
   poetry run python -m instruction_learning.cli infer \
       --csv_path /path/to/custom.csv \
       --text_column text \
       --output_dir artifacts/inference
   ```
3. Outputs (all in `artifacts/inference` by default):
   - `base.npy`: raw base embeddings produced by Qwen3 encoder.
   - `adapter.npy`: adapter-projected embeddings ready for instruction-aware consumers.
   - `teacher.npy`: (optional) direct instruction embeddings for latency benchmarking.
   - `metadata.json`: timings, cache hash, instruction, and MLflow run info. Latency metrics are also logged to MLflow when `logging.mlflow.tracking_uri` is configured.
