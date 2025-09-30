# pyannote-code

Two-stage **speaker diarization** workflow built on [pyannote.audio], with:

1) **Stage 1 – Segmentation fine-tuning** (with MUSAN augmentation)  
2) **Stage 2 – Pipeline tuning** on a dev split (single-pass or two-step)

Utilities are included to **prepare a call-level dataset** from chunked audio/RTTM folders and to **evaluate** a tuned pipeline.

---

## Contents

- `train.py` — two-stage training & tuning driver  
- `prepare_call_with_split.py` — build call-level dataset + 80/10/10 splits  
- `evaluate.py` — run evaluation on (train/dev/test) and export reports

---

## Quick start

### 0) Requirements

- Python 3.9+  
- CUDA-enabled PyTorch (recommended)  
- `pyannote.audio` ≥ 3.x, `pytorch-lightning`, `torchaudio`, `optuna` (via pyannote)  

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick your CUDA
pip install pyannote.audio pytorch-lightning optuna rich
```

### 1) Prepare your data (from chunked calls)

Your input tree should look like:

```
SRC/
  roomA/
    call_001/
      call_001_chunk0.wav
      call_001_chunk0.rttm
      call_001_chunk1.wav
      call_001_chunk1.rttm
      ...
  roomB/
    call_047/
      call_047_chunk0.wav
      call_047_chunk0.rttm
      ...
```

Build a call-level dataset and 80/10/10 splits:

```bash
python prepare_call_with_split.py   --src-root "/path/to/chunked/dataset"   --out-root "/path/to/output/call_level"   --audio-ext wav   --chunk-overlap 3.0   --seed 123   --make-uem
```

This writes:

```
OUT/
  audio/    # {uri}.wav (16k mono), one per call
  rttm/     # {uri}.rttm (concatenated timeline)
  uem/      # full-coverage UEMs per split (if --make-uem)
  lists/    # train.lst, dev.lst, test.lst
  database.yml
```

### 2) Environment

Set the following before training/evaluating:

```bash
# Path to the OUT/database.yml created above
export PYANNOTE_DATABASE_CONFIG=/abs/path/to/OUT/database.yml

# Hugging Face token (needed by pyannote pretrained models)
export HF_TOKEN=hf_................................
# or: huggingface-cli login

# (Optional) MUSAN augmentation
export MUSAN_ROOT=/musan
```

### 3) Train & tune (two-stage)

**Lightning multi-GPU (7 GPUs example):**
```bash
# Let Lightning spawn 7 processes (1 per GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train.py   --protocol MyDatabase.SpeakerDiarization.MyProtocol   --output-dir ./exp   --max-epochs 1   --batch-size 8   --num-workers 0   --gpus 7   --allow-tf32   --duration 5.0   --two-step-tuning   --tune-trials 2
```

**torchrun launcher (7 procs; pass `--gpus 1` to the script):**
```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( (RANDOM%50000) + 10000 ))
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp3s0f1np1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --standalone --nproc_per_node=7 train.py   --protocol MyDatabase.SpeakerDiarization.MyProtocol   --output-dir ./exp   --max-epochs 1   --batch-size 8   --num-workers 0   --gpus 1   --allow-tf32   --duration 5.0   --two-step-tuning   --tune-trials 2
```

> The script auto-configures helpful NCCL env defaults (P2P/IB disable, socket IFACE, etc.). You can override via environment variables.

#### Key `train.py` arguments

- `--protocol` — e.g., `MyDatabase.SpeakerDiarization.MyProtocol` (defined in `database.yml`)
- `--output-dir` — experiments root (`./exp` by default)
- `--max-epochs`, `--batch-size`, `--num-workers`, `--duration`, `--lr`
- `--gpus` — 0 for CPU, N for Lightning devices
- `--precision` — `32` | `16` | `bf16` (mixed precision)
- `--allow-tf32` — allow TF32 matmul on Ampere+
- **Augmentation**:
  - `--no-augment` to disable MUSAN
  - `--musan-root` or `MUSAN_ROOT` env
  - SNR ranges & probs: `--snr-noise`, `--snr-music`, `--snr-babble`, `--p-noise`, `--p-music`, `--p-babble`
- **Tuning**:
  - `--tune-trials` (Optuna trials)
  - `--dev-max` to cap number of dev files used during tuning
  - `--two-step-tuning` (segmentation thresholds with oracle clustering → clustering)

**Outputs**

- Stage-1 fine-tuned checkpoint(s) in `./exp/01_finetune_seg/`
- Best checkpoint path: `./exp/01_finetune_seg/best_checkpoint_path.txt`
- Stage-2 tuned params JSON(s) in `./exp/02_tune_pipeline/`  
  - `best_params.json` contains the final pipeline hyperparameters

### 4) Evaluate

Use the fine-tuned segmentation checkpoint and tuned pipeline params:

```bash
python evaluate.py   --protocol MyDatabase.SpeakerDiarization.MyProtocol   --checkpoint ./exp/01_finetune_seg/epoch=00-DiarizationErrorRate=0.295.ckpt   --pipeline-params ./exp/02_tune_pipeline/best_params.json   --split test   --output-dir ./exp/eval_test   --write-rttm
```

**Evaluation artifacts**

- `metrics.json` — overall DER and component totals
- `per_file.csv` — per-URI DER
- `report.txt` — text report (when available)
- `rttm/` — hypothesis RTTMs (if `--write-rttm`)
- `used_segmentation.txt` — path to the checkpoint used

---

## How it works (high level)

- **Data prep** (`prepare_call_with_split.py`)
  - Finds chunk files named `*_chunkN.wav` with matching `.rttm`
  - Concatenates chunks per call, trimming the **head overlap** from chunks 1..N-1 so audio and RTTM align
  - Writes merged call RTTMs and optional full-coverage UEMs per split
  - Produces `database.yml` compatible with pyannote protocols & preprocessors

- **Stage 1 – Segmentation**
  - Loads `pyannote/segmentation-3.0` and fine-tunes on your protocol
  - Optional MUSAN mix-in augmentation (noise/music/babble)

- **Stage 2 – Pipeline tuning**
  - Builds `SpeakerDiarization(segmentation=<fine-tuned>)`
  - Single-pass tuning **or** two-step (thresholds with oracle clustering → clustering)
  - Saves best hyperparameters to JSON

- **Evaluation**
  - Loads the fine-tuned segmentation checkpoint (robust loader)
  - Applies tuned params via `.instantiate()` or `.set_params(**best_params)`
  - Computes DER on chosen split, optional RTTM dump

---

## Repo structure (suggested)

```
pyannote-code/
  README.md
  train.py
  prepare_call_with_split.py
  evaluate.py
  exp/                     # outputs created here
  .gitignore
```

**Suggested `.gitignore`:**
```
__pycache__/
*.py[cod]
.venv/
.env
.envrc
.DS_Store
exp/
*.ckpt
*.pt
.ipynb_checkpoints/
```

---

## Tips & troubleshooting

- **HF auth**: if `Model.from_pretrained(...)` fails with auth, ensure `HF_TOKEN` is set or run `huggingface-cli login`.
- **NCCL errors / multi-GPU**: try setting
  ```bash
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_IFNAME=<your_nic>
  ```
- **“No chunks found”**: check your `*_chunkN.wav` naming and `--audio-ext`. RTTMs must share the same stem.
- **Slow I/O**: increase `--num-workers`; consider SSD for the `audio/` directory.

---

## License

This repo glues together scripts around `pyannote.audio`. Check the upstream project’s license and any pretrained model licenses before redistribution.

---

## Acknowledgements

Built with ❤️ on top of [pyannote.audio] and the broader PyTorch ecosystem.

[pyannote.audio]: https://github.com/pyannote/pyannote-audio
