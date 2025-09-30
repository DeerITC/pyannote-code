#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stage speaker diarization training & tuning.

Stage 1 (training):
    Fine-tune a segmentation model on protocol, with MUSAN augmentation.

Stage 2 (tuning):
    Tune a SpeakerDiarization pipeline on the dev set, reusing the Stage-1
    fine-tuned segmentation weights. Optionally perform a two-step tuning
    (segmentation-first with oracle clustering, then clustering).

Environment (example):
    export PYANNOTE_DATABASE_CONFIG=/abs/path/to/data/database.yml
    export HF_TOKEN=...
    export MUSAN_ROOT=/musan

Launch examples:

  # Let Lightning spawn 7 processes (1 per GPU)
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  python train.py \
    --protocol MyDatabase.SpeakerDiarization.MyProtocol \
    --output-dir ./exp \
    --max-epochs 1 \
    --batch-size 8 \
    --num-workers 0 \
    --gpus 7 \
    --allow-tf32 \
    --duration 5.0 \
    --two-step-tuning \
    --tune-trials 2

  # OR: Use torchrun to launch 7 processes; pass --gpus 1 to the script
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=$(( (RANDOM%50000) + 10000 ))
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_IFNAME=enp3s0f1np1
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  torchrun --standalone --nproc_per_node=7 train.py \
    --protocol MyDatabase.SpeakerDiarization.MyProtocol \
    --output-dir ./exp \
    --max-epochs 1 \
    --batch-size 8 \
    --num-workers 0 \
    --gpus 1 \
    --allow-tf32 \
    --duration 5.0 \
    --two-step-tuning \
    --tune-trials 2
"""
from __future__ import annotations

# --- numpy edge case workaround on some stacks ---
import numpy as np
if not hasattr(np, "NAN"):
    np.NAN = np.nan




import os
import json
import math
import random
import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from types import SimpleNamespace

import torch
import torchaudio
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

from pyannote.database import registry
from pyannote.pipeline import Optimizer
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.tasks import Segmentation
from pyannote.audio import Model

# -----------------------------------------------------------------------------
# Logging & warnings
# -----------------------------------------------------------------------------
log = logging.getLogger("train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
warnings.filterwarnings("ignore", message=".*delim_whitespace.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*deprecated.*")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# -----------------------------------------------------------------------------
# Distributed / NCCL config (mirrors your working smoke test)
# -----------------------------------------------------------------------------
def _configure_dist_env() -> None:
    # Key fix for your system: disable P2P (you can try re-enabling later)
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    # Disable Infiniband; use sockets
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    # Helpful NCCL/Torch robustness flags
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    # Keep NCCL logs lighter unless debugging
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    # Avoid rendezvous collisions
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(random.randint(10000, 60000))
    # Pin to your NIC if not already pinned
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "enp3s0f1np1")
    # Unbuffered stdout helps in multi-proc logs
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
@dataclass
class Args:
    protocol: str
    output_dir: Path
    seed: int
    max_epochs: int
    batch_size: int
    num_workers: int
    duration: float
    max_speakers_per_chunk: int
    lr: float
    gpus: int
    precision: str
    allow_tf32: bool
    tune_trials: int
    dev_max: Optional[int]
    no_augment: bool
    musan_root: Optional[Path]
    snr_noise: Tuple[float, float]
    snr_music: Tuple[float, float]
    snr_babble: Tuple[float, float]
    p_noise: float
    p_music: float
    p_babble: float
    two_step_tuning: bool  # attempt segmentation-first then clustering


def parse_args() -> Args:
    p = argparse.ArgumentParser("Two-stage diarization (seg fine-tune -> pipeline tune)")
    p.add_argument("--protocol", required=True, help="e.g., MyDatabase.SpeakerDiarization.MyProtocol")
    p.add_argument("--output-dir", type=Path, default=Path("./exp"))

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--max-speakers-per-chunk", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--gpus", type=int, default=1, help="0 -> CPU")
    p.add_argument("--precision", default="32", choices=["32", "16", "bf16"])
    p.add_argument("--allow-tf32", action="store_true")

    p.add_argument("--tune-trials", type=int, default=15, help="Optuna trials (per phase if two-step)")
    p.add_argument("--dev-max", type=int, default=None, help="Cap number of dev files during tuning")

    

    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--musan-root", type=Path, default=None)

    # MUSAN SNR ranges and probabilities
    p.add_argument("--snr-noise", type=float, nargs=2, default=(5.0, 20.0))
    p.add_argument("--snr-music", type=float, nargs=2, default=(5.0, 20.0))
    p.add_argument("--snr-babble", type=float, nargs=2, default=(10.0, 20.0))
    p.add_argument("--p-noise", type=float, default=0.4)
    p.add_argument("--p-music", type=float, default=0.4)
    p.add_argument("--p-babble", type=float, default=0.4)

    p.add_argument(
        "--two-step-tuning",
        action="store_true",
        help="Attempt segmentation-first (oracle clustering) then clustering. "
             "Falls back gracefully to single-pass if unsupported.",
    )

    a = p.parse_args()
    return Args(
        protocol=a.protocol,
        output_dir=a.output_dir,
        seed=a.seed,
        max_epochs=a.max_epochs,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        duration=a.duration,
        max_speakers_per_chunk=a.max_speakers_per_chunk,
        lr=a.lr,
        gpus=a.gpus,
        precision=a.precision,
        allow_tf32=a.allow_tf32,
        tune_trials=a.tune_trials,
        dev_max=a.dev_max,
        no_augment=a.no_augment,
        musan_root=a.musan_root,
        snr_noise=tuple(a.snr_noise),
        snr_music=tuple(a.snr_music),
        snr_babble=tuple(a.snr_babble),
        p_noise=a.p_noise,
        p_music=a.p_music,
        p_babble=a.p_babble,
        two_step_tuning=a.two_step_tuning,
    )


# -----------------------------------------------------------------------------
# Helpers & preprocessors
# -----------------------------------------------------------------------------
def db_cfg_path_from_env() -> Path:
    env = os.environ.get("PYANNOTE_DATABASE_CONFIG")
    if not env:
        raise RuntimeError("Please export PYANNOTE_DATABASE_CONFIG=/abs/path/to/data/database.yml")
    p = Path(env)
    if not p.exists():
        raise FileNotFoundError(f"PYANNOTE_DATABASE_CONFIG points to missing file: {p}")
    return p


class UriToAudioPath:
    """Minimal preprocessor: resolves file['uri'] to an audio file path."""
    def __init__(self, pattern: str):
        self.pattern = pattern

    def __call__(self, file) -> str:
        return self.pattern.format(uri=file["uri"])


def build_dev_files_with_audio(proto, limit: Optional[int] = None) -> List[Dict]:
    files: List[Dict] = []
    for f in proto.development():
        d = {"uri": f["uri"], "annotation": f["annotation"]}
        if "annotated" in f:
            d["annotated"] = f["annotated"]
        d["audio"] = f["audio"]  # injected by our preprocessor
        files.append(d)
        if limit is not None and len(files) >= limit:
            break
    return files


# -----------------------------------------------------------------------------
# MUSAN augmentation
# -----------------------------------------------------------------------------
def _list_audio_files(root: Path) -> List[Path]:
    exts = (".wav", ".flac", ".mp3", ".ogg")
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


class MusanAugment:
    """Lightweight MUSAN background mixer (noise/music/babble)."""
    def __init__(
        self,
        musan_root: Path,
        p_noise: float = 0.4,
        p_music: float = 0.4,
        p_babble: float = 0.4,
        snr_noise: Tuple[float, float] = (5.0, 20.0),
        snr_music: Tuple[float, float] = (5.0, 20.0),
        snr_babble: Tuple[float, float] = (10.0, 20.0),
    ):
        self.p_noise = p_noise
        self.p_music = p_music
        self.p_babble = p_babble
        self.snr_noise = snr_noise
        self.snr_music = snr_music
        self.snr_babble = snr_babble

        self._train = True  # pyannote toggles augmentation in val/test

        self.noise = _list_audio_files(musan_root / "noise") if (musan_root / "noise").is_dir() else []
        self.music = _list_audio_files(musan_root / "music") if (musan_root / "music").is_dir() else []
        self.speech = _list_audio_files(musan_root / "speech") if (musan_root / "speech").is_dir() else []

        if len(self.noise) + len(self.music) + len(self.speech) == 0:
            log.warning("MUSAN root provided but found no audio under noise/music/speech.")

    def train(self, mode: bool = True):
        self._train = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp((x ** 2).mean(dim=-1, keepdim=True), min=1e-12))

    def _load_random_segment(self, files: List[Path], target_len: int, sr: int, device) -> Optional[torch.Tensor]:
        if not files:
            return None
        path = random.choice(files)
        wav, file_sr = torchaudio.load(str(path))  # (C, N) on CPU
        if wav.numel() == 0:
            return None
        wav = wav.mean(dim=0, keepdim=True) if wav.shape[0] > 1 else wav[:1, :]
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_freq=file_sr, new_freq=sr)
        wav = wav.to(device)

        n = wav.shape[-1]
        if n >= target_len:
            start = random.randint(0, n - target_len)
            wav = wav[:, start:start + target_len]
        else:
            reps = math.ceil(target_len / n)
            wav = wav.repeat(1, reps)[:, :target_len]
        return wav

    def _mix(self, clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        if noise.dim() == 2:
            noise = noise.unsqueeze(0)  # (1,1,T)
        if noise.shape[1] != 1:
            noise = noise[:, :1, :]

        Px = self._rms(clean.squeeze(1)).squeeze(-1)
        Pn = self._rms(noise.squeeze(1)).squeeze(-1)
        snr_lin = 10.0 ** (snr_db / 10.0)
        a = (Px / (Pn * torch.sqrt(torch.tensor(snr_lin, device=clean.device)))).view(-1, 1, 1)
        return torch.clamp(clean + a * noise, -1.0, 1.0)

    def __call__(self, *args: Any, **kwargs: Any):
        if not self._train:
            if "samples" in kwargs:
                return SimpleNamespace(
                    samples=kwargs["samples"],
                    targets=kwargs.get("targets", None),
                    sample_rate=int(kwargs.get("sample_rate", 16000)),
                )
            if args and isinstance(args[0], dict):
                return args[0]
            return args[0] if args else kwargs

        # Kwargs path
        if "samples" in kwargs:
            samples = kwargs["samples"]
            targets = kwargs.get("targets", None)
            sr = int(kwargs.get("sample_rate", 16000))

            x = samples
            orig_dims = x.dim()
            if x.dim() == 2:
                x = x.unsqueeze(1)
            elif x.dim() != 3:
                return SimpleNamespace(samples=samples, targets=targets, sample_rate=sr)

            if x.shape[1] != 1:
                x = x.mean(dim=1, keepdim=True)

            B, _, T = x.shape
            device = x.device
            x_out = x.clone()

            cats, probs, pools, snr_ranges = [], [], [], []
            if self.noise and self.p_noise > 0:
                cats.append("noise"); probs.append(self.p_noise); pools.append(self.noise); snr_ranges.append(self.snr_noise)
            if self.music and self.p_music > 0:
                cats.append("music"); probs.append(self.p_music); pools.append(self.music); snr_ranges.append(self.snr_music)
            if self.speech and self.p_babble > 0:
                cats.append("babble"); probs.append(self.p_babble); pools.append(self.speech); snr_ranges.append(self.snr_babble)

            if cats:
                total = sum(probs)
                probs = [p / total for p in probs]
                for b in range(B):
                    if random.random() < 0.5:
                        continue
                    r, acc, idx = random.random(), 0.0, 0
                    for i, p in enumerate(probs):
                        acc += p
                        if r <= acc:
                            idx = i
                            break
                    pool = pools[idx]
                    snr_low, snr_high = snr_ranges[idx]
                    snr = random.uniform(snr_low, snr_high)
                    seg = self._load_random_segment(pool, target_len=T, sr=sr, device=device)
                    if seg is not None:
                        x_out[b:b+1] = self._mix(x_out[b:b+1], seg.unsqueeze(0), snr_db=snr)

            out = x_out.squeeze(1) if orig_dims == 2 else x_out
            return SimpleNamespace(samples=out, targets=targets, sample_rate=sr)

        # Dict path (legacy).
        if args and isinstance(args[0], dict):
            batch = args[0]
            if "waveform" not in batch or "sample_rate" not in batch:
                return batch

            x = batch["waveform"]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            elif x.dim() != 3:
                return batch
            if x.shape[1] != 1:
                x = x.mean(dim=1, keepdim=True)

            B, _, T = x.shape
            sr = int(batch["sample_rate"])
            device = x.device
            x_out = x.clone()

            cats, probs, pools, snr_ranges = [], [], [], []
            if self.noise and self.p_noise > 0:
                cats.append("noise"); probs.append(self.p_noise); pools.append(self.noise); snr_ranges.append(self.snr_noise)
            if self.music and self.p_music > 0:
                cats.append("music"); probs.append(self.p_music); pools.append(self.music); snr_ranges.append(self.snr_music)
            if self.speech and self.p_babble > 0:
                cats.append("babble"); probs.append(self.p_babble); pools.append(self.speech); snr_ranges.append(self.snr_babble)

            if cats:
                total = sum(probs)
                probs = [p / total for p in probs]
                for b in range(B):
                    if random.random() < 0.5:
                        continue
                    r, acc, idx = random.random(), 0.0, 0
                    for i, p in enumerate(probs):
                        acc += p
                        if r <= acc:
                            idx = i
                            break
                    pool = pools[idx]
                    snr_low, snr_high = snr_ranges[idx]
                    snr = random.uniform(snr_low, snr_high)
                    seg = self._load_random_segment(pool, target_len=T, sr=sr, device=device)
                    if seg is not None:
                        x_out[b:b+1] = self._mix(x_out[b:b+1], seg.unsqueeze(0), snr_db=snr)

            batch["waveform"] = x_out
            return batch

        return args[0] if args else kwargs


# -----------------------------------------------------------------------------
# Stage 1 — segmentation fine-tuning
# -----------------------------------------------------------------------------
def _map_precision(p: str) -> str:
    """Map simple flags to PL precision strings."""
    if p == "16":
        return "16-mixed"
    if p.lower() in {"bf16", "bfloat16"}:
        return "bf16-mixed"
    return "32-true"


def finetune_segmentation(
    protocol_name: str,
    outdir: Path,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
    duration: float,
    max_speakers_per_chunk: int,
    lr: float,
    gpus: int,
    precision: str,
    allow_tf32: bool,
    no_augment: bool,
    musan_root: Optional[Path],
    snr_noise: Tuple[float, float],
    snr_music: Tuple[float, float],
    snr_babble: Tuple[float, float],
    p_noise: float,
    p_music: float,
    p_babble: float,
    cfg_path: Path,
) -> Path:
    # Resolve WAVs via preprocessor.
    audio_pat = str(cfg_path.parent / "audio" / "{uri}.wav")
    pre = {"audio": UriToAudioPath(audio_pat)}
    protocol = registry.get_protocol(protocol_name, preprocessors=pre)

    # Augmentation
    augmentation = None
    if not no_augment:
        musan_dir = musan_root if musan_root else Path(os.environ["MUSAN_ROOT"]) if "MUSAN_ROOT" in os.environ else None
        if musan_dir and musan_dir.exists():
            augmentation = MusanAugment(
                musan_root=musan_dir,
                p_noise=p_noise, p_music=p_music, p_babble=p_babble,
                snr_noise=snr_noise, snr_music=snr_music, snr_babble=snr_babble,
            )
            log.info("MUSAN augmentation enabled.")
        else:
            log.warning("MUSAN not found; training without augmentation.")

    task = Segmentation(
        protocol=protocol,
        duration=duration,
        max_speakers_per_chunk=max_speakers_per_chunk,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation=augmentation,
    )

    # Load base segmentation and fine-tune.
    model = Model.from_pretrained("pyannote/segmentation-3.0")
    if hasattr(model, "learning_rate"):
        model.learning_rate = lr
    model.task = task

    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(outdir),
        filename="{epoch:02d}-{DiarizationErrorRate:.3f}",
        monitor="DiarizationErrorRate",
        mode="min",
        save_top_k=-1,
        save_last=True,
    )
    early_cb = EarlyStopping(monitor="DiarizationErrorRate", mode="min", patience=15, verbose=True)
    bar_cb = RichProgressBar()

    if allow_tf32:
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    accelerator = "gpu" if (gpus and torch.cuda.is_available()) else "cpu"
    devices = gpus if accelerator == "gpu" else None

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy="ddp" if accelerator == "gpu" else "auto",
        callbacks=[checkpoint_cb, early_cb, bar_cb],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir=str(outdir),
        precision=_map_precision(precision),
    )

    log.info("Starting segmentation fine-tuning …")
    trainer.fit(model)
    log.info("Fine-tuning finished.")

    if checkpoint_cb.best_model_path:
        log.info(f"Best checkpoint: {checkpoint_cb.best_model_path}")
        return Path(checkpoint_cb.best_model_path)

    last = outdir / "last.ckpt"
    return last if last.exists() else outdir


# -----------------------------------------------------------------------------
# Stage 2 — pipeline tuning (using fine-tuned segmentation)
# -----------------------------------------------------------------------------
def _load_segmentation_from_ckpt(ckpt: Path) -> Model:
    if ckpt is None:
        raise ValueError("segmentation_ckpt is None; expected a path to a fine-tuned checkpoint.")
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {ckpt}")

    # Strategy A: direct load
    try:
        m = Model.from_pretrained(str(ckpt))
        log.info(f"[tuning] Using fine-tuned segmentation from: {ckpt}")
        return m
    except Exception as e1:
        log.warning(f"[tuning] Direct load failed ({e1}). Trying state_dict path…")

    # Strategy B: load state_dict into base architecture
    try:
        obj = torch.load(str(ckpt), map_location="cpu")
        state = obj.get("state_dict", obj)
        base = Model.from_pretrained("pyannote/segmentation-3.0")
        missing, unexpected = base.load_state_dict(state, strict=False)
        log.info(
            "[tuning] Loaded checkpoint state_dict into base architecture "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )
        return base
    except Exception as e2:
        raise RuntimeError(
            f"Failed to load segmentation checkpoint '{ckpt}': "
            f"direct and state_dict paths both failed ({e1}) / ({e2})."
        ) from e2


def optimize_pipeline(
    protocol_name: str,
    outdir: Path,
    cfg_path: Path,
    tune_trials: int,
    dev_max: Optional[int],
    segmentation_ckpt: Optional[Path],
    two_step: bool,
) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)

    # Build dev file list with audio paths.
    audio_pat = str(cfg_path.parent / "audio" / "{uri}.wav")
    pre = {"audio": UriToAudioPath(audio_pat)}
    proto = registry.get_protocol(protocol_name, preprocessors=pre)
    dev_files = build_dev_files_with_audio(proto, limit=dev_max)
    log.info("[tuning] dev files: %d", len(dev_files))

    # Load fine-tuned segmentation for the pipeline.
    seg_model = _load_segmentation_from_ckpt(segmentation_ckpt) if segmentation_ckpt else Model.from_pretrained("pyannote/segmentation-3.0")

    # Attempt two-step tuning if requested & supported.
    if two_step:
        try:
            from pyannote.audio.pipelines.speaker_diarization import OracleClustering  # type: ignore

            # Step A: segmentation thresholds with oracle clustering.
            pipe_seg = SpeakerDiarization(segmentation=seg_model, clustering=OracleClustering())
            optA = Optimizer(pipe_seg)
            try:
                optA.tune(dev_files, n_trials=tune_trials)
            except TypeError:
                optA.tune(dev_files, n_iterations=tune_trials)

            # Extract best params from step A.
            bestA = None
            for attr in ("best_params_", "best_params", "best_hparams"):
                if hasattr(optA, attr) and isinstance(getattr(optA, attr), dict):
                    bestA = getattr(optA, attr)
                    break
            if bestA is None and hasattr(optA, "study_") and hasattr(optA.study_, "best_params"):
                bestA = optA.study_.best_params
            bestA = bestA or {}

            # Step B: clustering tuning (segmentation params fixed).
            pipe_full = SpeakerDiarization(segmentation=seg_model)
            try:
                pipe_full.set_params(**bestA)  # no-op if unavailable
            except Exception:
                pass

            optB = Optimizer(pipe_full)
            try:
                optB.tune(dev_files, n_trials=tune_trials)
            except TypeError:
                optB.tune(dev_files, n_iterations=tune_trials)

            # Collect final best params from step B.
            final_params = None
            for attr in ("best_params_", "best_params", "best_hparams"):
                if hasattr(optB, attr) and isinstance(getattr(optB, attr), dict):
                    final_params = getattr(optB, attr)
                    break
            if final_params is None and hasattr(optB, "study_") and hasattr(optB.study_, "best_params"):
                final_params = optB.study_.best_params
            final_params = final_params or {}

            (outdir / "optimizer_stepA.json").write_text(json.dumps({"best_params": bestA}, indent=2), encoding="utf-8")
            (outdir / "optimizer_stepB.json").write_text(json.dumps({"best_params": final_params}, indent=2), encoding="utf-8")
            (outdir / "best_params.json").write_text(json.dumps(final_params, indent=2), encoding="utf-8")
            log.info("[tuning] two-step tuning completed.")
            return final_params

        except Exception as e:
            log.warning(f"[tuning] Two-step tuning unavailable or failed ({e}). Falling back to single-pass.")

    # Single-pass tuning
    pipeline = SpeakerDiarization(segmentation=seg_model)
    opt = Optimizer(pipeline)
    try:
        opt.tune(dev_files, n_trials=tune_trials)
    except TypeError:
        opt.tune(dev_files, n_iterations=tune_trials)

    best_params = None
    for attr in ("best_params_", "best_params", "best_hparams"):
        if hasattr(opt, attr) and isinstance(getattr(opt, attr), dict):
            best_params = getattr(opt, attr)
            break
    if best_params is None and hasattr(opt, "study_") and hasattr(opt.study_, "best_params"):
        best_params = opt.study_.best_params
    best_params = best_params or {}

    (outdir / "optimizer.json").write_text(json.dumps({"best_params": best_params}, indent=2), encoding="utf-8")
    (outdir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    log.info("[tuning] single-pass tuning completed.")
    return best_params


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    _configure_dist_env()  # ensure NCCL/distributed defaults match smoke test
    args = parse_args()

    seed_everything(args.seed, workers=True)
    log.info(f"Seed={args.seed} | Torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"CUDA {torch.version.cuda} | GPU0: {torch.cuda.get_device_name(0)}")

    cfg_path = db_cfg_path_from_env()
    registry.load_database(str(cfg_path))

    # Stage 1: segmentation fine-tuning (with optional MUSAN).
    stage1_dir = args.output_dir / "01_finetune_seg"
    best_ckpt = finetune_segmentation(
        protocol_name=args.protocol,
        outdir=stage1_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        duration=args.duration,
        max_speakers_per_chunk=args.max_speakers_per_chunk,
        lr=args.lr,
        gpus=args.gpus,
        precision=args.precision,
        allow_tf32=args.allow_tf32,
        no_augment=args.no_augment,
        musan_root=args.musan_root,
        snr_noise=args.snr_noise,
        snr_music=args.snr_music,
        snr_babble=args.snr_babble,
        p_noise=args.p_noise,
        p_music=args.p_music,
        p_babble=args.p_babble,
        cfg_path=cfg_path,
    )
    (stage1_dir / "best_checkpoint_path.txt").write_text(str(best_ckpt), encoding="utf-8")

    # Stage 2: pipeline tuning (uses Stage-1 segmentation).
    stage2_dir = args.output_dir / "02_tune_pipeline"
    best_params = optimize_pipeline(
        protocol_name=args.protocol,
        outdir=stage2_dir,
        cfg_path=cfg_path,
        tune_trials=args.tune_trials,
        dev_max=args.dev_max,
        segmentation_ckpt=best_ckpt,
        two_step=args.two_step_tuning,
    )
    (stage2_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    log.info("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.")