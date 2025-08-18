#!/usr/bin/env python3
import os
import re
import json
import glob
import argparse
from pathlib import Path

import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization


def _extract_der_from_name(name: str):
    """Return float DER from filename or None if not present."""
    # strict: capture only a proper number (no trailing dot)
    for pat in [
        r"DiarizationErrorRate=([0-9]+(?:\.[0-9]+)?)\b",
        r"DER=([0-9]+(?:\.[0-9]+)?)\b",
    ]:
        m = re.search(pat, name)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    # tolerant: if previous didn't match, try a looser pattern then strip trailing dots
    m = re.search(r"DiarizationErrorRate=([0-9.]+)", name)
    if m:
        s = m.group(1).rstrip(".")  # fix names like ...=0.287.
        try:
            return float(s)
        except Exception:
            return None
    return None


def pick_best_ckpt(ckpt_dir: Path, explicit: str = None) -> str:
    """Pick checkpoint with lowest DER in filename; fallback to last.ckpt or newest file."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--checkpoint not found: {explicit}")
        return str(p)

    ckpts = sorted(glob.glob(str(ckpt_dir / "*.ckpt")))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt found in {ckpt_dir}")

    scored = []
    for c in ckpts:
        der = _extract_der_from_name(Path(c).name)
        if der is not None:
            scored.append((der, c))

    if scored:
        scored.sort(key=lambda x: x[0])  # lowest DER is best
        return scored[0][1]

    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return str(last)

    # fallback: newest by mtime
    ckpts.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return ckpts[0]


def load_best_params(tune_dir: Path) -> dict:
    """Load tuned hyperparameters (handles several optimizer.json layouts)."""
    opt_path = tune_dir / "optimizer.json"
    if not opt_path.exists():
        raise FileNotFoundError(f"optimizer.json not found in {tune_dir}")
    data = json.loads(opt_path.read_text())

    # common keys across versions
    for k in ("best_params", "best", "params", "bestParameters"):
        if isinstance(data, dict) and k in data:
            data = data[k]
            break
    if not isinstance(data, dict):
        raise ValueError("optimizer.json does not contain a parameter dict.")
    return data


def pretty_print(diar):
    for segment, _, label in diar.itertracks(yield_label=True):
        print(f"[ {segment.start:10.3f} --> {segment.end:10.3f} ] {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="exp", help="Root exp folder")
    ap.add_argument("--wav", required=True, help="Path to input WAV")
    ap.add_argument("--rttm-out", default="diarization.rttm", help="Path to save RTTM")
    ap.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Hugging Face token (or set $HF_TOKEN)")
    ap.add_argument("--checkpoint", default=None, help="Explicit .ckpt path to use (optional)")
    ap.add_argument("--override-threshold", type=float, default=None,
                    help="Override clustering.threshold (optional)")
    ap.add_argument("--enable-tf32", action="store_true", help="Enable TF32 (Ampere+ GPUs)")
    args = ap.parse_args()

    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    exp = Path(args.exp_dir)
    tune_dir = exp / "01_tune_pipeline"
    finetune_dir = exp / "02_finetune_seg"

    # 1) tuned hyper-parameters
    best_params = load_best_params(tune_dir)
    print("Best params:", best_params)

    if args.override_threshold is not None:
        best_params.setdefault("clustering", {})
        best_params["clustering"]["threshold"] = float(args.override_threshold)
        print(f"Overriding clustering.threshold -> {args.override_threshold}")

    # 2) checkpoint
    best_ckpt = pick_best_ckpt(finetune_dir, explicit=args.checkpoint)
    print("Using checkpoint:", best_ckpt)

    # 3) pipeline with your fine-tuned segmentation
    seg_model = Model.from_pretrained(best_ckpt)
    seg_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    pipe = SpeakerDiarization(segmentation=seg_model, use_auth_token=args.hf_token)
    try:
        pipe = pipe.instantiate(best_params)  # newer versions
    except Exception:
        pipe.set_params(**best_params)        # older versions

    # 4) run
    wav = Path(args.wav)
    if not wav.exists():
        raise FileNotFoundError(wav)
    diar = pipe({"audio": str(wav)})

    pretty_print(diar)
    with open(args.rttm_out, "w") as f:
        diar.write_rttm(f)
    print(f"RTTM saved to: {args.rttm_out}")


if __name__ == "__main__":
    main()
