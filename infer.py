#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run diarization inference with a fine-tuned segmentation + tuned pipeline.

This script loads:
  1)  fine-tuned **segmentation** checkpoint (.ckpt), and
  2) The tuned **pipeline hyperparameters** (best_params.json),

then runs the full `SpeakerDiarization` pipeline on one or more audio files.

Example
-------
python infer.py \
  --inputs file1.wav folder \
  --checkpoint ./exp/01_finetune_seg/epoch=00-DiarizationErrorRate=0.295.ckpt \
  --pipeline-params ./exp/02_tune_pipeline/best_params.json \
  --output-dir ./exp/infer_test \
  --write-json --write-rttm
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization

log = logging.getLogger("infer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# --------------------------------------------------------------------------- #
#  loading & I/O helpers
# --------------------------------------------------------------------------- #

def _load_segmentation_strict(ckpt: Path) -> Model:
    """Load a segmentation model strictly from a fine-tuned Lightning checkpoint.

    Strategy
    --------
    A) Try `Model.from_pretrained(ckpt)`.
    B) Otherwise, load checkpoint `state_dict` into base 'pyannote/segmentation-3.0' (non-strict).
       (Still uses fine-tuned weights.) If both fail, raise.

    Parameters
    ----------
    ckpt : Path
        Path to the fine-tuned checkpoint.

    Returns
    -------
    Model
        A segmentation model carrying checkpoint weights.

    Raises
    ------
    FileNotFoundError
        If `ckpt` doesn't exist.
    RuntimeError
        If both loading strategies fail.
    """
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # Strategy A: direct load
    try:
        m = Model.from_pretrained(str(ckpt))
        log.info(f"[infer] Using fine-tuned segmentation from: {ckpt}")
        return m
    except Exception as e1:
        log.warning(f"[infer] Direct load failed ({e1}). Trying state_dict path…")

    # Strategy B: state_dict into base architecture
    try:
        obj = torch.load(str(ckpt), map_location="cpu")
        state = obj.get("state_dict", obj)
        base = Model.from_pretrained("pyannote/segmentation-3.0")
        missing, unexpected = base.load_state_dict(state, strict=False)
        log.info(
            "[infer] Loaded checkpoint state_dict into base architecture "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )
        return base
    except Exception as e2:
        raise RuntimeError(
            f"Failed to load segmentation checkpoint '{ckpt}' "
            f"via direct and state_dict strategies: ({e1}) / ({e2})"
        ) from e2


def _read_best_params(path: Path) -> Dict:
    """Read tuned pipeline parameters from JSON.

    Accepts either:
      - {"best_params": {...}} or
      - {...} directly.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("best_params", data)


def _collect_audio(inputs: List[Path]) -> List[Path]:
    """Expand a mix of files/folders into a list of audio files."""
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    out: List[Path] = []
    for p in inputs:
        p = Path(p)
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
        elif p.is_dir():
            out.extend([f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts])
        else:
            log.warning(f"[infer] Skipping non-audio path: {p}")
    # Deduplicate & sort
    out = sorted(set(out))
    if not out:
        raise FileNotFoundError("No audio files found under the provided inputs.")
    return out


def _write_rttm(uri: str, annotation, path: Path) -> None:
    """Write RTTM lines for an Annotation to `path`.

    Format
    ------
    SPEAKER <uri> 1 <tbeg> <tdur> <NA> <NA> <label> <NA> <NA>
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for segment, _, label in annotation.itertracks(yield_label=True):
            tbeg = float(segment.start)
            tdur = float(segment.duration)
            f.write(f"SPEAKER {uri} 1 {tbeg:.3f} {tdur:.3f} <NA> <NA> {label} <NA> <NA>\n")


def _annotation_to_json(annotation) -> Dict:
    """Convert a pyannote Annotation to a simple JSON structure."""
    segments = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": str(label),
        })
    speakers = [str(s) for s in annotation.labels()]
    return {"speakers": speakers, "segments": segments}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser("Diarization inference (fine-tuned segmentation + tuned pipeline)")
    ap.add_argument("--inputs", nargs="+", type=Path, required=True,
                    help="Audio files and/or directories (recursively searched).")
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to fine-tuned segmentation .ckpt")
    ap.add_argument("--pipeline-params", type=Path, required=True,
                    help="JSON produced by tuning, e.g., ./exp/02_tune_pipeline/best_params.json")
    ap.add_argument("--output-dir", type=Path, default=Path("./exp/infer"),
                    help="Directory for outputs (RTTM/JSON/manifests)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Device for segmentation model (default: auto)")
    ap.add_argument("--write-rttm", action="store_true",
                    help="Write RTTM files to <output-dir>/rttm/")
    ap.add_argument("--write-json", action="store_true",
                    help="Write JSON files to <output-dir>/json/")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect audio files
    audio_files = _collect_audio(args.inputs)
    log.info(f"[infer] Found {len(audio_files)} audio file(s).")

    # Load fine-tuned segmentation checkpoint (fail-fast) and place on device.
    seg = _load_segmentation_strict(args.checkpoint)
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        seg.to("cuda")
        log.info("[infer] Using CUDA for segmentation.")
    else:
        seg.to("cpu")
        log.info("[infer] Using CPU for segmentation.")
    seg.eval()  # pure inference

    # Load tuned pipeline parameters and build pipeline
    best_params = _read_best_params(args.pipeline_params)
    pipeline = SpeakerDiarization(segmentation=seg)
    applied = False
    try:
        pipeline.instantiate(best_params)
        applied = True
    except Exception:
        pass
    if not applied:
        try:
            pipeline.set_params(**best_params)  # older pyannote versions
            applied = True
        except Exception:
            pass
    if not applied:
        log.warning("[infer] Could not apply best_params via instantiate/set_params; "
                    "continuing with pipeline defaults.")

    # Write audit info
    (args.output_dir / "used_segmentation.txt").write_text(str(args.checkpoint) + "\n", encoding="utf-8")
    (args.output_dir / "used_params.txt").write_text(str(args.pipeline_params) + "\n", encoding="utf-8")

    # Inference loop
    summary_rows: List[Dict] = []
    with torch.no_grad():
        for i, wav in enumerate(audio_files, 1):
            uri = wav.stem
            log.info(f"[infer] ({i}/{len(audio_files)}) {wav}")
            hyp = pipeline({"audio": str(wav)})

            # Save RTTM
            if args.write_rttm:
                _write_rttm(uri, hyp, args.output_dir / "rttm" / f"{uri}.rttm")

            # Save JSON
            if args.write_json:
                js = _annotation_to_json(hyp)
                json_path = args.output_dir / "json" / f"{uri}.json"
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(json.dumps(js, indent=2), encoding="utf-8")

            # Summary stats
            num_speakers = len(hyp.labels())
            # Count tracks (one row per RTTM line)
            num_tracks = sum(1 for _ in hyp.itertracks(yield_label=True))
            summary_rows.append({
                "file": str(wav),
                "num_speakers": num_speakers,
                "num_hyp_segments": num_tracks,
            })

    # Write summary CSV
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=["file", "num_speakers", "num_hyp_segments"])
        w.writeheader()
        w.writerows(summary_rows)

    log.info(f"[infer] Done. Wrote outputs to: {args.output_dir}")
    print("Inference complete. Results written to:", str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
