#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a diarization pipeline using a fine-tuned segmentation checkpoint.

This script loads:
  1) fine-tuned **segmentation** checkpoint from Stage 1, and
  2) The tuned **pipeline hyperparameters** from Stage 2 (best_params.json),

then evaluates the full `SpeakerDiarization` pipeline on a chosen split
(train/dev/test) and writes concise reports.

Key points
----------
- **Fail-fast**: If the fine-tuned checkpoint cannot be loaded, the script raises.
- **Version-tolerant**: Tries both `.instantiate(best_params)` and `.set_params(**best_params)`.
- **Clean artifacts**: Writes overall metrics (`metrics.json`), per-file DER (`per_file.csv`),
  optional RTTMs (`rttm/`), and `used_segmentation.txt`.


Example
-------
python evaluate.py \
  --protocol MyDatabase.SpeakerDiarization.MyProtocol \
  --checkpoint ./exp/01_finetune_seg/epoch=00-DiarizationErrorRate=0.295.ckpt \
  --pipeline-params ./exp/02_tune_pipeline/best_params.json \
  --split test \
  --output-dir ./exp/eval_test \
  --write-rttm
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable

import torch
from pyannote.database import registry
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.metrics.diarization import DiarizationErrorRate

log = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# --------------------------------------------------------------------------- #
# Helpers: config, preprocessors, robust loading, RTTM writer
# --------------------------------------------------------------------------- #

def db_cfg_path_from_env() -> Path:
    """Resolve and validate the database config path from the environment.

    Returns
    -------
    Path
        Absolute path to `database.yml`.

    Raises
    ------
    RuntimeError
        If the environment variable is not set.
    FileNotFoundError
        If the file does not exist.
    """
    import os
    env = os.environ.get("PYANNOTE_DATABASE_CONFIG")
    if not env:
        raise RuntimeError("Please export PYANNOTE_DATABASE_CONFIG=/abs/path/to/data/database.yml")
    p = Path(env)
    if not p.exists():
        raise FileNotFoundError(f"PYANNOTE_DATABASE_CONFIG points to missing file: {p}")
    return p


class UriToAudioPath:
    """Minimal preprocessor: map file['uri'] -> '/.../data/audio/{uri}.wav'."""

    def __init__(self, pattern: str):
        self.pattern = pattern

    def __call__(self, file) -> str:
        return self.pattern.format(uri=file["uri"])


def _load_segmentation_strict(ckpt: Path) -> Model:
    """Load a segmentation model strictly from a fine-tuned Lightning checkpoint.

    Strategy
    --------
    A) Try `Model.from_pretrained(ckpt)`.
    B) Otherwise, load the checkpoint `state_dict` into the base
       'pyannote/segmentation-3.0' architecture (non-strict). This still uses
        checkpoint weights. If both fail, raise.

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
        log.info(f"[eval] Using fine-tuned segmentation from: {ckpt}")
        return m
    except Exception as e1:
        log.warning(f"[eval] Direct load failed ({e1}). Trying state_dict path…")

    # Strategy B: state_dict into base architecture
    try:
        obj = torch.load(str(ckpt), map_location="cpu")
        state = obj.get("state_dict", obj)
        base = Model.from_pretrained("pyannote/segmentation")
        missing, unexpected = base.load_state_dict(state, strict=False)
        log.info(
            "[eval] Loaded checkpoint state_dict into base architecture "
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

    Parameters
    ----------
    path : Path
        Path to JSON file.

    Returns
    -------
    dict
        Best parameter mapping.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("best_params", data)


def _iter_split(proto, split: str) -> Iterable[dict]:
    """Yield files from the requested split.

    Parameters
    ----------
    proto : Any
        Pyannote protocol object.
    split : {'train', 'dev', 'test'}
        Protocol split.

    Returns
    -------
    Iterable[dict]
        Items of the chosen split.
    """
    if split == "train":
        return proto.train()
    if split == "dev":
        return proto.development()
    if split == "test":
        return proto.test()
    raise ValueError(f"Unknown split: {split}")


def _write_rttm(uri: str, annotation, path: Path) -> None:
    """Write RTTM lines for an Annotation to `path`.

    Format
    ------
    SPEAKER <uri> 1 <tbeg> <tdur> <NA> <NA> <label> <NA> <NA>

    Parameters
    ----------
    uri : str
        Recording identifier.
    annotation : pyannote.core.Annotation
        Hypothesis annotation.
    path : Path
        Output RTTM path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for segment, _, label in annotation.itertracks(yield_label=True):
            tbeg = float(segment.start)
            tdur = float(segment.duration)
            f.write(f"SPEAKER {uri} 1 {tbeg:.3f} {tdur:.3f} <NA> <NA> {label} <NA> <NA>\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser("Evaluate diarization (fine-tuned segmentation + tuned pipeline)")
    ap.add_argument("--protocol", required=True, help="e.g., MyDatabase.SpeakerDiarization.MyProtocol")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to fine-tuned segmentation .ckpt")
    ap.add_argument("--pipeline-params", type=Path, required=True,
                    help="JSON produced by tuning, e.g., ./exp/02_tune_pipeline/best_params.json")
    ap.add_argument("--split", choices=["train", "dev", "test"], default="test",
                    help="Protocol split to evaluate (default: test)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Directory for reports (default: ./exp/eval_<split>)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Device for segmentation model (default: auto)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of files to evaluate (debug)")
    ap.add_argument("--write-rttm", action="store_true",
                    help="Write hypothesis RTTMs to <output-dir>/rttm/")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()

    outdir = args.output_dir or Path(f"./exp/eval_{args.split}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve database and inject uri->path preprocessor.
    cfg = db_cfg_path_from_env()
    registry.load_database(str(cfg))
    pre = {"audio": UriToAudioPath(str(cfg.parent / "audio" / "{uri}.wav"))}
    proto = registry.get_protocol(args.protocol, preprocessors=pre)

    # Load fine-tuned segmentation checkpoint (fail-fast) and place on device.
    seg = _load_segmentation_strict(args.checkpoint)
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        seg.to("cuda")
        log.info("[eval] Using CUDA for segmentation.")
    else:
        seg.to("cpu")
        log.info("[eval] Using CPU for segmentation.")
    seg.eval()  # pure inference

    # Load tuned pipeline parameters.
    best_params = _read_best_params(args.pipeline_params)

    # Build pipeline and apply params.
    pipeline = SpeakerDiarization(segmentation=seg)
    applied = False
    try:
        pipeline.instantiate(best_params)
        applied = True
    except Exception:
        pass
    if not applied:
        try:
            pipeline.set_params(**best_params)  
            applied = True
        except Exception:
            pass
    if not applied:
        log.warning("[eval] Could not apply best_params via instantiate/set_params; "
                    "continuing with pipeline defaults (this may hurt accuracy).")

    # Evaluate on the chosen split.
    metric = DiarizationErrorRate()
    rows = []

    iterable = _iter_split(proto, args.split)
    count = 0
    with torch.no_grad():
        for f in iterable:
            if args.limit is not None and count >= args.limit:
                break
            uri = f["uri"]
            ref = f["annotation"]
            hyp = pipeline({"audio": f["audio"]})
            uem = f.get("annotated", None)  # optional evaluation mask
            value = metric(ref, hyp, uem=uem)
            rows.append({"uri": uri, "der": float(value)})
            count += 1

            if args.write_rttm:
                _write_rttm(uri, hyp, outdir / "rttm" / f"{uri}.rttm")

    # Aggregate & write reports.
    overall = float(abs(metric))  # overall DER across all evaluated files

    # Aggregated component totals (durations). Guard for version differences.
    def _safe_get(k: str):
        try:
            return float(metric[k])
        except Exception:
            return None

    details: Dict[str, float] = {
        "false_alarm": _safe_get("false alarm"),
        "missed_detection": _safe_get("missed detection"),
        "confusion": _safe_get("confusion"),
        "total": _safe_get("total"),
    }

    # Used segmentation path (auditability).
    (outdir / "used_segmentation.txt").write_text(str(args.checkpoint) + "\n", encoding="utf-8")

    # Per-file CSV.
    with (outdir / "per_file.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=["uri", "der"])
        w.writeheader()
        w.writerows(rows)

    # Summary JSON.
    summary = {
        "DER": overall,
        "details": details,
        "num_files": len(rows),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "pipeline_params": str(args.pipeline_params),
    }
    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    
    try:
        report = metric.report(display=False)
        (outdir / "report.txt").write_text(str(report), encoding="utf-8")
    except Exception:
        pass

    log.info(json.dumps(summary, indent=2))
    print("Evaluation complete. Results written to:", str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
