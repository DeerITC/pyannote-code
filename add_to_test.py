#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append an additional nested dataset *directly* to the **test** split
of a pyannote-compatible folder prepared by `prepare_dataset.py`.

What it does
------------
- Recursively scans `--new-root` for `(audio, rttm)` pairs.
- Derives URIs (same scheme as the original script) from the relative path.
- Optionally prefixes URIs via `--uri-prefix` to avoid collisions.
- Links or copies new audio to `<output>/audio/{uri}.wav`.
- Appends normalized RTTM entries to `<output>/rttm/test.rttm` (recording
  field forced to `{uri}`).
- Appends full-coverage segments to `<output>/uem/test.uem`.
- Appends URIs to `<output>/lists/test.lst` (dedup-aware, optional sorting).

It **does not** touch train/dev files.

Example
-------
python add_to_test.py \
  --new-root /path/to/new_nested_data \
  --output-dir ./data \
  --uri-prefix NEW \
  --copy

Requirements
------------
- Python 3.9+
- torchaudio (optional; falls back to `wave` for PCM WAVs)
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import torchaudio  # type: ignore
    _HAVE_TORCHAUDIO = True
except Exception:
    _HAVE_TORCHAUDIO = False

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recording:
    uri: str
    wav: Path
    rttm: Path

# ---------------------------------------------------------------------------
# Utilities (kept intentionally in sync with prepare_dataset.py)
# ---------------------------------------------------------------------------

def sanitize_component(component: str) -> str:
    out = component.replace(" ", "_")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_+.@"
    out = "".join(ch for ch in out if ch in allowed)
    return out or "_"


def derive_uri(rel_parts: Sequence[str]) -> str:
    if not rel_parts:
        raise ValueError("empty rel_parts; cannot derive URI")
    *folders, filename = list(rel_parts)
    stem = Path(filename).stem
    safe = [sanitize_component(p) for p in folders]
    safe_stem = sanitize_component(stem)
    return "__".join([*safe, safe_stem]) if folders else safe_stem


def audio_duration_seconds(path: Path) -> float:
    if _HAVE_TORCHAUDIO:
        try:
            info = torchaudio.info(str(path))
            return float(info.num_frames) / float(info.sample_rate)
        except Exception:
            pass
    import wave
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except Exception:
            pass
    if copy:
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, dst)
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def scan_new_dataset(new_root: Path, audio_exts: Sequence[str]) -> List[Recording]:
    root = new_root.resolve()
    out: List[Recording] = []
    for wav in root.rglob("*"):
        if not wav.is_file() or wav.suffix not in audio_exts:
            continue
        rttm = wav.with_suffix(".rttm")
        if not rttm.exists():
            logging.warning("skip (no RTTM next to audio): %s", wav)
            continue
        rel = wav.relative_to(root)
        rel_parts = rel.as_posix().split("/")
        uri = derive_uri(rel_parts)
        out.append(Recording(uri=uri, wav=wav.resolve(), rttm=rttm.resolve()))
    out.sort(key=lambda r: (str(r.wav), str(r.rttm)))
    # de-dup within the new dataset (keep first occurrence)
    seen = {}
    unique: List[Recording] = []
    for r in out:
        if r.uri in seen:
            logging.warning("duplicate URI within new-root: %s (keeping first)", r.uri)
            continue
        seen[r.uri] = True
        unique.append(r)
    return unique


def read_existing_test_uris(lst_path: Path) -> List[str]:
    if not lst_path.exists():
        return []
    return [ln.strip() for ln in lst_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_list_dedup(lst_path: Path, uris: Iterable[str], sort_list: bool) -> None:
    existing = set(read_existing_test_uris(lst_path))
    merged = list(existing.union(uris))
    if sort_list:
        merged.sort()
    ensure_dir(lst_path.parent)
    lst_path.write_text("\n".join(merged) + "\n", encoding="utf-8")


def append_rttm_normalized(rttm_out: Path, recs: List[Recording]) -> None:
    ensure_dir(rttm_out.parent)
    with rttm_out.open("a", encoding="utf-8") as out:
        for r in recs:
            with r.rttm.open("r", encoding="utf-8") as f:
                for ln in f:
                    if not ln.startswith("SPEAKER "):
                        continue
                    parts = ln.strip().split()
                    if len(parts) < 9:
                        continue
                    parts[1] = r.uri  # normalize recording field
                    out.write(" ".join(parts) + "\n")


def append_uem(uem_out: Path, recs: List[Recording], channel: int) -> None:
    ensure_dir(uem_out.parent)
    with uem_out.open("a", encoding="utf-8") as out:
        for r in recs:
            dur = audio_duration_seconds(r.wav)
            out.write(f"{r.uri} {channel} 0.000 {dur:.3f}\n")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Append new nested (wav,rttm) data directly into the TEST split of a prepared pyannote dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--new-root", type=Path, required=True, help="Root of the additional nested dataset")
    ap.add_argument("--output-dir", type=Path, required=True, help="Existing prepared data root (with audio/, rttm/, uem/, lists/)")
    ap.add_argument("--audio-exts", nargs="+", default=[".wav", ".WAV"], help="Audio extensions to include")
    ap.add_argument("--uri-prefix", type=str, default="", help="Optional prefix to prepend to each derived URI (e.g., 'NEW')")
    ap.add_argument("--uem-channel", type=int, default=1, help="Channel index for UEM")
    ap.add_argument("--copy", action="store_true", help="Copy audio instead of creating symlinks")
    ap.add_argument("--sort-list", action="store_true", help="Sort test.lst after merge (default: keep set-union unsorted)")
    ap.add_argument("--skip-existing", action="store_true", help="If a target audio already exists, skip it instead of replacing")
    ap.add_argument("--dry-run", action="store_true", help="Scan and report; do not write any files")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(levelname)s] %(message)s")

    new_root: Path = args.new_root
    out_root: Path = args.output_dir

    # Basic checks
    if not new_root.exists():
        logging.error("New data root does not exist: %s", new_root)
        return 2
    required = [out_root / "audio", out_root / "lists", out_root / "rttm", out_root / "uem"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logging.error("Output dir does not look like a prepared dataset (missing: %s)", ", ".join(missing))
        return 2

    # 1) Scan new dataset
    recs = scan_new_dataset(new_root, tuple(args.audio_exts))
    if not recs:
        logging.error("No (wav,rttm) pairs found under %s", new_root)
        return 2

    # 2) Apply optional URI prefix and check collisions with existing test
    existing_uris = set(read_existing_test_uris(out_root / "lists" / "test.lst"))
    adjusted: List[Recording] = []
    for r in recs:
        uri = (args.uri_prefix + "__" + r.uri) if args.uri_prefix else r.uri
        if uri in existing_uris:
            logging.warning("URI already in test list: %s (will skip)", uri)
            continue
        adjusted.append(Recording(uri=uri, wav=r.wav, rttm=r.rttm))

    if not adjusted:
        logging.info("Nothing to add (all URIs already present).")
        return 0

    logging.info("Will add %d recording(s) to TEST.", len(adjusted))

    if args.dry_run:
        for r in adjusted[:10]:
            logging.info("example → %s", r.uri)
        logging.info("Dry run: not writing any files.")
        return 0

    # 3) Link/copy audio
    for r in adjusted:
        dst = out_root / "audio" / f"{r.uri}.wav"
        if args.skip_existing and dst.exists():
            logging.info("skip existing audio: %s", dst)
            continue
        link_or_copy(r.wav, dst, copy=args.copy)

    # 4) Append RTTM & UEM
    append_rttm_normalized(out_root / "rttm" / "test.rttm", adjusted)
    append_uem(out_root / "uem" / "test.uem", adjusted, channel=args.uem_channel)

    # 5) Update test.lst
    write_list_dedup(out_root / "lists" / "test.lst", [r.uri for r in adjusted], sort_list=args.sort_list)

    logging.info("Done. Test split augmented with %d new file(s).", len(adjusted))
    logging.info("Updated paths → %s | %s | %s | %s",
                 out_root / "lists" / "test.lst",
                 out_root / "rttm" / "test.rttm",
                 out_root / "uem" / "test.uem",
                 out_root / "audio")
    return 0


if __name__ == "__main__":
    import sys
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
