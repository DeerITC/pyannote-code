#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a pyannote-compatible diarization dataset from a nested folder tree.

This script **scans a deeply nested dataset** to locate `(audio, rttm[, txt])`
triplets and generates:

- `lists/train.lst`, `lists/dev.lst`, `lists/test.lst` (URIs only)
- `rttm/train.rttm`, `rttm/dev.rttm`, `rttm/test.rttm` (merged RTTMs per split)
- `uem/train.uem`,  `uem/dev.uem`,  `uem/test.uem` (evaluation masks)
- `database.yml` (ready for `pyannote.database`)
- `README_PREP.md` (what was created & how to use it)

It is **session-aware by default**: files are grouped by the **last N folders** of
their relative path (configurable via `--group-depth`) so that all recordings
from the same session/folder land in the same split (avoiding leakage across
train/dev/test).

The output layout is **flat & simple** so you do not need custom preprocessors
or TSV maps for pyannote:

```
<dst>/
  audio/{uri}.wav
  rttm/{train,dev,test}.rttm
  uem/{train,dev,test}.uem
  lists/{train,dev,test}.lst
  database.yml
  README_PREP.md
```

This script writes **symlinks** into `<dst>/audio` by default (saves disk);
use `--copy` to physically copy files instead.

Example
-------
Run a default, leakage-safe split (80/10/10), grouping by the last 2 folders:

```bash
python prepare_dataset.py \
  --dataset-root ./dataset \
  --output-dir   ./data \
  --group-depth  2 \
  --ratios 0.8 0.1 0.1 \
  --audio-exts .wav
```

Then smoke-test the protocol:

```python
from pyannote.database import registry, get_protocol
registry.load_database("/data/exp_prep/database.yml")
proto = get_protocol("MyDatabase.SpeakerDiarization.MyProtocol")
print(len(list(proto.train())), len(list(proto.development())), len(list(proto.test())))
```

Requirements
------------
- Python 3.9+
- `torchaudio`.

"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torchaudio  # type: ignore
    _HAVE_TORCHAUDIO = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_TORCHAUDIO = False

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recording:
    """Represents a single discovered recording.

    Attributes:
      uri: Unique identifier (derived from relative path + stem).
      wav: Absolute path to the WAV audio file.
      rttm: Absolute path to the RTTM file.
      txt: Optional absolute path to a sidecar TXT file with the same stem.
      group_key: Key used for session-aware splitting (last N folders by default).
    """

    uri: str
    wav: Path
    rttm: Path
    txt: Optional[Path]
    group_key: str


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sanitize_component(component: str) -> str:
    """Sanitize a single path component for URI composition.

    Replaces spaces with underscores and removes problematic characters.

    Args:
      component: Raw folder or filename component.

    Returns:
      A safe component string suitable for URIs.
    """
    out = component.replace(" ", "_")
    # Keep common URL/filename-safe characters; strip others.
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_+.@"
    out = "".join(ch for ch in out if ch in allowed)
    return out or "_"


def derive_uri(rel_parts: Sequence[str]) -> str:
    """Create a unique, readable URI from relative path parts.

    Example: `2025_03_10_20_01_18_102_77/2025_03_10_20_01_18_102_77_chunk0.wav` →
    `102__2025_03_10_21_21_46_102_77__2025_03_10_21_21_46_102_77_chunk0`.

    Args:
      rel_parts: Relative path split into components, e.g., `("A", "B", "f.wav")`.

    Returns:
      URI string.
    """
    if not rel_parts:
        raise ValueError("empty rel_parts; cannot derive URI")
    *folders, filename = list(rel_parts)
    stem = Path(filename).stem
    safe = [sanitize_component(p) for p in folders]
    safe_stem = sanitize_component(stem)
    return "__".join([*safe, safe_stem]) if folders else safe_stem


def derive_group_key(rel_parts: Sequence[str], group_depth: int) -> str:
    """Derive a session/group key from the last `group_depth` folders.

    Args:
      rel_parts: Relative path split into components (including filename).
      group_depth: Number of trailing folders to include in the key (>=1).

    Returns:
      A group key string used for stable, leakage-safe splits.
    """
    dirs_only = list(rel_parts[:-1])  # drop filename
    if not dirs_only:
        return "root"
    depth = max(1, min(group_depth, len(dirs_only)))
    last = dirs_only[-depth:]
    return "/".join(last)


def sha1_unit(s: str) -> float:
    """Map a string deterministically to the unit interval [0, 1)."""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def pick_split(group_key: str, ratios: Tuple[float, float, float]) -> str:
    """Assign a split label based on a group key.

    Args:
      group_key: The session/group identifier.
      ratios: (train, dev, test) ratios that sum to 1.0.

    Returns:
      One of `"train"`, `"dev"`, `"test"`.
    """
    t, d, _ = ratios
    x = sha1_unit(group_key)
    if x < t:
        return "train"
    if x < t + d:
        return "dev"
    return "test"


def largest_remainder_counts(total: int, ratios: Tuple[float, float, float]) -> List[int]:
    """Compute integer counts using the largest remainder method.

    Args:
      total: Total number of items.
      ratios: (train, dev, test) ratios that sum to 1.0.

    Returns:
      List of three integers adding to `total`.
    """
    targets = [total * r for r in ratios]
    floors = [int(math.floor(x)) for x in targets]
    remainder = total - sum(floors)
    fracs = sorted([(targets[i] - floors[i], i) for i in range(3)], reverse=True)
    for k in range(remainder):
        floors[fracs[k][1]] += 1
    return floors


def ensure_nonempty_dev_test(counts: List[int], total: int) -> List[int]:
    """Guarantee dev and test are non-empty when feasible.

    Args:
      counts: [n_train, n_dev, n_test].
      total: Total number of items.

    Returns:
      Adjusted counts that still sum to `total`.
    """
    n_train, n_dev, n_test = counts
    if total >= 2 and n_dev == 0:
        if n_train > 1:
            n_train -= 1; n_dev += 1
        elif n_test > 1:
            n_test -= 1; n_dev += 1
    if total >= 2 and n_test == 0:
        if n_train > 1:
            n_train -= 1; n_test += 1
        elif n_dev > 1:
            n_dev -= 1; n_test += 1
    # Rebalance if we drifted
    s = n_train + n_dev + n_test
    if s != total:
        n_train += (total - s)
    return [n_train, n_dev, n_test]


# ---------------------------------------------------------------------------
# Scanning & validation
# ---------------------------------------------------------------------------

def validate_rttm_minimal(rttm_path: Path) -> Optional[str]:
    """Perform a minimal sanity-check on an RTTM file.

    Ensures presence of at least one SPEAKER line and parsable start/duration.

    Args:
      rttm_path: Path to an RTTM file.

    Returns:
      `None` if OK, otherwise a short error string.
    """
    try:
        text = rttm_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - filesystem
        return f"cannot read RTTM: {exc}"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    spk = [ln for ln in lines if ln.startswith("SPEAKER ")]
    if not spk:
        return "no SPEAKER lines found"
    for i, ln in enumerate(spk, start=1):
        parts = ln.split()
        if len(parts) < 9:
            return f"line {i}: expected >= 9 fields"
        try:
            float(parts[3]); float(parts[4])
        except Exception:
            return f"line {i}: non-numeric start/duration"
    return None


def scan_dataset(
    dataset_root: Path,
    audio_exts: Sequence[str] = (".wav", ".WAV"),
    group_depth: int = 2,
) -> List[Recording]:
    """Recursively scan the dataset for `(wav, rttm[, txt])` triplets.

    Args:
      dataset_root: Root directory of the nested dataset.
      audio_exts: Audio extensions to include.
      group_depth: Number of trailing folders that define the session group key.

    Returns:
      Sorted list of `Recording` objects.
    """
    root = dataset_root.resolve()
    results: List[Recording] = []
    for wav in root.rglob("*"):
        if not wav.is_file() or wav.suffix not in audio_exts:
            continue
        rttm = wav.with_suffix(".rttm")
        if not rttm.exists():
            logging.debug("skip (no RTTM): %s", wav)
            continue
        txt = wav.with_suffix(".txt")
        rel = wav.relative_to(root)
        rel_parts = rel.as_posix().split("/")
        uri = derive_uri(rel_parts)
        group_key = derive_group_key(rel_parts, group_depth=group_depth)
        err = validate_rttm_minimal(rttm)
        if err is not None:
            logging.warning("RTTM validation issue for %s: %s", rttm, err)
        results.append(Recording(uri=uri, wav=wav.resolve(), rttm=rttm.resolve(),
                                 txt=(txt.resolve() if txt.exists() else None),
                                 group_key=group_key))
    results.sort(key=lambda r: (str(r.wav), str(r.rttm)))
    # Guard against URI collisions
    seen: Dict[str, Recording] = {}
    for r in results:
        if r.uri in seen:
            raise RuntimeError(f"Duplicate URI generated: {r.uri}\n  A: {seen[r.uri].wav}\n  B: {r.wav}")
        seen[r.uri] = r
    return results


# ---------------------------------------------------------------------------
# Splitting, linking/copying, and writers
# ---------------------------------------------------------------------------

def split_session_aware(records: List[Recording], ratios: Tuple[float, float, float]) -> Tuple[List[str], List[str], List[str]]:
    """Split by session/group (leakage-safe).

    Args:
      records: List of `Recording`.
      ratios: (train, dev, test) ratios.

    Returns:
      (`train_uris`, `dev_uris`, `test_uris`).
    """
    # Map group -> URIs
    groups: Dict[str, List[str]] = {}
    for r in records:
        groups.setdefault(r.group_key, []).append(r.uri)
    group_names = list(groups.keys())
    if len(group_names) < 2:
        raise ValueError("Not enough groups to perform a session-aware split.")

    # Deterministic ordering, then allocate counts
    group_names.sort()
    n_groups = len(group_names)
    counts = largest_remainder_counts(n_groups, ratios)
    counts = ensure_nonempty_dev_test(counts, n_groups)
    n_train, n_dev, n_test = counts

    # Assign by hash (stable), not random shuffle — ensures reproducibility
    def score(g: str) -> float:
        return sha1_unit(g)

    group_names.sort(key=score)
    g_train = set(group_names[:n_train])
    g_dev = set(group_names[n_train:n_train + n_dev])
    g_test = set(group_names[n_train + n_dev:])

    train = sorted([u for g in g_train for u in groups[g]])
    dev = sorted([u for g in g_dev for u in groups[g]])
    test = sorted([u for g in g_test for u in groups[g]])

    if len(group_names) >= 2 and (len(dev) == 0 or len(test) == 0):
        raise ValueError("Degenerate split: dev/test ended up empty.")

    return train, dev, test


def split_per_file(records: List[Recording], ratios: Tuple[float, float, float]) -> Tuple[List[str], List[str], List[str]]:
    """Split by file (ignores session grouping)."""
    uris = sorted(r.uri for r in records)
    n = len(uris)
    if n == 0:
        return [], [], []
    counts = largest_remainder_counts(n, ratios)
    counts = ensure_nonempty_dev_test(counts, n)
    n_train, n_dev, n_test = counts
    # Deterministic order by hash instead of random
    uris.sort(key=sha1_unit)
    train = sorted(uris[:n_train])
    dev = sorted(uris[n_train:n_train + n_dev])
    test = sorted(uris[n_train + n_dev:n_train + n_dev + n_test])
    return train, dev, test


def ensure_dir(p: Path) -> None:
    """Create a directory (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    """Create a symlink to `src` at `dst` or copy if symlinks fail.

    Args:
      src: Source file path.
      dst: Destination path.
      copy: If True, always copy; otherwise prefer symlink and fall back to copy.
    """
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except Exception:  # pragma: no cover - filesystem edge
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


def write_list(path: Path, uris: Sequence[str]) -> None:
    """Write a `.lst` file with one URI per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(uris) + "\n", encoding="utf-8")


def audio_duration_seconds(path: Path) -> float:
    """Return the duration of a WAV file in seconds.

    Uses `torchaudio` when available, otherwise falls back to the built-in
    `wave` module (PCM WAV only).
    """
    if _HAVE_TORCHAUDIO:
        try:
            info = torchaudio.info(str(path))
            return float(info.num_frames) / float(info.sample_rate)
        except Exception:  # pragma: no cover - codec/format edge
            pass
    # Fallback: wave (PCM only)
    import wave
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate)


def write_uem(path: Path, uris: Sequence[str], items: Dict[str, Recording], channel: int = 1) -> None:
    """Write a UEM file with one full-coverage segment per URI.

    Args:
      path: Destination UEM path.
      uris: URIs to include.
      items: Mapping from URI to `Recording`.
      channel: UEM channel index (pyannote typically uses `1`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for uri in uris:
            wav = items[uri].wav
            if not wav.exists():  # pragma: no cover - safety
                continue
            dur = audio_duration_seconds(wav)
            out.write(f"{uri} {channel} 0.000 {dur:.3f}\n")


def fix_and_merge_rttm(path: Path, uris: Sequence[str], items: Dict[str, Recording]) -> None:
    """Normalize and merge per-file RTTMs into a single RTTM for the split.

    Ensures each SPEAKER line's recording field (col #1) equals the split URI.

    Args:
      path: Destination merged RTTM path.
      uris: URIs to include.
      items: Mapping from URI to `Recording`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for uri in uris:
            rttm_path = items[uri].rttm
            if not rttm_path.exists():  # pragma: no cover - safety
                continue
            with rttm_path.open("r", encoding="utf-8") as f:
                for ln in f:
                    if not ln.startswith("SPEAKER "):
                        continue
                    parts = ln.strip().split()
                    if len(parts) < 9:
                        continue  # skip malformed lines
                    parts[1] = uri  # normalize the recording ID
                    # Optional: normalize channel to "1"
                    # parts[2] = "1"
                    out.write(" ".join(parts) + "\n")


def write_database_yml(dst_root: Path) -> None:
    """Write a minimal `database.yml` compatible with pyannote.

    Points directly to flattened audio and split-level RTTMs/UEMs.

    Args:
      dst_root: Root output directory containing `audio/`, `lists/`, `rttm/`, `uem/`.
    """
    yml = f"""Protocols:
  MyDatabase:
    SpeakerDiarization:
      MyProtocol:
        scope: file
        train:
          uri: lists/train.lst
          audio: audio/{{uri}}.wav
          annotation: rttm/train.rttm
          annotated: uem/train.uem
        development:
          uri: lists/dev.lst
          audio: audio/{{uri}}.wav
          annotation: rttm/dev.rttm
          annotated: uem/dev.uem
        test:
          uri: lists/test.lst
          audio: audio/{{uri}}.wav
          annotation: rttm/test.rttm
          annotated: uem/test.uem
"""
    (dst_root / "database.yml").write_text(yml, encoding="utf-8")


def write_readme(dst_root: Path, n_train: int, n_dev: int, n_test: int, group_depth: int) -> None:
    """Emit a short README documenting what was created."""
    md = f"""# Dataset preparation artifacts\n\nThis folder was generated by `prepare_dataset.py`.\n\n## Files\n- `lists/train.lst` — one URI per line (training set)\n- `lists/dev.lst`   — one URI per line (development set)\n- `lists/test.lst`  — one URI per line (test set)\n- `rttm/*.rttm`     — merged RTTMs per split\n- `uem/*.uem`       — full-coverage UEMs per split\n- `database.yml`    — ready-to-use protocol for pyannote\n\n## Splitting policy\nSession-aware splitting using the last **{group_depth}** folder(s) of the relative path as the group key. All files\nsharing the same key are assigned to the same split (to avoid leakage).\n\n## Counts\n- train: {n_train}\n- dev:   {n_dev}\n- test:  {n_test}\n\n## Next steps\nLoad `database.yml` with `pyannote.database` and proceed to tuning/training.\n"""
    (dst_root / "README_PREP.md").write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface.

    Returns:
      An `ArgumentParser` instance.
    """
    ap = argparse.ArgumentParser(
        description="Flatten a nested diarization dataset into a pyannote-ready layout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="Root directory containing nested audio + RTTM files")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write lists/, rttm/, uem/, audio/, database.yml")
    ap.add_argument("--audio-exts", nargs="+", default=[".wav", ".WAV"],
                    help="Audio file extensions to include")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN", "DEV", "TEST"),
                    help="Split ratios (must sum to 1.0)")
    ap.add_argument("--group-depth", type=int, default=2,
                    help="How many trailing folders compose the session group key")
    ap.add_argument("--split-mode", choices=["auto", "file", "group"], default="auto",
                    help="'group' enforces session-aware split; 'file' ignores grouping; 'auto' tries group then falls back")
    ap.add_argument("--uem-channel", type=int, default=1, help="Channel index for UEM lines")
    ap.add_argument("--copy", action="store_true", help="Copy audio instead of creating symlinks")
    ap.add_argument("--dry-run", action="store_true", help="Scan and report; do not write files")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Returns:
      Process exit code (0 on success).
    """
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(levelname)s] %(message)s")

    # Validate ratios
    t, d, te = args.ratios
    s = t + d + te
    if abs(s - 1.0) > 1e-6:
        logging.error("--ratios must sum to 1.0 (got %.3f)", s)
        return 2

    dataset_root: Path = args.dataset_root
    output_dir: Path = args.output_dir

    if not dataset_root.exists():
        logging.error("Dataset root does not exist: %s", dataset_root)
        return 2

    # 1) Scan
    recs = scan_dataset(dataset_root, audio_exts=tuple(args.audio_exts), group_depth=args.group_depth)
    if not recs:
        logging.error("No (wav,rttm) pairs found under %s", dataset_root)
        return 2

    logging.info("Found %d recording(s).", len(recs))

    # 2) Split
    try:
        if args.split_mode == "group":
            train_uris, dev_uris, test_uris = split_session_aware(recs, (t, d, te))
        elif args.split_mode == "file":
            train_uris, dev_uris, test_uris = split_per_file(recs, (t, d, te))
        else:  # auto
            try:
                train_uris, dev_uris, test_uris = split_session_aware(recs, (t, d, te))
            except Exception:
                logging.warning("Falling back to per-file split (insufficient/imbalanced groups).")
                train_uris, dev_uris, test_uris = split_per_file(recs, (t, d, te))
    except Exception as exc:
        logging.error("Failed to split dataset: %s", exc)
        return 2

    logging.info("Split sizes → train: %d | dev: %d | test: %d",
                 len(train_uris), len(dev_uris), len(test_uris))

    if args.dry_run:
        logging.info("Dry run: not writing any files.")
        return 0

    # Index for quick lookup
    items: Dict[str, Recording] = {r.uri: r for r in recs}

    # 3) Create target layout
    ensure_dir(output_dir / "audio")
    ensure_dir(output_dir / "lists")
    ensure_dir(output_dir / "rttm")
    ensure_dir(output_dir / "uem")

    # 4) Link/copy audio
    logging.info("Linking/copying audio → %s", output_dir / "audio")
    for uri in train_uris + dev_uris + test_uris:
        link_or_copy(items[uri].wav, output_dir / "audio" / f"{uri}.wav", copy=args.copy)

    # 5) Write lists + UEM + merged RTTMs
    logging.info("Writing list files & metadata")
    write_list(output_dir / "lists" / "train.lst", train_uris)
    write_list(output_dir / "lists" / "dev.lst",   dev_uris)
    write_list(output_dir / "lists" / "test.lst",  test_uris)

    fix_and_merge_rttm(output_dir / "rttm" / "train.rttm", train_uris, items)
    fix_and_merge_rttm(output_dir / "rttm" / "dev.rttm",   dev_uris,   items)
    fix_and_merge_rttm(output_dir / "rttm" / "test.rttm",  test_uris,  items)

    write_uem(output_dir / "uem" / "train.uem", train_uris, items, channel=args.uem_channel)
    write_uem(output_dir / "uem" / "dev.uem",   dev_uris,   items, channel=args.uem_channel)
    write_uem(output_dir / "uem" / "test.uem",  test_uris,  items, channel=args.uem_channel)

    # 6) Write database.yml + README
    write_database_yml(output_dir)
    write_readme(output_dir, len(train_uris), len(dev_uris), len(test_uris), args.group_depth)

    logging.info("Done. New data root: %s", output_dir.resolve())
    logging.info("Examples: %s | %s | %s | %s",
                 output_dir / "lists" / "train.lst",
                 output_dir / "rttm" / "train.rttm",
                 output_dir / "uem" / "train.uem",
                 output_dir / "database.yml")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    import sys
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
