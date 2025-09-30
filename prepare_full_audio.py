#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a call-level pyannote dataset from chunked folders and create 80/10/10 splits.

INPUT TREE (example)
--------------------
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

ASSUMPTIONS
-----------
- Each chunk .wav has a matching .rttm with the same stem (times are 0-based within that chunk).
- Chunks are named with a suffix `_chunkN` (N = 0,1,2,...).
- Consecutive chunks overlap by `--chunk-overlap` seconds (e.g., 3.0).
- Speaker labels are already consistent *within a call* (speaker_0 means the same person across chunks of the same call).

OUTPUT TREE
-----------
OUT/
  audio/         # one WAV per call (16kHz mono)
  rttm/          # one RTTM per call (times on the concatenated timeline)
  uem/           # one UEM per split, or one per call (optional)
  lists/
    train.lst    # URIs (one per call) for training set
    dev.lst
    test.lst
  database.yml   # ready for pyannote

USAGE
-----
python prepare_call_with_split.py \
  --src-root "/path/to/chunked/dataset" \
  --out-root "/path/to/output/call_level" \
  --audio-ext wav \
  --chunk-overlap 3.0 \
  --seed 123 \
  --make-uem

Then point PYANNOTE_DATABASE_CONFIG to OUT/database.yml.

NOTES
-----
- Splitting is at the *call* level (80/10/10). No time slicing across calls.
- When concatenating, we *remove overlap from the head of chunks 1..N-1* so audio and RTTMs align 1:1.
"""

from __future__ import annotations
import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torchaudio

# ---------- helpers: file discovery & naming ---------- #

CHUNK_RE = re.compile(r"^(?P<base>.*)_chunk(?P<idx>\d+)$", re.IGNORECASE)

def find_chunks(src_root: Path, audio_ext: str) -> Dict[str, List[Path]]:
    """
    Return {call_key: [chunk_wav_paths_sorted_by_index]}.
    The call_key is the filename stem without the trailing `_chunkN`.
    """
    groups: Dict[str, List[Tuple[int, Path]]] = {}
    for wav in src_root.rglob(f"*.{audio_ext}"):
        stem = wav.stem  # e.g., call_001_chunk3
        m = CHUNK_RE.match(stem)
        if not m:
            # not a chunk; skip
            continue
        base = m.group("base")  # e.g., call_001
        idx = int(m.group("idx"))
        groups.setdefault(base, []).append((idx, wav))
    # sort by index
    ordered: Dict[str, List[Path]] = {k: [p for _, p in sorted(v, key=lambda t: t[0])] for k, v in groups.items()}
    return ordered

def sibling(path: Path, new_ext: str) -> Path:
    return path.with_suffix(f".{new_ext.lstrip('.')}")

# ---------- helpers: audio IO ---------- #

def load_mono_16k(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio -> mono, 16k, float32 numpy array in [-1, 1]."""
    wav, sr = torchaudio.load(str(path))  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav.squeeze(0).numpy().astype(np.float32), sr

def save_wav_16k_mono(path: Path, data: np.ndarray):
    """Save float32 mono at 16k."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torchaudio.tensor_to_audio_tensor(data) if hasattr(torchaudio, "tensor_to_audio_tensor") else None
    if tensor is None:
        import torch
        tensor = torch.from_numpy(data).unsqueeze(0)
    torchaudio.save(str(path), tensor, 16000)

# ---------- helpers: RTTM parsing/writing ---------- #

def parse_rttm(path: Path) -> List[Tuple[float, float, str]]:
    """
    Return list of (start, duration, speaker) for SPEAKER lines.
    Ignores other RTTM object types.
    """
    out: List[Tuple[float, float, str]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER "):
                continue
            # SPEAKER <uri> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                start = float(parts[3])
                dur   = float(parts[4])
                spk   = parts[7]
                out.append((start, dur, spk))
            except Exception:
                continue
    return out

def write_rttm(call_uri: str, events: List[Tuple[float, float, str]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for start, dur, spk in events:
            f.write(f"SPEAKER {call_uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n")

# ---------- core: build one call ---------- #

def build_call(
    call_key: str,
    chunk_wavs: List[Path],
    chunk_overlap: float,
    audio_ext_for_rttm: str,
    out_audio_dir: Path,
    out_rttm_dir: Path,
) -> Tuple[str, float]:
    """
    Concatenate chunks and merge RTTM to a single call.

    Returns
    -------
    (call_uri, total_duration_sec)
    """
    # call_uri can be the call_key itself (it may already contain room tokens etc.)
    call_uri = call_key

    concat_audio: List[np.ndarray] = []
    concat_events: List[Tuple[float, float, str]] = []
    cur_sec = 0.0
    total_dur = 0.0

    for i, wav_path in enumerate(chunk_wavs):
        # 1) audio
        arr, sr = load_mono_16k(wav_path)             # arr shape: (T,)
        orig_dur = len(arr) / sr
        head = chunk_overlap if i > 0 else 0.0
        if head >= orig_dur:
            # fully trimmed — skip (rare)
            kept = np.zeros((0,), dtype=np.float32)
            kept_dur = 0.0
        else:
            head_samp = int(round(head * sr))
            kept = arr[head_samp:]
            kept_dur = len(kept) / sr

        concat_audio.append(kept)

        # 2) rttm
        rttm_path = sibling(wav_path, "rttm")
        events = parse_rttm(rttm_path)
        # crop to [head, orig_dur] then shift to call timeline
        for s, d, spk in events:
            a, b = s, s + d
            lo = max(a, head)
            hi = min(b, orig_dur)
            if hi > lo:
                start_out = cur_sec + (lo - head)  # position inside kept + call offset
                dur_out = hi - lo
                concat_events.append((start_out, dur_out, spk))

        cur_sec += kept_dur
        total_dur += kept_dur

    # write audio
    call_wav_path = out_audio_dir / f"{call_uri}.wav"
    if concat_audio:
        full = np.concatenate(concat_audio) if len(concat_audio) > 1 else concat_audio[0]
    else:
        full = np.zeros((0,), dtype=np.float32)
    save_wav_16k_mono(call_wav_path, full)

    # write rttm
    call_rttm_path = out_rttm_dir / f"{call_uri}.rttm"
    write_rttm(call_uri, concat_events, call_rttm_path)

    return call_uri, total_dur

# ---------- splitting & merged RTTMs ---------- #

def split_uris(uris: List[str], ratios=(0.8, 0.1, 0.1), seed=123) -> Tuple[List[str], List[str], List[str]]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    rnd = random.Random(seed)
    pool = uris[:]
    rnd.shuffle(pool)
    n = len(pool)
    n_train = int(round(ratios[0] * n))
    n_dev = int(round(ratios[1] * n))
    # ensure we don't exceed
    if n_train + n_dev > n:
        n_dev = max(0, n - n_train)
    n_test = n - n_train - n_dev

    train = pool[:n_train]
    dev   = pool[n_train:n_train + n_dev]
    test  = pool[n_train + n_dev:]
    return train, dev, test

def write_list(path: Path, uris: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for u in uris:
            f.write(u + "\n")

def cat_rttms_for_split(out_rttm_dir: Path, uris: List[str], split_name: str):
    merged = out_rttm_dir / f"{split_name}.rttm"
    merged.parent.mkdir(parents=True, exist_ok=True)
    with merged.open("w", encoding="utf-8") as g:
        for u in uris:
            p = out_rttm_dir / f"{u}.rttm"
            if p.exists():
                g.write(p.read_text(encoding="utf-8"))
    return merged

def write_uem_full_coverage(out_uem_dir: Path, split_name: str, uri_to_dur: Dict[str, float], uris: List[str]):
    out = out_uem_dir / f"{split_name}.uem"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for u in uris:
            dur = uri_to_dur.get(u, 0.0)
            if dur > 0:
                f.write(f"{u} 1 0.000 {dur:.3f}\n")
    return out

# ---------- database.yml ---------- #

def write_database_yaml(out_root: Path):
    yml = f"""# database.yml (auto-generated)

Databases:
  MyDatabase:
    - {out_root.as_posix()}

Protocols:
  MyDatabase:
    SpeakerDiarization:
      MyProtocol:
        scope: file
        train:
          uri:        lists/train.lst
          annotation: rttm/train.rttm
          annotated:  uem/train.uem
        development:
          uri:        lists/dev.lst
          annotation: rttm/dev.rttm
          annotated:  uem/dev.uem
        test:
          uri:        lists/test.lst
          annotation: rttm/test.rttm
          annotated:  uem/test.uem

Preprocessors:
  audio:
    name: pyannote.database.FileFinder
    params:
      paths:
        MyDatabase: {out_root.as_posix()}/audio/{{uri}}.wav

  torchaudio.info:
    name: pyannote.database.preprocessors.TorchaudioInfo
    params: {{}}
"""
    (out_root / "database.yml").write_text(yml, encoding="utf-8")

# ---------- CLI ---------- #

def parse_args():
    ap = argparse.ArgumentParser("Build call-level dataset from chunked folders + make 80/10/10 splits")
    ap.add_argument("--src-root", required=True, help="Root of chunked dataset (rooms/calls/chunks)")
    ap.add_argument("--out-root", required=True, help="Output root for call-level dataset")
    ap.add_argument("--audio-ext", default="wav", help="Chunk audio extension (default: wav)")
    ap.add_argument("--chunk-overlap", type=float, default=0.0, help="Seconds overlapped between consecutive chunks")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for 80/10/10 split")
    ap.add_argument("--ratio-train", type=float, default=0.8)
    ap.add_argument("--ratio-dev", type=float, default=0.1)
    ap.add_argument("--ratio-test", type=float, default=0.1)
    ap.add_argument("--make-uem", action="store_true", help="Write full-coverage UEMs per split")
    return ap.parse_args()

def main():
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()

    # output dirs
    out_audio = out_root / "audio"
    out_rttm  = out_root / "rttm"
    out_uem   = out_root / "uem"
    out_lists = out_root / "lists"
    for d in (out_audio, out_rttm, out_lists):
        d.mkdir(parents=True, exist_ok=True)
    if args.make_uem:
        out_uem.mkdir(parents=True, exist_ok=True)

    # 1) discover chunks grouped by call
    groups = find_chunks(src_root, args.audio_ext)
    if not groups:
        print(f"[ERROR] No chunks found under {src_root} matching '*_chunkN.{args.audio_ext}'.", file=sys.stderr)
        return 2

    print(f"Found {len(groups)} calls. Building call-level audio+RTTM (trim overlap={args.chunk_overlap}s)…")

    # 2) build each call
    uri_to_dur: Dict[str, float] = {}
    for call_key, chunk_wavs in sorted(groups.items()):
        call_uri, dur = build_call(
            call_key=call_key,
            chunk_wavs=chunk_wavs,
            chunk_overlap=args.chunk_overlap,
            audio_ext_for_rttm=args.audio_ext,
            out_audio_dir=out_audio,
            out_rttm_dir=out_rttm,
        )
        uri_to_dur[call_uri] = dur

    uris = sorted(uri_to_dur.keys())
    print(f"Done building {len(uris)} calls.")

    # 3) split 80/10/10 (by calls, not by time)
    ratios = (args.ratio_train, args.ratio_dev, args.ratio_test)
    if abs(sum(ratios) - 1.0) > 1e-6:
        print("[WARN] ratios do not sum to 1.0 exactly; normalizing.")
        s = sum(ratios)
        ratios = tuple(r / s for r in ratios)

    train, dev, test = split_uris(uris, ratios=ratios, seed=args.seed)

    write_list(out_lists / "train.lst", train)
    write_list(out_lists / "dev.lst", dev)
    write_list(out_lists / "test.lst", test)
    print(f"Split counts: train={len(train)} dev={len(dev)} test={len(test)}")

    # 4) merged split RTTMs
    cat_rttms_for_split(out_rttm, train, "train")
    cat_rttms_for_split(out_rttm, dev,   "dev")
    cat_rttms_for_split(out_rttm, test,  "test")

    # 5) UEMs (full coverage of each call)
    if args.make_uem:
        write_uem_full_coverage(out_uem, "train", uri_to_dur, train)
        write_uem_full_coverage(out_uem, "dev",   uri_to_dur, dev)
        write_uem_full_coverage(out_uem, "test",  uri_to_dur, test)

    # 6) database.yml
    write_database_yaml(out_root)

    print(f"\nAll set.\nCall-level dataset written to: {out_root}\n"
          f"- audio/: {len(uris)} WAVs\n- rttm/: {len(list((out_rttm).glob('*.rttm')))} files\n"
          f"- lists/: train/dev/test\n- database.yml ready for PYANNOTE_DATABASE_CONFIG")

if __name__ == "__main__":
    raise SystemExit(main())
