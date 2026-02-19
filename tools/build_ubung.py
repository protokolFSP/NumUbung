#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from tqdm import tqdm

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/protokolFSP/FSPtranskript/main/transcripts"

# --- SRT parsing ---

@dataclass
class Cue:
    start: float
    end: float
    text: str

def srt_time_to_sec(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", ".")
    m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})(\.\d+)?$", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3))
    frac = float(m.group(4) or 0.0)
    return hh * 3600 + mm * 60 + ss + frac

def parse_srt(text: str) -> List[Cue]:
    t = (text or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", t)
    cues: List[Cue] = []
    for b in blocks:
        lines = [ln.rstrip() for ln in b.split("\n") if ln.strip()]
        if not lines:
            continue
        idx = 0
        if re.fullmatch(r"\d+", lines[0].strip()):
            idx = 1
        if idx >= len(lines):
            continue
        tm = re.match(r"(.+?)\s*-->\s*(.+)", lines[idx])
        if not tm:
            continue
        a = srt_time_to_sec(tm.group(1))
        b2 = srt_time_to_sec(tm.group(2).split()[0])
        if a is None or b2 is None or b2 <= a:
            continue
        body = "\n".join(lines[idx + 1:]).strip()
        body = re.sub(r"<[^>]+>", "", body).strip()
        if not body:
            continue
        cues.append(Cue(a, b2, body))
    cues.sort(key=lambda c: c.start)
    return cues

def sec_to_srt_time(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    ms = int(round((sec - int(sec)) * 1000))
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def slice_srt(cues: List[Cue], start: float, end: float) -> str:
    out = []
    n = 1
    for c in cues:
        if c.end <= start:
            continue
        if c.start >= end:
            break
        ns = max(0.0, c.start - start)
        ne = min(end - start, c.end - start)
        if ne <= ns:
            continue
        out.append(str(n))
        out.append(f"{sec_to_srt_time(ns)} --> {sec_to_srt_time(ne)}")
        out.append(c.text)
        out.append("")
        n += 1
    return "\n".join(out).strip() + "\n"

def slice_txt(cues: List[Cue], start: float, end: float) -> str:
    parts = []
    for c in cues:
        if c.end <= start:
            continue
        if c.start >= end:
            break
        parts.append(c.text.strip())
    return "\n\n".join([p for p in parts if p]) + "\n"

# --- IA + downloads ---

def ia_metadata(identifier: str) -> dict:
    url = f"https://archive.org/metadata/{requests.utils.quote(identifier)}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def ia_file_download_url(identifier: str, name: str) -> str:
    # IMPORTANT: IA download path is /download/<id>/<name>, name must be url-encoded per path segment
    parts = name.split("/")
    parts_enc = [requests.utils.quote(p) for p in parts]
    return f"https://archive.org/download/{identifier}/" + "/".join(parts_enc)

def safe_stem(filename: str) -> str:
    base = filename.split("/")[-1]
    base = re.sub(r"\.(m4a|mp3)$", "", base, flags=re.I)
    return base

def pick_audio_files(meta: dict, max_files: int) -> List[dict]:
    files = meta.get("files") or []
    out = []
    for f in files:
        name = str(f.get("name") or "")
        low = name.lower()
        if not (low.endswith(".m4a") or low.endswith(".mp3")):
            continue
        # prefer "original" if present
        if "source" in f and str(f.get("source") or "").lower() != "original":
            continue
        out.append(f)
    # stable order (by name)
    out.sort(key=lambda x: str(x.get("name") or "").lower())
    if max_files and max_files > 0:
        out = out[:max_files]
    return out

def fetch_text(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=60)
        if not r.ok:
            return None
        return r.text
    except Exception:
        return None

def build_srt_urls(stem: str, srt_subdir: str) -> List[str]:
    # transcripts/<subdir>/<stem>.srt  (subdir can be empty => transcripts/<stem>.srt)
    sub = (srt_subdir or "").strip().strip("/")
    rel = f"{sub}/{stem}.srt" if sub else f"{stem}.srt"
    rel2 = f"{sub}/A {stem}.srt" if sub else f"A {stem}.srt"
    rel3 = f"{sub}/A_{stem}.srt" if sub else f"A_{stem}.srt"
    rel4 = f"{sub}/A-{stem}.srt" if sub else f"A-{stem}.srt"
    return [
        f"{GITHUB_RAW_BASE}/{requests.utils.quote(rel)}",
        f"{GITHUB_RAW_BASE}/{requests.utils.quote(rel2)}",
        f"{GITHUB_RAW_BASE}/{requests.utils.quote(rel3)}",
        f"{GITHUB_RAW_BASE}/{requests.utils.quote(rel4)}",
    ]

def load_matching_srt(stem: str, srt_subdir: str, debug: bool = False) -> Tuple[Optional[str], List[str]]:
    tried = build_srt_urls(stem, srt_subdir)
    for u in tried:
        txt = fetch_text(u)
        if txt and "-->" in txt:
            return txt, tried
    return None, tried

# --- clip logic: "keskin 4'lü" => group 4 consecutive cues into one clip ---
def build_4cue_segments(cues: List[Cue], first_minutes: float, max_clip_seconds: float) -> List[Tuple[float, float]]:
    lim = max(0.0, first_minutes * 60.0)
    # take cues that start within lim
    scoped = [c for c in cues if c.start < lim]
    segs: List[Tuple[float, float]] = []
    i = 0
    while i + 3 < len(scoped):
        c0, c1, c2, c3 = scoped[i], scoped[i+1], scoped[i+2], scoped[i+3]
        start = c0.start
        end = c3.end
        if end - start > max_clip_seconds:
            end = start + max_clip_seconds
        if end - start >= 2.0:
            segs.append((start, end))
        i += 4
    return segs

def run_ffmpeg_extract(in_url: str, out_path: Path, start: float, end: float) -> None:
    # Re-encode to mp3 for universal playback
    dur = max(0.05, end - start)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", in_url,
        "-t", f"{dur:.3f}",
        "-vn",
        "-ac", "2",
        "-ar", "44100",
        "-b:a", "128k",
        "-y",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ia-id", required=True)
    ap.add_argument("--srt-subdir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-minutes", type=float, default=6.0)
    ap.add_argument("--max-clip-seconds", type=float, default=90.0)
    ap.add_argument("--max-files", type=int, default=9999)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)

    meta = ia_metadata(args.ia_id)
    audios = pick_audio_files(meta, args.max_files)

    index = {
        "ia_identifier": args.ia_id,
        "srt_subdir": args.srt_subdir,
        "built_at_utc": datetime.utcnow().isoformat() + "Z",
        "first_minutes": args.first_minutes,
        "max_clip_seconds": args.max_clip_seconds,
        "files": [],
        "clips": [],
        "warnings": [],
    }

    if not audios:
        index["warnings"].append("No audio files found in IA metadata (m4a/mp3).")
        (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), "utf-8")
        print("No audio files found.")
        return

    # per source file
    clip_global = 0
    for f in tqdm(audios, desc=f"{args.ia_id} files"):
        name = str(f.get("name") or "")
        title = str(f.get("title") or "").strip() or name.split("/")[-1]
        stem = safe_stem(name)

        audio_url = ia_file_download_url(args.ia_id, name)

        srt_text, tried = load_matching_srt(stem, args.srt_subdir, debug=args.debug)
        if not srt_text:
            index["warnings"].append(f"SRT not found for: {name} (stem={stem})")
            if args.debug:
                index["warnings"].append("Tried: " + " | ".join(tried))
            # still include file entry
            index["files"].append({"name": name, "title": title, "audio_url": audio_url, "srt_found": False})
            continue

        cues = parse_srt(srt_text)
        if not cues:
            index["warnings"].append(f"SRT parse failed/empty: {stem}.srt")
            index["files"].append({"name": name, "title": title, "audio_url": audio_url, "srt_found": True, "cues": 0})
            continue

        index["files"].append({"name": name, "title": title, "audio_url": audio_url, "srt_found": True, "cues": len(cues)})

        segs = build_4cue_segments(cues, args.first_minutes, args.max_clip_seconds)
        if not segs:
            index["warnings"].append(f"No segments produced (maybe too short/first_minutes too low): {name}")
            continue

        for si, (st, en) in enumerate(segs, start=1):
            clip_global += 1
            clip_id = f"{stem}__{si:04d}"
            mp3_name = f"{clip_id}.mp3"
            srt_name = f"{clip_id}.srt"
            txt_name = f"{clip_id}.txt"

            mp3_path = out_dir / "clips" / mp3_name
            srt_path = out_dir / "clips" / srt_name
            txt_path = out_dir / "clips" / txt_name

            # audio
            try:
                run_ffmpeg_extract(audio_url, mp3_path, st, en)
            except subprocess.CalledProcessError as e:
                index["warnings"].append(f"ffmpeg failed for {clip_id}: {e}")
                continue

            # srt + txt slices
            srt_slice = slice_srt(cues, st, en)
            txt_slice = slice_txt(cues, st, en)
            srt_path.write_text(srt_slice, "utf-8")
            txt_path.write_text(txt_slice, "utf-8")

            index["clips"].append({
                "clip_id": clip_id,
                "title": title,
                "source_file": name,
                "audio_src": f"clips/{mp3_name}",
                "srt_src": f"clips/{srt_name}",
                "txt_src": f"clips/{txt_name}",
                "start": round(st, 3),
                "end": round(en, 3),
                "duration": round(en - st, 3),
            })

    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), "utf-8")
    print(f"Done. {args.ia_id} -> {out_dir} | clips: {len(index['clips'])} | warnings: {len(index['warnings'])}")

if __name__ == "__main__":
    main()
