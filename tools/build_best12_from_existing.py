# =========================================
# file: tools/build_best12_from_existing.py
# =========================================
"""
Build the best 12-clip practice MP3 from existing outputs/clips/*.mp3 + *.txt.

Usage:
  python tools/build_best12_from_existing.py \
    --clips-dir outputs/clips \
    --out-dir outputs/ubung \
    --count 12 \
    --min-duration 12 \
    --max-duration 80 \
    --min-digits 2 \
    --silence-seconds 0.6 \
    --max-per-group 2 \
    --lambda-score 0.75

Outputs:
  outputs/ubung/best12_###.mp3
  outputs/ubung/best12_###.txt
  outputs/ubung/best12_###.json

Notes:
- Requires ffmpeg/ffprobe available in PATH.
- rapidfuzz is optional (better diversity). If missing, falls back to simple selection.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None  # type: ignore


# --- Heuristics (tune here) ---
TRANSITION_RE = re.compile(
    r"\b(was\s+f(ü|ue)hrt\s+sie(\s+zu\s+uns|\s+heute|\s+her)?)\b|"
    r"\b(was\s+kann\s+ich\s+f(ü|ue)r\s+sie\s+tun)\b|"
    r"\b(was\s+bringt\s+sie\s+zu\s+uns)\b|"
    r"\b(was\s+f(ü|ue)r\s+beschwerden\s+haben\s+sie)\b|"
    r"\b(beschwerden)\b",
    re.IGNORECASE,
)

DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b")
AGE_Q_RE = re.compile(r"\bwie\s+alt\b", re.IGNORECASE)
DOB_RE = re.compile(r"\b(geboren|geburtsdatum|wann\s+sind\s+sie\s+geboren)\b", re.IGNORECASE)
HEIGHT_RE = re.compile(r"\b(wie\s+gro(ß|ss)|wie\s+gross|k(ö|oe)rpergr(ö|oe)(ß|ss)e)\b", re.IGNORECASE)
WEIGHT_RE = re.compile(r"\b(wie\s+viel\s+wiegen|wie\s+schwer|gewicht|kilo|kg)\b", re.IGNORECASE)
NUM_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class ClipInfo:
    stem: str
    mp3: str
    txt: str
    duration_s: float
    digits: int
    has_age_q: bool
    has_dob: bool
    has_height: bool
    has_weight: bool
    has_transition: bool
    score: float
    group: str


def _which(bin_name: str) -> str:
    p = shutil.which(bin_name)
    if not p:
        raise SystemExit(f"{bin_name} not found in PATH.")
    return p


def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.stdout.strip()


def _norm(s: str) -> str:
    s = s.lower()
    s = s.replace("ß", "ss").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    s = re.sub(r"[^a-z0-9\s./-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ffprobe_duration_s(ffprobe: str, mp3_path: Path) -> float:
    out = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(mp3_path),
        ]
    )
    try:
        return float(out)
    except Exception:
        return 0.0


def group_key_from_stem(stem: str) -> str:
    # Try to keep diagnosis/case group stable across variants
    s = stem
    low = s.lower()
    if " dr " in low:
        cut = low.split(" dr ")[0]
        return _norm(cut)
    # fallback: first 2 words
    parts = _norm(s).split()
    return " ".join(parts[:2]) if parts else _norm(s)


def score_clip(text: str, duration_s: float, min_duration: float, max_duration: float) -> Tuple[float, Dict[str, bool], int]:
    n = _norm(text)
    digits = len(NUM_RE.findall(n))

    has_age_q = bool(AGE_Q_RE.search(n))
    has_dob = bool(DOB_RE.search(n) or DATE_RE.search(n))
    has_height = bool(HEIGHT_RE.search(n) or re.search(r"\b1[,.]\d{2}\b|\b1\d{2}\b", n))
    has_weight = bool(WEIGHT_RE.search(n) or re.search(r"\b\d{2,3}\s*(kg|kilo)\b", n))
    has_transition = bool(TRANSITION_RE.search(n))

    # Base points: numbers + demographic completeness
    score = 0.0
    score += digits * 2.0
    score += 2.0 if has_age_q else 0.0
    score += 4.0 if has_dob else 0.0
    score += 3.0 if has_height else 0.0
    score += 3.0 if has_weight else 0.0

    # Penalties
    if has_transition:
        score -= 10.0  # keep "aktuelle Anamnese" out
    if duration_s < min_duration:
        score -= (min_duration - duration_s) * 1.0
    if duration_s > max_duration:
        score -= (duration_s - max_duration) * 0.7
    if digits == 0:
        score -= 8.0

    feats = {
        "has_age_q": has_age_q,
        "has_dob": has_dob,
        "has_height": has_height,
        "has_weight": has_weight,
        "has_transition": has_transition,
    }
    return score, feats, digits


def similarity(a: str, b: str) -> float:
    if fuzz is None:
        return 0.0
    return float(fuzz.token_set_ratio(_norm(a), _norm(b))) / 100.0


def pick_best_mmr(
    candidates: List[Tuple[ClipInfo, str]],
    count: int,
    lambda_score: float,
    max_per_group: int,
) -> List[ClipInfo]:
    if not candidates:
        return []

    # Normalize scores to [0,1]
    infos = [c[0] for c in candidates]
    texts = {c[0].stem: c[1] for c in candidates}
    smin = min(i.score for i in infos)
    smax = max(i.score for i in infos)
    denom = (smax - smin) if (smax - smin) > 1e-9 else 1.0

    def norm_score(x: float) -> float:
        return (x - smin) / denom

    # Start with best score
    remaining = candidates[:]
    remaining.sort(key=lambda t: t[0].score, reverse=True)

    selected: List[ClipInfo] = []
    per_group: Dict[str, int] = {}

    while remaining and len(selected) < count:
        if not selected:
            pick = remaining.pop(0)[0]
            selected.append(pick)
            per_group[pick.group] = per_group.get(pick.group, 0) + 1
            continue

        best_idx = -1
        best_val = -1e9

        for idx, (ci, txt) in enumerate(remaining):
            if per_group.get(ci.group, 0) >= max_per_group:
                continue

            ns = norm_score(ci.score)
            max_sim = 0.0
            for s in selected:
                max_sim = max(max_sim, similarity(txt, texts[s.stem]))
            val = lambda_score * ns - (1.0 - lambda_score) * max_sim
            if val > best_val:
                best_val = val
                best_idx = idx

        if best_idx == -1:
            # group constraint blocks; relax
            ci, _txt = remaining.pop(0)
        else:
            ci, _txt = remaining.pop(best_idx)

        selected.append(ci)
        per_group[ci.group] = per_group.get(ci.group, 0) + 1

    return selected


def make_silence_mp3(ffmpeg: str, out_mp3: Path, seconds: float) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=mono:sample_rate=44100",
            "-t",
            f"{seconds:.3f}",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "96k",
            str(out_mp3),
        ],
        check=True,
    )


def concat_mp3(ffmpeg: str, mp3s: List[Path], out_mp3: Path) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="concat_") as td:
        tdir = Path(td)
        lst = tdir / "list.txt"
        lines = []
        for p in mp3s:
            lines.append(f"file '{p.as_posix()}'")
        lst.write_text("\n".join(lines) + "\n", encoding="utf-8")

        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "96k",
                str(out_mp3),
            ],
            check=True,
        )


def next_index(out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_i = 0
    for p in out_dir.glob(f"{prefix}_*.mp3"):
        m = re.match(rf"{re.escape(prefix)}_(\d+)\.mp3$", p.name)
        if m:
            max_i = max(max_i, int(m.group(1)))
    return max_i + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", default="outputs/clips")
    ap.add_argument("--out-dir", default="outputs/ubung")
    ap.add_argument("--count", type=int, default=12)

    ap.add_argument("--min-duration", type=float, default=12.0)
    ap.add_argument("--max-duration", type=float, default=80.0)
    ap.add_argument("--min-digits", type=int, default=2)

    ap.add_argument("--silence-seconds", type=float, default=0.6)
    ap.add_argument("--max-per-group", type=int, default=2)
    ap.add_argument("--lambda-score", type=float, default=0.75)

    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ffmpeg = _which("ffmpeg")
    ffprobe = _which("ffprobe")

    clips_dir = Path(args.clips_dir)
    out_dir = Path(args.out_dir)

    txt_files = sorted(clips_dir.glob("*.txt"), key=lambda p: p.name.casefold())
    if not txt_files:
        raise SystemExit(f"No .txt found in {clips_dir}")

    candidates: List[Tuple[ClipInfo, str]] = []

    for txt_path in txt_files:
        stem = txt_path.stem
        mp3_path = clips_dir / f"{stem}.mp3"
        if not mp3_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        dur = ffprobe_duration_s(ffprobe, mp3_path)
        score, feats, digits = score_clip(text, dur, args.min_duration, args.max_duration)

        if dur <= 0.0:
            continue
        if digits < args.min_digits:
            continue

        ci = ClipInfo(
            stem=stem,
            mp3=str(mp3_path),
            txt=str(txt_path),
            duration_s=dur,
            digits=digits,
            has_age_q=feats["has_age_q"],
            has_dob=feats["has_dob"],
            has_height=feats["has_height"],
            has_weight=feats["has_weight"],
            has_transition=feats["has_transition"],
            score=score,
            group=group_key_from_stem(stem),
        )
        candidates.append((ci, text))

    if not candidates:
        raise SystemExit("No eligible candidates after filtering (min_durations/min_digits).")

    # Prefer non-transition clips; still keep them as fallback
    candidates.sort(key=lambda t: t[0].score, reverse=True)

    selected = pick_best_mmr(
        candidates=candidates,
        count=args.count,
        lambda_score=args.lambda_score,
        max_per_group=args.max_per_group,
    )

    if not selected:
        raise SystemExit("Selection failed unexpectedly.")

    selected.sort(key=lambda x: x.score, reverse=True)

    if args.dry_run:
        for i, s in enumerate(selected, 1):
            print(f"{i:02d}. {s.stem}  score={s.score:.2f} dur={s.duration_s:.1f}s digits={s.digits} group={s.group}")
        return 0

    idx = next_index(out_dir, "best12")
    base = f"best12_{idx:03d}"
    out_mp3 = out_dir / f"{base}.mp3"
    out_txt = out_dir / f"{base}.txt"
    out_json = out_dir / f"{base}.json"

    # Build concat list with optional silence
    mp3s: List[Path] = []
    silence_mp3: Optional[Path] = None
    if args.silence_seconds > 0.0:
        silence_mp3 = out_dir / "_silence.mp3"
        make_silence_mp3(ffmpeg, silence_mp3, args.silence_seconds)

    for i, s in enumerate(selected):
        mp3s.append(Path(s.mp3))
        if silence_mp3 and i != (len(selected) - 1):
            mp3s.append(silence_mp3)

    concat_mp3(ffmpeg, mp3s, out_mp3)

    # Write combined transcript
    parts: List[str] = []
    for i, s in enumerate(selected, 1):
        t = Path(s.txt).read_text(encoding="utf-8", errors="ignore").strip()
        parts.append(f"--- {i:02d} | {s.stem} | {s.duration_s:.1f}s | digits={s.digits} | score={s.score:.2f} ---\n{t}\n")
    out_txt.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")

    out_json.write_text(
        json.dumps(
            {
                "output_mp3": str(out_mp3),
                "count": len(selected),
                "params": {
                    "min_duration": args.min_duration,
                    "max_duration": args.max_duration,
                    "min_digits": args.min_digits,
                    "silence_seconds": args.silence_seconds,
                    "max_per_group": args.max_per_group,
                    "lambda_score": args.lambda_score,
                },
                "selected": [asdict(s) for s in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {out_mp3}")
    print(f"Wrote: {out_txt}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

