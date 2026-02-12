"""
Build an Übung MP3 by selecting the best N clips from outputs/clips (mp3+txt).

Features:
- Scores clips by digits + demographic signals (age/dob/height/weight).
- Penalizes "aktuelle Anamnese" transitions (Was führt Sie..., Beschwerden...).
- Picks a DIFFERENT set each run via seeded stochastic MMR sampling:
    - greedy if temperature <= 0
    - otherwise softmax sampling over MMR utility
- Prints score ranking to stdout and writes ranking files:
    outputs/ubung/best{N}_###_ranking.csv
    outputs/ubung/best{N}_###_ranking.jsonl

Outputs:
  outputs/ubung/best{N}_###.mp3
  outputs/ubung/best{N}_###.txt
  outputs/ubung/best{N}_###.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None  # type: ignore


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


@dataclass(frozen=True)
class RankedCandidate:
    stem: str
    duration_s: float
    digits: int
    score: float
    group: str
    has_age_q: bool
    has_dob: bool
    has_height: bool
    has_weight: bool
    has_transition: bool


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
    low = stem.lower()
    if " dr " in low:
        return _norm(low.split(" dr ")[0])
    parts = _norm(stem).split()
    return " ".join(parts[:2]) if parts else _norm(stem)


def score_clip(
    text: str, duration_s: float, min_duration: float, max_duration: float
) -> Tuple[float, Dict[str, bool], int]:
    n = _norm(text)
    digits = len(NUM_RE.findall(n))

    has_age_q = bool(AGE_Q_RE.search(n))
    has_dob = bool(DOB_RE.search(n) or DATE_RE.search(n))
    has_height = bool(HEIGHT_RE.search(n) or re.search(r"\b1[,.]\d{2}\b|\b1\d{2}\b", n))
    has_weight = bool(WEIGHT_RE.search(n) or re.search(r"\b\d{2,3}\s*(kg|kilo)\b", n))
    has_transition = bool(TRANSITION_RE.search(n))

    score = 0.0
    score += digits * 2.0
    score += 2.0 if has_age_q else 0.0
    score += 4.0 if has_dob else 0.0
    score += 3.0 if has_height else 0.0
    score += 3.0 if has_weight else 0.0

    if has_transition:
        score -= 12.0
    if duration_s < min_duration:
        score -= (min_duration - duration_s) * 1.0
    if duration_s > max_duration:
        score -= (max_duration - duration_s) * 0.7  # negative, because duration_s > max_duration
        score -= (duration_s - max_duration) * 0.7
    if digits == 0:
        score -= 10.0

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


def make_silence_wav(ffmpeg: str, out_wav: Path, seconds: float, sample_rate: int) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=mono",
            "-t",
            f"{seconds:.3f}",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ],
        check=True,
    )


def mp3_to_wav(ffmpeg: str, mp3: Path, wav: Path, sample_rate: int) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp3),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(wav),
        ],
        check=True,
    )


def concat_wavs_to_mp3(ffmpeg: str, wavs: List[Path], out_mp3: Path) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    for w in wavs:
        cmd += ["-i", str(w)]

    if len(wavs) == 1:
        cmd += ["-c:a", "libmp3lame", "-b:a", "96k", str(out_mp3)]
        subprocess.run(cmd, check=True)
        return

    inputs = "".join([f"[{i}:a]" for i in range(len(wavs))])
    filt = f"{inputs}concat=n={len(wavs)}:v=0:a=1[a]"
    cmd += ["-filter_complex", filt, "-map", "[a]", "-c:a", "libmp3lame", "-b:a", "96k", str(out_mp3)]
    subprocess.run(cmd, check=True)


def next_index(out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_i = 0
    for p in out_dir.glob(f"{prefix}_*.mp3"):
        m = re.match(rf"{re.escape(prefix)}_(\d+)\.mp3$", p.name)
        if m:
            max_i = max(max_i, int(m.group(1)))
    return max_i + 1


def softmax_sample(rng: random.Random, items: List[Tuple[ClipInfo, float]], temperature: float) -> ClipInfo:
    # items: (candidate, utility)
    if temperature <= 0:
        return max(items, key=lambda x: x[1])[0]

    utils = [u for _, u in items]
    m = max(utils)
    # stable softmax
    exps = [math.exp((u - m) / max(temperature, 1e-6)) for u in utils]
    s = sum(exps)
    if s <= 0:
        return max(items, key=lambda x: x[1])[0]

    r = rng.random() * s
    acc = 0.0
    for (ci, _u), w in zip(items, exps):
        acc += w
        if acc >= r:
            return ci
    return items[-1][0]


def pick_best_mmr_stochastic(
    candidates: List[Tuple[ClipInfo, str]],
    count: int,
    lambda_score: float,
    max_per_group: int,
    rng: random.Random,
    temperature: float,
    pool_size: int,
) -> List[ClipInfo]:
    if not candidates:
        return []

    infos = [c[0] for c in candidates]
    texts = {c[0].stem: c[1] for c in candidates}

    smin = min(i.score for i in infos)
    smax = max(i.score for i in infos)
    denom = (smax - smin) if (smax - smin) > 1e-9 else 1.0

    def norm_score(x: float) -> float:
        return (x - smin) / denom

    remaining = candidates[:]
    remaining.sort(key=lambda t: t[0].score, reverse=True)

    selected: List[ClipInfo] = []
    per_group: Dict[str, int] = {}

    while remaining and len(selected) < count:
        if not selected:
            pool = remaining[: max(1, min(pool_size, len(remaining)))]
            pool_items = [(ci, norm_score(ci.score)) for ci, _txt in pool]
            pick = softmax_sample(rng, pool_items, temperature)
            # remove picked
            remaining = [(ci, t) for ci, t in remaining if ci.stem != pick.stem]
            selected.append(pick)
            per_group[pick.group] = per_group.get(pick.group, 0) + 1
            continue

        scored_pool: List[Tuple[ClipInfo, float]] = []
        for ci, txt in remaining:
            if per_group.get(ci.group, 0) >= max_per_group:
                continue

            ns = norm_score(ci.score)
            max_sim = 0.0
            for s in selected:
                max_sim = max(max_sim, similarity(txt, texts[s.stem]))
            util = lambda_score * ns - (1.0 - lambda_score) * max_sim
            scored_pool.append((ci, util))

        if not scored_pool:
            # relax group constraint
            ci, _ = remaining.pop(0)
            selected.append(ci)
            per_group[ci.group] = per_group.get(ci.group, 0) + 1
            continue

        scored_pool.sort(key=lambda x: x[1], reverse=True)
        scored_pool = scored_pool[: max(1, min(pool_size, len(scored_pool)))]

        pick = softmax_sample(rng, scored_pool, temperature)
        remaining = [(ci, t) for ci, t in remaining if ci.stem != pick.stem]
        selected.append(pick)
        per_group[pick.group] = per_group.get(pick.group, 0) + 1

    return selected


def write_ranking(out_csv: Path, out_jsonl: Path, ranking: List[RankedCandidate]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "stem",
                "score",
                "duration_s",
                "digits",
                "group",
                "has_age_q",
                "has_dob",
                "has_height",
                "has_weight",
                "has_transition",
            ]
        )
        for i, r in enumerate(ranking, 1):
            w.writerow(
                [
                    i,
                    r.stem,
                    f"{r.score:.2f}",
                    f"{r.duration_s:.1f}",
                    r.digits,
                    r.group,
                    int(r.has_age_q),
                    int(r.has_dob),
                    int(r.has_height),
                    int(r.has_weight),
                    int(r.has_transition),
                ]
            )

    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, r in enumerate(ranking, 1):
            d = asdict(r)
            d["rank"] = i
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def print_ranking(ranking: List[RankedCandidate], top_n: int) -> None:
    top_n = max(1, min(top_n, len(ranking)))
    print(f"\n=== SCORE RANKING (top {top_n}/{len(ranking)}) ===")
    for i in range(top_n):
        r = ranking[i]
        flags = []
        if r.has_age_q:
            flags.append("AGE")
        if r.has_dob:
            flags.append("DOB")
        if r.has_height:
            flags.append("H")
        if r.has_weight:
            flags.append("W")
        if r.has_transition:
            flags.append("TRANSITION")
        fl = ",".join(flags) if flags else "-"
        print(f"{i+1:03d}. score={r.score:6.2f} dur={r.duration_s:5.1f}s dig={r.digits:2d} [{fl}]  {r.stem}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", default="outputs/clips")
    ap.add_argument("--out-dir", default="outputs/ubung")
    ap.add_argument("--count", type=int, default=12)

    ap.add_argument("--min-duration", type=float, default=12.0)
    ap.add_argument("--max-duration", type=float, default=80.0)
    ap.add_argument("--min-digits", type=int, default=2)

    ap.add_argument("--silence-seconds", type=float, default=0.6)
    ap.add_argument("--sample-rate", type=int, default=16000)

    ap.add_argument("--max-per-group", type=int, default=2)
    ap.add_argument("--lambda-score", type=float, default=0.75)

    ap.add_argument("--seed", type=int, default=0)  # 0 => auto (run id / time)
    ap.add_argument("--temperature", type=float, default=0.7)  # >0 => different picks
    ap.add_argument("--pool-size", type=int, default=40)  # top pool considered each step

    ap.add_argument("--print-ranking", type=int, default=60)
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
        if dur <= 0.0:
            continue

        score, feats, digits = score_clip(text, dur, args.min_duration, args.max_duration)
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
        raise SystemExit("No eligible candidates after filtering.")

    candidates.sort(key=lambda t: t[0].score, reverse=True)

    ranking = [
        RankedCandidate(
            stem=ci.stem,
            duration_s=ci.duration_s,
            digits=ci.digits,
            score=ci.score,
            group=ci.group,
            has_age_q=ci.has_age_q,
            has_dob=ci.has_dob,
            has_height=ci.has_height,
            has_weight=ci.has_weight,
            has_transition=ci.has_transition,
        )
        for ci, _t in candidates
    ]
    print_ranking(ranking, args.print_ranking)

    # Seed: prefer GitHub run id if present
    seed = args.seed
    if seed == 0:
        env_seed = os.environ.get("GITHUB_RUN_ID") or os.environ.get("GITHUB_RUN_NUMBER")
        seed = int(env_seed) if env_seed and env_seed.isdigit() else int(random.SystemRandom().randint(1, 2**31 - 1))
    rng = random.Random(seed)
    print(f"\n=== SELECTION SETTINGS ===")
    print(f"seed={seed} temperature={args.temperature} pool_size={args.pool_size} lambda_score={args.lambda_score} max_per_group={args.max_per_group}")

    selected = pick_best_mmr_stochastic(
        candidates=candidates,
        count=args.count,
        lambda_score=args.lambda_score,
        max_per_group=args.max_per_group,
        rng=rng,
        temperature=args.temperature,
        pool_size=args.pool_size,
    )
    if not selected:
        raise SystemExit("Selection empty. Try relaxing filters.")

    print(f"\n=== SELECTED ({len(selected)}) ===")
    for i, s in enumerate(selected, 1):
        flags = []
        if s.has_age_q:
            flags.append("AGE")
        if s.has_dob:
            flags.append("DOB")
        if s.has_height:
            flags.append("H")
        if s.has_weight:
            flags.append("W")
        if s.has_transition:
            flags.append("TRANSITION")
        fl = ",".join(flags) if flags else "-"
        print(f"{i:02d}. score={s.score:.2f} dur={s.duration_s:.1f}s dig={s.digits} [{fl}] {s.stem}")

    idx = next_index(out_dir, f"best{args.count}")
    base = f"best{args.count}_{idx:03d}"

    out_mp3 = out_dir / f"{base}.mp3"
    out_txt = out_dir / f"{base}.txt"
    out_json = out_dir / f"{base}.json"
    out_rank_csv = out_dir / f"{base}_ranking.csv"
    out_rank_jsonl = out_dir / f"{base}_ranking.jsonl"

    write_ranking(out_rank_csv, out_rank_jsonl, ranking)

    if args.dry_run:
        print(f"\nDRY RUN: would write {out_mp3.name}, {out_txt.name}, {out_json.name}")
        print(f"Ranking written: {out_rank_csv.name}, {out_rank_jsonl.name}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="best_concat_") as td:
        tdir = Path(td)
        tmp_wavs_dir = tdir / "wavs"
        tmp_wavs_dir.mkdir(parents=True, exist_ok=True)

        silence_wav = tdir / "silence.wav"
        if args.silence_seconds > 0:
            make_silence_wav(ffmpeg, silence_wav, args.silence_seconds, args.sample_rate)

        wavs: List[Path] = []
        for i, s in enumerate(selected):
            wav = tmp_wavs_dir / f"{i:02d}.wav"
            mp3_to_wav(ffmpeg, Path(s.mp3), wav, args.sample_rate)
            wavs.append(wav)
            if args.silence_seconds > 0 and i != (len(selected) - 1):
                wavs.append(silence_wav)

        concat_wavs_to_mp3(ffmpeg, wavs, out_mp3)

    parts: List[str] = []
    for i, s in enumerate(selected, 1):
        t = Path(s.txt).read_text(encoding="utf-8", errors="ignore").strip()
        parts.append(f"--- {i:02d} | {s.stem} | {s.duration_s:.1f}s | digits={s.digits} | score={s.score:.2f} ---\n{t}\n")
    out_txt.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")

    out_json.write_text(
        json.dumps(
            {
                "seed": seed,
                "temperature": args.temperature,
                "pool_size": args.pool_size,
                "output_mp3": str(out_mp3),
                "count": len(selected),
                "params": {
                    "min_duration": args.min_duration,
                    "max_duration": args.max_duration,
                    "min_digits": args.min_digits,
                    "silence_seconds": args.silence_seconds,
                    "sample_rate": args.sample_rate,
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

    print(f"\nWrote: {out_mp3}")
    print(f"Wrote: {out_txt}")
    print(f"Wrote: {out_json}")
    print(f"Ranking: {out_rank_csv} / {out_rank_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
