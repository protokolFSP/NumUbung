# file: tools/process_next_batch.py
"""
Process IA files in resumable batches and produce a 20-clip practice MP3 per batch.

Outputs (committable):
- outputs/clips/<stem>.mp3
- outputs/clips/<stem>.txt
- outputs/ubung/batch_###.mp3
- outputs/state.json
- outputs/failures.json
- outputs/debug/<stem>.json   (optional)

Key behavior:
- "batch-size" means: target number of PRODUCED clips in this run.
- If some files fail (no window), continues with later files until target is met.
- Avoids infinite retries via failures.json (max_failures).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None  # type: ignore


IA_METADATA_URL = "https://archive.org/metadata/{identifier}"
IA_DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

# Expanded to cover variations seen in your transcripts:
# - "Wie groß und wie schwer sind Sie?" :contentReference[oaicite:4]{index=4}
# - "Wissen Sie zurzeit, wie viel Sie wiegen?" :contentReference[oaicite:5]{index=5}
# - "Kennen Sie Ihr aktueller Gewicht?" :contentReference[oaicite:6]{index=6}
# - "Wie viele ... aktuell wiegen?" :contentReference[oaicite:7]{index=7}
START_PHRASES = [
    "wie alt sind sie",
    "wie alt bist du",
    "jahre alt",
    "ihr alter",
    "welches alter",
    "geburtsdatum",
    "wann sind sie geboren",
    "wann bist du geboren",
    "wann genau sind sie geboren",
    "geboren am",
]

END_PHRASES = [
    # height
    "wie gross sind sie",
    "wie groß sind sie",
    "ihre groesse",
    "ihre größe",
    "koerpergroesse",
    "körpergröße",
    # weight
    "wie viel wiegen sie",
    "wie viel wiegen sie zurzeit",
    "wie viel wiegen sie momentan",
    "wissen sie zurzeit wie viel sie wiegen",
    "kennen sie ihr aktuelles gewicht",
    "kennen sie ihr gewicht",
    "ihr gewicht",
    "gewicht",
    "wie schwer sind sie",
    # combined
    "wie gross und wie schwer sind sie",
    "wie groß und wie schwer sind sie",
    "wie gross und wie schwer",
    "wie groß und wie schwer",
    "wie groß und wie schwer sind sie",
    "wie groß und wie schwer bist du",
    # "wie viele ... aktuell wiegen" :contentReference[oaicite:8]{index=8}
    "wie viele sie aktuell wiegen",
    "wie viele sie aktuell wiegen",
    "wie viele wiegen sie aktuell",
]

NUM_RE = re.compile(
    r"\d+|(\bnull\b|\bein\b|\beins\b|\bzwei\b|\bdrei\b|\bvier\b|\bfünf\b|\bfuenf\b|\bsechs\b|\bsieben\b|\bacht\b|\bneun\b|\bzehn\b|\belf\b|\bzwölf\b|\bzwoelf\b)",
    re.IGNORECASE,
)
QUESTIONISH_RE = re.compile(
    r"^\s*(wie|wann|wo|was|welche|welcher|wieviel|haben|nehmen|sind|ist|können|koennen|dürfen|duerfen)\b",
    re.IGNORECASE,
)

# Fallback tokens (even if phrases missed)
START_TOKENS = ["alt", "jahre", "geboren", "geburtsdatum"]
END_TOKENS = ["kg", "kilo", "gewicht", "m.", "meter", "m "]


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def _which_ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise SystemExit("ffmpeg not found. Install: apt-get install ffmpeg")
    return p


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_m4a_files(identifier: str, timeout_s: int = 30) -> List[str]:
    url = IA_METADATA_URL.format(identifier=identifier)
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    meta = r.json()

    files = meta.get("files", [])
    if isinstance(files, dict):
        files = list(files.values())

    names: List[str] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("file") or entry.get("filename")
        if isinstance(name, str) and name.lower().endswith(".m4a"):
            names.append(name)

    return sorted(names, key=lambda s: s.casefold())


def ia_url(identifier: str, filename: str) -> str:
    return IA_DOWNLOAD_URL.format(identifier=identifier, filename=quote(filename))


def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("ß", "ss").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    s = re.sub(r"[^a-z0-9\s./-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def rolling_text(segs: List[Segment], i: int, window: int = 2) -> str:
    lo = max(0, i - window)
    return " ".join(segs[j].text for j in range(lo, i + 1)).strip()


def _fuzzy_best(text: str, phrases: List[str]) -> Tuple[str, int]:
    if fuzz is None:
        return "", -1
    t = normalize_text(text)
    best_p, best_s = "", -1
    for p in phrases:
        sc = fuzz.partial_ratio(t, p)
        if sc > best_s:
            best_s = sc
            best_p = p
    return best_p, int(best_s)


def is_hit(text: str, phrases: List[str], threshold: int) -> Optional[str]:
    t = normalize_text(text)
    for p in phrases:
        if p in t:
            return f"substr:{p}"
    if fuzz is None:
        return None
    p, s = _fuzzy_best(text, phrases)
    if s >= threshold:
        return f"fuzzy:{p}:{s}"
    return None


def ffmpeg_preview_to_wav(ffmpeg: str, src_url: str, out_wav: Path, preview_seconds: int, sample_rate: int) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0",
            "-t",
            str(preview_seconds),
            "-i",
            src_url,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
    )


def transcribe_preview(preview_wav: Path, model: str, language: str) -> List[Segment]:
    from faster_whisper import WhisperModel  # type: ignore

    wm = WhisperModel(model, device="cpu", compute_type="int8")
    seg_iter, _info = wm.transcribe(
        str(preview_wav),
        language=language,
        task="transcribe",
        beam_size=1,
        vad_filter=True,
    )
    return [Segment(float(s.start), float(s.end), str(s.text)) for s in seg_iter]


def _find_token_index(segs: List[Segment], tokens: List[str], max_seconds: float, first: bool) -> Optional[int]:
    rng = range(len(segs)) if first else range(len(segs) - 1, -1, -1)
    for i in rng:
        if segs[i].start > max_seconds:
            continue
        t = normalize_text(rolling_text(segs, i, 2))
        if any(tok in t for tok in tokens):
            return i
    return None


def pick_window(
    segs: List[Segment],
    first_minutes: float,
    fuzzy_threshold: int,
    answer_tail_seconds: float,
    max_gap_seconds: float,
) -> Optional[Tuple[float, float, str]]:
    max_search = first_minutes * 60.0

    # Candidates for start (phrases)
    start_candidates: List[Tuple[int, str]] = []
    for i, s in enumerate(segs):
        if s.start > max_search:
            break
        hit = is_hit(rolling_text(segs, i, 2), START_PHRASES, fuzzy_threshold)
        if hit:
            start_candidates.append((i, hit))

    # Fallback: token start
    if not start_candidates:
        tok_i = _find_token_index(segs, START_TOKENS, max_search, first=True)
        if tok_i is not None:
            start_candidates.append((tok_i, "token_start"))

    if not start_candidates:
        return None

    best: Optional[Tuple[int, float, float, str]] = None  # score, start, end, reason

    for start_i, start_hit in start_candidates:
        # Find last end question after start
        end_q: Optional[Tuple[int, str]] = None
        for j in range(start_i, len(segs)):
            if segs[j].start > max_search:
                break
            hit = is_hit(rolling_text(segs, j, 2), END_PHRASES, fuzzy_threshold)
            if hit:
                end_q = (j, hit)

        # Fallback: token end (kg/m)
        if end_q is None:
            tok_j = _find_token_index(segs, END_TOKENS, max_search, first=False)
            if tok_j is not None and tok_j >= start_i:
                end_q = (tok_j, "token_end")

        if end_q is None:
            continue

        end_q_idx, end_hit = end_q
        q_start = segs[end_q_idx].start
        end_at = segs[end_q_idx].end
        saw_num = False

        for k in range(end_q_idx + 1, len(segs)):
            if segs[k].start - q_start > answer_tail_seconds:
                break

            gap = segs[k].start - segs[k - 1].end
            t = segs[k].text.strip()

            if t and NUM_RE.search(t):
                saw_num = True

            if saw_num and gap >= max_gap_seconds:
                break
            if saw_num and t and QUESTIONISH_RE.search(t) and not NUM_RE.search(t):
                break

            end_at = segs[k].end

        txt = " ".join(s.text for s in segs if s.start >= segs[start_i].start and s.end <= end_at)
        digit_count = len(re.findall(r"\d+", txt))
        score = digit_count

        reason = f"start={start_hit}; end={end_hit}; digits={digit_count}"
        if best is None or score > best[0]:
            best = (score, segs[start_i].start, end_at, reason)

    if not best:
        return None

    _, st, en, rs = best
    if en <= st:
        return None
    return st, en, rs


def cut_clip_mp3(ffmpeg: str, preview_wav: Path, start: float, end: float, out_mp3: Path) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.001, end - start)
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{dur:.3f}",
            "-i",
            str(preview_wav),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "96k",
            str(out_mp3),
        ]
    )


def make_silence_wav(ffmpeg: str, out_wav: Path, seconds: float, sample_rate: int) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run(
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
        ]
    )


def mp3_to_wav(ffmpeg: str, mp3: Path, wav: Path, sample_rate: int) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    _run(
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
        ]
    )


def wavs_to_mp3_filter_concat(ffmpeg: str, wavs: List[Path], out_mp3: Path) -> None:
    if not wavs:
        raise SystemExit("No WAVs to concat.")
    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    for w in wavs:
        cmd += ["-i", str(w)]

    if len(wavs) == 1:
        cmd += ["-c:a", "libmp3lame", "-b:a", "96k", str(out_mp3)]
        _run(cmd)
        return

    inputs = "".join([f"[{i}:a]" for i in range(len(wavs))])
    filt = f"{inputs}concat=n={len(wavs)}:v=0:a=1[a]"
    cmd += ["-filter_complex", filt, "-map", "[a]", "-c:a", "libmp3lame", "-b:a", "96k", str(out_mp3)]
    _run(cmd)


def next_batch_index(ubung_dir: Path) -> int:
    ubung_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for p in ubung_dir.glob("batch_*.mp3"):
        m = re.match(r"batch_(\d+)\.mp3$", p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identifier", required=True)
    ap.add_argument("--out-dir", default="outputs")

    ap.add_argument("--limit-total", type=int, default=252)
    ap.add_argument("--batch-size", type=int, default=20, help="Target number of PRODUCED clips this run.")
    ap.add_argument("--max-attempts", type=int, default=120, help="Max files to try in this run to fill batch.")
    ap.add_argument("--max-failures", type=int, default=3, help="After N failures, mark file as permanently skipped.")

    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default="de")

    ap.add_argument("--preview-seconds", type=int, default=300)
    ap.add_argument("--retry-preview-seconds", type=int, default=600)
    ap.add_argument("--first-minutes", type=float, default=4.0)
    ap.add_argument("--retry-first-minutes", type=float, default=7.0)

    ap.add_argument("--answer-tail-seconds", type=float, default=25.0)
    ap.add_argument("--max-gap-seconds", type=float, default=1.2)
    ap.add_argument("--fuzzy-threshold", type=int, default=84)

    ap.add_argument("--pre-pad", type=float, default=0.25)
    ap.add_argument("--post-pad", type=float, default=0.25)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--silence-seconds", type=float, default=0.6)

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    ffmpeg = _which_ffmpeg()
    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    debug_dir = out_dir / "debug"
    ubung_dir = out_dir / "ubung"
    clips_dir.mkdir(parents=True, exist_ok=True)
    ubung_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    failures_path = out_dir / "failures.json"
    failures: Dict[str, Dict] = _load_json(failures_path, {})

    def is_perm_skipped(stem: str) -> bool:
        ent = failures.get(stem)
        return bool(ent and ent.get("permanent"))

    done_stems = {p.stem for p in clips_dir.glob("*.mp3")}

    all_files = fetch_m4a_files(args.identifier)[: args.limit_total]
    pending: List[str] = []
    for fn in all_files:
        stem = Path(fn).with_suffix("").name
        if stem in done_stems:
            continue
        if is_perm_skipped(stem):
            continue
        pending.append(fn)

    target = args.batch_size
    produced_mp3: List[Path] = []
    tried = 0

    state = {
        "identifier": args.identifier,
        "limit_total": args.limit_total,
        "done": len(done_stems),
        "pending": len(pending),
        "target_produced": target,
        "max_attempts": args.max_attempts,
    }
    _save_json(out_dir / "state.json", state)

    if not pending:
        print("Nothing pending.")
        return 0

    with tempfile.TemporaryDirectory(prefix="numubung_") as td:
        tdir = Path(td)

        for fn in pending:
            if len(produced_mp3) >= target:
                break
            if tried >= args.max_attempts:
                break
            tried += 1

            stem = Path(fn).with_suffix("").name
            url = ia_url(args.identifier, fn)
            print(f"[try {tried}/{args.max_attempts}] produce {len(produced_mp3)}/{target}: {fn}")

            attempt_cfgs = [
                (args.preview_seconds, args.first_minutes),
                (args.retry_preview_seconds, args.retry_first_minutes),
            ]

            ok = False
            last_reason = "no_window"
            last_debug = {}

            for pass_i, (preview_s, first_m) in enumerate(attempt_cfgs, start=1):
                preview_wav = tdir / f"{tried:03d}_{stem}_{preview_s}s.wav"
                try:
                    ffmpeg_preview_to_wav(ffmpeg, url, preview_wav, preview_s, args.sample_rate)
                except subprocess.CalledProcessError:
                    last_reason = f"preview_failed_{preview_s}s"
                    continue

                try:
                    segs = transcribe_preview(preview_wav, args.model, args.language)
                except Exception as e:
                    last_reason = f"transcribe_failed:{e}"
                    continue

                picked = pick_window(
                    segs=segs,
                    first_minutes=first_m,
                    fuzzy_threshold=args.fuzzy_threshold,
                    answer_tail_seconds=args.answer_tail_seconds,
                    max_gap_seconds=args.max_gap_seconds,
                )
                if not picked:
                    last_reason = f"window_not_found(pass={pass_i},preview={preview_s},firstm={first_m})"
                    last_debug = {"file": fn, "url": url, "pass": pass_i, "preview_seconds": preview_s, "first_minutes": first_m, "segments": [asdict(s) for s in segs]}
                    continue

                start, end, reason = picked
                start = max(0.0, start - args.pre_pad)
                end = min(float(preview_s), end + args.post_pad)

                out_mp3 = clips_dir / f"{stem}.mp3"
                out_txt = clips_dir / f"{stem}.txt"

                try:
                    cut_clip_mp3(ffmpeg, preview_wav, start, end, out_mp3)
                    parts = [s.text.strip() for s in segs if s.start >= start and s.end <= end and s.text.strip()]
                    out_txt.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
                except subprocess.CalledProcessError:
                    last_reason = "cut_failed"
                    continue

                if args.debug:
                    _save_json(
                        debug_dir / f"{stem}.json",
                        {
                            "file": fn,
                            "url": url,
                            "picked": {"start": start, "end": end, "reason": reason, "preview_seconds": preview_s, "first_minutes": first_m},
                            "segments": [asdict(s) for s in segs],
                        },
                    )

                produced_mp3.append(out_mp3)
                ok = True
                break

            if not ok:
                ent = failures.get(stem, {"attempts": 0, "permanent": False})
                ent["attempts"] = int(ent.get("attempts", 0)) + 1
                ent["last_reason"] = last_reason
                if ent["attempts"] >= args.max_failures:
                    ent["permanent"] = True
                failures[stem] = ent
                _save_json(failures_path, failures)

                if args.debug and last_debug:
                    _save_json(debug_dir / f"{stem}_fail.json", last_debug)

    if not produced_mp3:
        print("No clips produced in this run; no batch mp3 created.")
        state["this_run_produced"] = 0
        state["this_run_tried"] = tried
        _save_json(out_dir / "state.json", state)
        return 0

    # Build this run's Übung MP3
    batch_idx = next_batch_index(ubung_dir)
    batch_mp3 = ubung_dir / f"batch_{batch_idx:03d}.mp3"

    silence_wav = out_dir / "_silence.wav"
    make_silence_wav(ffmpeg, silence_wav, args.silence_seconds, args.sample_rate)

    wavs: List[Path] = []
    for mp3 in produced_mp3:
        wav = out_dir / "_tmp" / f"{mp3.stem}.wav"
        mp3_to_wav(ffmpeg, mp3, wav, args.sample_rate)
        wavs.extend([wav, silence_wav])
    wavs = wavs[:-1]

    print(f"Creating {batch_mp3} (clips={len(produced_mp3)})")
    wavs_to_mp3_filter_concat(ffmpeg, wavs, batch_mp3)

    state["this_run_produced"] = len(produced_mp3)
    state["this_run_tried"] = tried
    state["last_batch_mp3"] = str(batch_mp3)
    _save_json(out_dir / "state.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
