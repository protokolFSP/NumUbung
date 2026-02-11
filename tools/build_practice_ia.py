# ================================
# file: tools/process_next_batch.py
# ================================
"""
Process the NEXT batch of IA files, resuming from what's already in outputs/clips.

Key behavior:
- Lists .m4a from IA, sorts, takes first --limit-total
- Skips stems that already exist in outputs/clips/*.mp3
- Processes only next --batch-size items
- Writes:
    outputs/clips/<stem>.mp3
    outputs/clips/<stem>.txt
    outputs/state.json (progress)
    outputs/debug/<stem>.json (optional)

This avoids reprocessing and enables hourly schedule until all are done.
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
from typing import List, Optional, Tuple
from urllib.parse import quote

import requests

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None  # type: ignore

IA_METADATA_URL = "https://archive.org/metadata/{identifier}"
IA_DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

DEFAULT_START_PHRASES = [
    "wie alt sind sie",
    "wie alt bist du",
    "ihr alter",
    "welches alter",
    "geburtsdatum",
    "wann sind sie geboren",
    "wann bist du geboren",
]

DEFAULT_END_PHRASES = [
    "wie viel wiegen sie",
    "was wiegen sie",
    "ihr gewicht",
    "gewicht",
    "wie gross sind sie",
    "wie groß sind sie",
    "ihre groesse",
    "ihre größe",
    "koerpergroesse",
    "körpergröße",
]

NUM_RE = re.compile(r"\d+|(\bnull\b|\bein\b|\beins\b|\bzwei\b|\bdrei\b|\bvier\b|\bfünf\b|\bfuenf\b|\bsechs\b|\bsieben\b|\bacht\b|\bneun\b|\bzehn\b|\belf\b|\bzwölf\b|\bzwoelf\b)", re.IGNORECASE)
QUESTIONISH_RE = re.compile(r"^\s*(wie|wann|wo|was|welche|welcher|wieviel|haben|nehmen|sind|ist|können|koennen|dürfen|duerfen)\b", re.IGNORECASE)


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


def fuzzy_hit(text: str, phrases: List[str], threshold: int) -> Optional[Tuple[str, int]]:
    if fuzz is None:
        return None
    t = normalize_text(text)
    best_p, best_s = "", -1
    for p in phrases:
        sc = fuzz.partial_ratio(t, p)
        if sc > best_s:
            best_s = sc
            best_p = p
    if best_s >= threshold:
        return best_p, int(best_s)
    return None


def is_hit(text: str, phrases: List[str], threshold: int) -> Optional[str]:
    t = normalize_text(text)
    for p in phrases:
        if p in t:
            return f"substr:{p}"
    fh = fuzzy_hit(text, phrases, threshold)
    if fh:
        p, s = fh
        return f"fuzzy:{p}:{s}"
    return None


def ffmpeg_preview_to_wav(ffmpeg: str, src_url: str, out_wav: Path, preview_seconds: int, sample_rate: int) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-ss", "0", "-t", str(preview_seconds),
        "-i", src_url, "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(out_wav)
    ])


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


def pick_window(
    segs: List[Segment],
    start_phrases: List[str],
    end_phrases: List[str],
    max_search_seconds: float,
    fuzzy_threshold: int,
    answer_tail_seconds: float,
    max_gap_seconds: float,
) -> Optional[Tuple[float, float, str]]:
    # start candidates
    starts: List[Tuple[int, float, str]] = []
    for i, s in enumerate(segs):
        if s.start > max_search_seconds:
            break
        hit = is_hit(rolling_text(segs, i, 2), start_phrases, fuzzy_threshold)
        if hit:
            starts.append((i, s.start, hit))
    if not starts:
        return None

    best: Optional[Tuple[int, float, float, str]] = None  # score, start, end, reason

    for start_i, start_at, start_hit in starts:
        end_q: Optional[Tuple[int, str]] = None
        for j in range(start_i, len(segs)):
            if segs[j].start > max_search_seconds:
                break
            hit = is_hit(rolling_text(segs, j, 2), end_phrases, fuzzy_threshold)
            if hit:
                end_q = (j, hit)

        if not end_q:
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

        # score: prefer windows with many digits
        txt = " ".join(s.text for s in segs if s.start >= start_at and s.end <= end_at)
        digits = len(re.findall(r"\d+", txt))
        score = digits

        reason = f"start={start_hit}; end={end_hit}; digits={digits}"
        if best is None or score > best[0]:
            best = (score, start_at, end_at, reason)

    if not best:
        return None
    _, st, en, rs = best
    if en <= st:
        return None
    return st, en, rs


def cut_clip_mp3(ffmpeg: str, preview_wav: Path, start: float, end: float, out_mp3: Path) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.001, end - start)
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", str(preview_wav),
        "-c:a", "libmp3lame", "-b:a", "96k",
        str(out_mp3),
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identifier", required=True)
    ap.add_argument("--out-dir", default="outputs")

    ap.add_argument("--limit-total", type=int, default=252)
    ap.add_argument("--batch-size", type=int, default=20)

    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default="de")

    ap.add_argument("--preview-seconds", type=int, default=300)
    ap.add_argument("--first-minutes", type=float, default=4.0)
    ap.add_argument("--answer-tail-seconds", type=float, default=25.0)
    ap.add_argument("--max-gap-seconds", type=float, default=1.2)
    ap.add_argument("--fuzzy-threshold", type=int, default=85)

    ap.add_argument("--pre-pad", type=float, default=0.25)
    ap.add_argument("--post-pad", type=float, default=0.25)
    ap.add_argument("--sample-rate", type=int, default=16000)

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    ffmpeg = _which_ffmpeg()
    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    debug_dir = out_dir / "debug"
    clips_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # resume state: existing clips
    done_stems = {p.stem for p in clips_dir.glob("*.mp3")}
    all_files = fetch_m4a_files(args.identifier)[: args.limit_total]

    pending: List[str] = []
    for fn in all_files:
        stem = Path(fn).with_suffix("").name
        if stem not in done_stems:
            pending.append(fn)

    to_process = pending[: args.batch_size]

    state = {
        "identifier": args.identifier,
        "limit_total": args.limit_total,
        "done": len(done_stems),
        "pending": len(pending),
        "this_run": len(to_process),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done={len(done_stems)} pending={len(pending)} process_now={len(to_process)} (limit_total={args.limit_total})")

    if not to_process:
        print("Nothing to do. All processed (or no pending within limit_total).")
        return 0

    start_phrases = [normalize_text(x) for x in DEFAULT_START_PHRASES]
    end_phrases = [normalize_text(x) for x in DEFAULT_END_PHRASES]
    max_search_seconds = args.first_minutes * 60.0

    with tempfile.TemporaryDirectory(prefix="numubung_") as td:
        tdir = Path(td)

        for idx, fn in enumerate(to_process, start=1):
            stem = Path(fn).with_suffix("").name
            url = ia_url(args.identifier, fn)
            print(f"[{idx}/{len(to_process)}] {fn}")

            preview_wav = tdir / f"{idx:03d}_{stem}.wav"
            try:
                ffmpeg_preview_to_wav(ffmpeg, url, preview_wav, args.preview_seconds, args.sample_rate)
            except subprocess.CalledProcessError:
                print("  !! preview failed")
                continue

            try:
                segs = transcribe_preview(preview_wav, args.model, args.language)
            except Exception as e:
                print(f"  !! transcribe failed: {e}")
                continue

            picked = pick_window(
                segs=segs,
                start_phrases=start_phrases,
                end_phrases=end_phrases,
                max_search_seconds=max_search_seconds,
                fuzzy_threshold=args.fuzzy_threshold,
                answer_tail_seconds=args.answer_tail_seconds,
                max_gap_seconds=args.max_gap_seconds,
            )

            if not picked:
                print("  !! window not found (skip)")
                if args.debug:
                    (debug_dir / f"{stem}.json").write_text(
                        json.dumps({"file": fn, "url": url, "segments": [asdict(s) for s in segs]}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                continue

            start, end, reason = picked
            start = max(0.0, start - args.pre_pad)
            end = min(float(args.preview_seconds), end + args.post_pad)

            out_mp3 = clips_dir / f"{stem}.mp3"
            out_txt = clips_dir / f"{stem}.txt"

            try:
                cut_clip_mp3(ffmpeg, preview_wav, start, end, out_mp3)
                # write text
                parts = [s.text.strip() for s in segs if s.start >= start and s.end <= end and s.text.strip()]
                out_txt.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
            except subprocess.CalledProcessError:
                print("  !! cut failed")
                continue

            if args.debug:
                (debug_dir / f"{stem}.json").write_text(
                    json.dumps(
                        {"file": fn, "url": url, "picked": {"start": start, "end": end, "reason": reason}, "segments": [asdict(s) for s in segs]},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
