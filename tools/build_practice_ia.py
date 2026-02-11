# file: tools/build_practice_ia.py
"""
Build a German number-practice audio from an Internet Archive item (first N .m4a files).

Outputs:
- outputs/clips/<stem>.mp3
- outputs/clips/<stem>.txt
- outputs/practice_first20.mp3
- outputs/manifest.json
- outputs/debug/<stem>.json   (optional, --debug)

Robust matching:
- Regex + fuzzy matching (RapidFuzz) on normalized text
- Window selection with scoring
- Answer inclusion after end-question:
  - include number segments
  - stop if we saw numbers AND (gap > max_gap_s OR next segment looks like a new question)

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
from typing import Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover
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
    "wann sind sie gebohren",
    "wann bist du geboren",
]

DEFAULT_END_PHRASES = [
    "wie viel wiegen sie",
    "was wiegen sie",
    "ihr gewicht",
    "wie groß sind sie",
    "wie gross sind sie",
    "ihre größe",
    "ihre groesse",
    "körpergröße",
    "koerpergroesse",
]

NUM_RE = re.compile(r"\d+|(\bnull\b|\bein\b|\beins\b|\bzwei\b|\bdrei\b|\bvier\b|\bfünf\b|\bfuenf\b|\bsechs\b|\bsieben\b|\bacht\b|\bneun\b|\bzehn\b|\belf\b|\bzwölf\b|\bzwoelf\b)", re.IGNORECASE)
DOB_RE = re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b")
QUESTIONISH_RE = re.compile(
    r"^\s*(wie|wann|wo|was|welche|welcher|wieviel|haben|nehmen|sind|ist|können|koennen|dürfen|duerfen)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Clip:
    filename: str
    stem: str
    url: str
    start: float
    end: float
    reason: str
    clip_mp3: str
    clip_txt: str


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


def ffmpeg_preview_to_wav(ffmpeg: str, src_url: str, out_wav: Path, preview_seconds: int, sample_rate: int) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
    _run(cmd)


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


def rolling_text(segs: List[Segment], i: int, window: int = 2) -> str:
    lo = max(0, i - window)
    return " ".join(segs[j].text for j in range(lo, i + 1)).strip()


def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("ß", "ss").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    s = re.sub(r"[^a-z0-9\s./-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fuzzy_contains(text: str, phrases: List[str], threshold: int) -> Optional[Tuple[str, int]]:
    if fuzz is None:
        return None
    t = normalize_text(text)
    best_phrase = ""
    best_score = -1
    for p in phrases:
        score = fuzz.partial_ratio(t, p)
        if score > best_score:
            best_score = score
            best_phrase = p
    if best_score >= threshold:
        return best_phrase, int(best_score)
    return None


def is_start_hit(text: str, start_phrases: List[str], threshold: int) -> Optional[str]:
    t = normalize_text(text)
    for p in start_phrases:
        if p in t:
            return f"substr:{p}"
    fz = fuzzy_contains(text, start_phrases, threshold)
    if fz:
        phrase, score = fz
        return f"fuzzy:{phrase}:{score}"
    return None


def is_end_hit(text: str, end_phrases: List[str], threshold: int) -> Optional[str]:
    t = normalize_text(text)
    for p in end_phrases:
        if p in t:
            return f"substr:{p}"
    fz = fuzzy_contains(text, end_phrases, threshold)
    if fz:
        phrase, score = fz
        return f"fuzzy:{phrase}:{score}"
    return None


def pick_window_scored(
    segs: List[Segment],
    start_phrases: List[str],
    end_phrases: List[str],
    max_search_seconds: float,
    fuzzy_threshold: int,
    answer_tail_seconds: float,
    max_gap_s: float,
) -> Optional[Tuple[float, float, str, int, int]]:
    """
    Returns: (start_at, end_at, reason, start_idx, end_q_idx)
    """

    # collect candidate starts
    start_candidates: List[Tuple[int, float, str]] = []
    for i, s in enumerate(segs):
        if s.start > max_search_seconds:
            break
        hit = is_start_hit(rolling_text(segs, i, 2), start_phrases, fuzzy_threshold)
        if hit:
            start_candidates.append((i, s.start, hit))

    if not start_candidates:
        return None

    # for each start, find last end question after it
    best: Optional[Tuple[float, float, str, int, int, int]] = None  # score + details

    for start_i, start_at, start_hit in start_candidates:
        end_q_idx: Optional[int] = None
        end_hit: Optional[str] = None
        for j in range(start_i, len(segs)):
            if segs[j].start > max_search_seconds:
                break
            h = is_end_hit(rolling_text(segs, j, 2), end_phrases, fuzzy_threshold)
            if h:
                end_q_idx = j
                end_hit = h

        if end_q_idx is None:
            continue

        # extend to include answer
        q_start = segs[end_q_idx].start
        end_at = segs[end_q_idx].end
        saw_number = False

        for k in range(end_q_idx + 1, len(segs)):
            if segs[k].start - q_start > answer_tail_seconds:
                break

            gap = segs[k].start - segs[k - 1].end
            t = segs[k].text.strip()

            if t and NUM_RE.search(t):
                saw_number = True

            if saw_number and gap >= max_gap_s:
                break
            if saw_number and t and QUESTIONISH_RE.search(t) and not NUM_RE.search(t):
                break

            end_at = segs[k].end

        # score window: prefer windows that contain DOB + multiple numbers
        window_text = " ".join(s.text for s in segs if s.start >= start_at and s.end <= end_at)
        num_count = len(re.findall(r"\d+", window_text))
        has_dob = 1 if DOB_RE.search(window_text) else 0
        score = num_count + has_dob * 5

        reason = f"start={start_hit}; end={end_hit}; nums={num_count}; dob={has_dob}; gap={max_gap_s}s"
        if best is None or score > best[0]:
            best = (score, start_at, end_at, reason, start_i, end_q_idx)

    if best is None:
        return None

    _score, start_at, end_at, reason, start_i, end_q_idx = best
    if end_at <= start_at:
        return None
    return start_at, end_at, reason, start_i, end_q_idx


def cut_clip_mp3(ffmpeg: str, preview_wav: Path, start: float, end: float, out_mp3: Path) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.001, end - start)
    cmd = [
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
    _run(cmd)


def write_clip_txt(segs: List[Segment], start: float, end: float, out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    parts = [s.text.strip() for s in segs if s.start >= start and s.end <= end and s.text.strip()]
    out_txt.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")


def make_silence_wav(ffmpeg: str, out_wav: Path, seconds: float, sample_rate: int) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
    _run(cmd)


def mp3_to_wav(ffmpeg: str, mp3: Path, wav: Path, sample_rate: int) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
    _run(cmd)


def wavs_to_mp3_filter_concat(ffmpeg: str, wavs: List[Path], out_mp3: Path) -> None:
    if not wavs:
        raise SystemExit("No WAVs to concat (all clips skipped).")

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identifier", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default="de")

    ap.add_argument("--preview-seconds", type=int, default=300)  # slightly safer than 210
    ap.add_argument("--first-minutes", type=float, default=4.0)
    ap.add_argument("--pre-pad", type=float, default=0.25)
    ap.add_argument("--post-pad", type=float, default=0.25)

    ap.add_argument("--answer-tail-seconds", type=float, default=25.0)
    ap.add_argument("--max-gap-seconds", type=float, default=1.2)

    ap.add_argument("--silence-seconds", type=float, default=0.6)
    ap.add_argument("--sample-rate", type=int, default=16000)

    ap.add_argument("--fuzzy-threshold", type=int, default=85)

    ap.add_argument("--start-phrase", action="append", default=[])
    ap.add_argument("--end-phrase", action="append", default=[])

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    ffmpeg = _which_ffmpeg()
    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    debug_dir = out_dir / "debug"
    clips_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    start_phrases = [normalize_text(x) for x in (args.start_phrase or DEFAULT_START_PHRASES)]
    end_phrases = [normalize_text(x) for x in (args.end_phrase or DEFAULT_END_PHRASES)]

    files = fetch_m4a_files(args.identifier)[: args.limit]
    if not files:
        raise SystemExit("No .m4a files found in this identifier.")

    max_search_seconds = args.first_minutes * 60.0

    manifest: List[Clip] = []
    clip_mp3s: List[Path] = []

    with tempfile.TemporaryDirectory(prefix="numubung_") as td:
        tdir = Path(td)

        for idx, fn in enumerate(files, start=1):
            stem = Path(fn).with_suffix("").name
            url = ia_url(args.identifier, fn)
            print(f"[{idx}/{len(files)}] {fn}")

            preview_wav = tdir / f"{idx:03d}_preview.wav"
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

            picked = pick_window_scored(
                segs=segs,
                start_phrases=start_phrases,
                end_phrases=end_phrases,
                max_search_seconds=max_search_seconds,
                fuzzy_threshold=args.fuzzy_threshold,
                answer_tail_seconds=args.answer_tail_seconds,
                max_gap_s=args.max_gap_seconds,
            )
            if not picked:
                print("  !! window not found (skip)")
                if args.debug:
                    (debug_dir / f"{stem}.json").write_text(
                        json.dumps({"file": fn, "url": url, "segments": [asdict(s) for s in segs]}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                continue

            start, end, reason, start_i, end_q_i = picked
            start = max(0.0, start - args.pre_pad)
            end = min(float(args.preview_seconds), end + args.post_pad)

            out_mp3 = clips_dir / f"{stem}.mp3"
            out_txt = clips_dir / f"{stem}.txt"

            try:
                cut_clip_mp3(ffmpeg, preview_wav, start, end, out_mp3)
                write_clip_txt(segs, start, end, out_txt)
            except subprocess.CalledProcessError:
                print("  !! cut failed")
                continue

            if args.debug:
                (debug_dir / f"{stem}.json").write_text(
                    json.dumps(
                        {
                            "file": fn,
                            "url": url,
                            "picked": {"start": start, "end": end, "reason": reason, "start_i": start_i, "end_q_i": end_q_i},
                            "segments": [asdict(s) for s in segs],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            manifest.append(
                Clip(
                    filename=fn,
                    stem=stem,
                    url=url,
                    start=start,
                    end=end,
                    reason=reason,
                    clip_mp3=str(out_mp3),
                    clip_txt=str(out_txt),
                )
            )
            clip_mp3s.append(out_mp3)

    if not manifest:
        raise SystemExit(
            "No clips produced. Increase --first-minutes / --preview-seconds / --answer-tail-seconds, "
            "or add --start-phrase / --end-phrase."
        )

    silence_wav = out_dir / "_silence.wav"
    make_silence_wav(ffmpeg, silence_wav, args.silence_seconds, args.sample_rate)

    wavs_for_concat: List[Path] = []
    for mp3 in clip_mp3s:
        wav = out_dir / "_tmp" / (mp3.stem + ".wav")
        mp3_to_wav(ffmpeg, mp3, wav, args.sample_rate)
        wavs_for_concat.extend([wav, silence_wav])

    if wavs_for_concat and wavs_for_concat[-1] == silence_wav:
        wavs_for_concat = wavs_for_concat[:-1]

    practice_mp3 = out_dir / "practice_first20.mp3"
    print(f"Concatenating {len(wavs_for_concat)} wav parts -> {practice_mp3}")
    wavs_to_mp3_filter_concat(ffmpeg, wavs_for_concat, practice_mp3)

    (out_dir / "manifest.json").write_text(
        json.dumps([asdict(c) for c in manifest], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"done. clips={len(manifest)} -> {practice_mp3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
