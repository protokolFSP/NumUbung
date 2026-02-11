# file: tools/build_practice_ia.py
"""
Build a German number-practice audio from an Internet Archive item (first N .m4a files).

Outputs:
- outputs/clips/<stem>.mp3
- outputs/clips/<stem>.txt
- outputs/practice_first20.mp3
- outputs/manifest.json

Matching:
- Start: any of --start-regex patterns (defaults include common variants)
- End question: last match of any --end-regex pattern within first_minutes
- Answer included:
  after end question, extend window to include number-containing answer segments,
  and stop early if a new question begins after capturing numbers.

Why this version:
- Uses ffmpeg filter_complex concat (no concat_list.txt parsing issues).
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

IA_METADATA_URL = "https://archive.org/metadata/{identifier}"
IA_DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

DEFAULT_START_REGEXES = [
    r"\bwie\s+alt\s+sind\s+sie\b",
    r"\bwie\s+alt\s+bist\s+du\b",
    r"\bihr\s+alter\b",
    r"\bwelches\s+alter\b",
    r"\bgeburtsdatum\b",
    r"\bgeboren\b",
]

DEFAULT_END_REGEXES = [
    r"\bwie\s+viel\s+wiegen\s+sie\b",
    r"\bwie\s+viel\s+wiegen\s+du\b",
    r"\bwas\s+wiegen\s+sie\b",
    r"\bgewicht\b",
    r"\bwie\s+gro[ßs]\s+sind\s+sie\b",
    r"\bwie\s+gro[ßs]\s+bist\s+du\b",
    r"\bihre\s+gr(o|ö)[ßs]e\b",
    r"\bk(o|ö)rpergr(o|ö)[ßs]e\b",
]

NUM_RE = re.compile(
    r"\d+|(\bnull\b|\bein\b|\beins\b|\bzwei\b|\bdrei\b|\bvier\b|\bfünf\b|\bfuenf\b|\bsechs\b|"
    r"\bsieben\b|\bacht\b|\bneun\b|\bzehn\b|\belf\b|\bzwölf\b|\bzwoelf\b)",
    re.IGNORECASE,
)
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


def compile_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def find_first_match_index(segs: List[Segment], patterns: List[re.Pattern], max_seconds: float) -> Optional[int]:
    for i, s in enumerate(segs):
        if s.start > max_seconds:
            break
        txt = rolling_text(segs, i, 2)
        if any(p.search(txt) for p in patterns):
            return i
    return None


def find_last_match_index(segs: List[Segment], start_i: int, patterns: List[re.Pattern], max_seconds: float) -> Optional[int]:
    last: Optional[int] = None
    for j in range(start_i, len(segs)):
        if segs[j].start > max_seconds:
            break
        txt = rolling_text(segs, j, 2)
        if any(p.search(txt) for p in patterns):
            last = j
    return last


def extend_end_to_include_answer(segs: List[Segment], end_q_idx: int, max_extra_seconds: float) -> float:
    q_start = segs[end_q_idx].start
    end_at = segs[end_q_idx].end
    saw_number = False

    for k in range(end_q_idx + 1, len(segs)):
        if segs[k].start - q_start > max_extra_seconds:
            break

        t = segs[k].text.strip()
        if t:
            if NUM_RE.search(t):
                saw_number = True
            if saw_number and QUESTIONISH_RE.search(t) and not NUM_RE.search(t):
                break

        end_at = segs[k].end

    return end_at


def pick_window(
    segs: List[Segment],
    start_patterns: List[re.Pattern],
    end_patterns: List[re.Pattern],
    max_search_seconds: float,
    answer_tail_seconds: float,
) -> Optional[Tuple[float, float, str]]:
    start_i = find_first_match_index(segs, start_patterns, max_search_seconds)
    if start_i is None:
        return None

    end_q_idx = find_last_match_index(segs, start_i, end_patterns, max_search_seconds)
    if end_q_idx is None:
        return None

    start_at = segs[start_i].start
    end_at = extend_end_to_include_answer(segs, end_q_idx, max_extra_seconds=answer_tail_seconds)

    if end_at <= start_at:
        return None

    return start_at, end_at, f"start_idx={start_i}; end_q_idx={end_q_idx}; tail={answer_tail_seconds}s"


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

    cmd: List[str] = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    for w in wavs:
        cmd += ["-i", str(w)]

    if len(wavs) == 1:
        cmd += ["-c:a", "libmp3lame", "-b:a", "96k", str(out_mp3)]
        _run(cmd)
        return

    inputs = "".join([f"[{i}:a]" for i in range(len(wavs))])
    filt = f"{inputs}concat=n={len(wavs)}:v=0:a=1[a]"
    cmd += [
        "-filter_complex",
        filt,
        "-map",
        "[a]",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "96k",
        str(out_mp3),
    ]
    _run(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identifier", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default="de")

    ap.add_argument("--preview-seconds", type=int, default=210)
    ap.add_argument("--first-minutes", type=float, default=3.0)
    ap.add_argument("--pre-pad", type=float, default=0.25)
    ap.add_argument("--post-pad", type=float, default=0.25)

    ap.add_argument("--answer-tail-seconds", type=float, default=15.0)
    ap.add_argument("--silence-seconds", type=float, default=0.6)
    ap.add_argument("--sample-rate", type=int, default=16000)

    ap.add_argument("--start-regex", action="append", default=[])
    ap.add_argument("--end-regex", action="append", default=[])

    args = ap.parse_args()

    ffmpeg = _which_ffmpeg()
    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    start_patterns = compile_patterns(args.start_regex or DEFAULT_START_REGEXES)
    end_patterns = compile_patterns(args.end_regex or DEFAULT_END_REGEXES)

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

            win = pick_window(
                segs=segs,
                start_patterns=start_patterns,
                end_patterns=end_patterns,
                max_search_seconds=max_search_seconds,
                answer_tail_seconds=args.answer_tail_seconds,
            )
            if not win:
                print("  !! window not found (skip)")
                continue

            start, end, reason = win
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
            "No clips produced. Increase --first-minutes (e.g. 4) or --answer-tail-seconds (e.g. 25), "
            "or add --start-regex/--end-regex."
        )

    # Build final practice mp3
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
