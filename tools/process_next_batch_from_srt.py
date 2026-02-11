# file: tools/process_next_batch_from_srt.py
"""
Create number-practice clips using EXISTING SRT timestamps (no Whisper), with robust window picking.

Why this version is better:
- Handles variable order: weight before height, combined questions, detours/complaints.
- Uses a small state-machine (age/dob/height/weight) instead of "end question + fixed tail".
- Stops when a non-demographic question starts AFTER collecting at least one of height/weight.

Outputs:
- outputs/clips/<stem>.mp3
- outputs/clips/<stem>.txt
- outputs/ubung/batch_###.mp3
- outputs/state.json
- outputs/failures.json
- outputs/debug/<stem>.json (optional)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
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

START_PHRASES = [
    "wie alt sind sie",
    "wie alt bist du",
    "jahre alt",
    "geburtsdatum",
    "wann sind sie geboren",
    "wann bist du geboren",
    "wann genau sind sie geboren",
    "geboren am",
    "wie alt sind sie und wann sind sie geboren",
]

HEIGHT_Q_PHRASES = [
    "wie gross sind sie",
    "wie groß sind sie",
    "wie gross und wie schwer",
    "wie groß und wie schwer",
    "koerpergroesse",
    "körpergröße",
]

WEIGHT_Q_PHRASES = [
    "wie viel wiegen sie",
    "wie viel wiegen sie zurzeit",
    "wie viel wiegen sie momentan",
    "wie schwer sind sie",
    "ihr gewicht",
    "gewicht",
    "kennen sie ihr gewicht",
    "aktuelles gewicht",
    "aktuelle gewicht",
    "kennen sie ihre aktuellen gewicht",
]

# Very common "complaint / next section" starters (to stop at the right time)
NON_DEMOG_Q_RE = re.compile(
    r"^\s*(was|warum|wobei|woran|wo|seit wann|haben sie (schon|noch)|"
    r"wo tut|wo genau|wie stark|welche beschwerden|was führt sie|was bringt sie|"
    r"was betrifft|was kann ich|was ist der grund)\b",
    re.IGNORECASE,
)

QUESTIONISH_RE = re.compile(
    r"^\s*(wie|wann|wo|was|welche|welcher|wieviel|haben|nehmen|sind|ist|können|koennen|dürfen|duerfen)\b",
    re.IGNORECASE,
)

# Signals for answers
KG_RE = re.compile(r"\b(\d{2,3})\s*(kg|kilo)\b", re.IGNORECASE)
M_RE = re.compile(r"\b(\d([,.]\d{1,2})?)\s*(m|meter)\b", re.IGNORECASE)  # 1,74 m
METER_SPLIT_RE = re.compile(r"\b1\s*meter\s*(\d{2})\b", re.IGNORECASE)  # "1 Meter 65"
AGE_RE = re.compile(r"\b(\d{1,3})\s*(jahre)?\b", re.IGNORECASE)
DOB_DOT_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b")
DOB_MONTH_RE = re.compile(
    r"\b(\d{1,2})\.\s*(januar|februar|märz|maerz|april|mai|juni|juli|august|"
    r"september|oktober|november|dezember)\s*(\d{4})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SrtEntry:
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


def parse_ts(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: Path) -> List[SrtEntry]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", txt.strip(), flags=re.MULTILINE)
    out: List[SrtEntry] = []
    for b in blocks:
        lines = [ln.rstrip() for ln in b.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        time_line = lines[1] if "-->" in lines[1] else lines[0]
        m = re.search(r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)", time_line)
        if not m:
            continue
        st = parse_ts(m.group(1))
        en = parse_ts(m.group(2))
        text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
        text = " ".join(text_lines).strip()
        if text:
            out.append(SrtEntry(st, en, text))
    return out


def fuzzy_hit(text: str, phrases: List[str], threshold: int) -> Optional[str]:
    t = normalize_text(text)
    for p in phrases:
        if p in t:
            return f"substr:{p}"
    if fuzz is None:
        return None
    best_p, best_s = "", -1
    for p in phrases:
        sc = fuzz.partial_ratio(t, p)
        if sc > best_s:
            best_s, best_p = sc, p
    if best_s >= threshold:
        return f"fuzzy:{best_p}:{int(best_s)}"
    return None


def _is_age(text: str) -> bool:
    t = normalize_text(text)
    if "jahre alt" in t:
        return True
    m = AGE_RE.search(t)
    if not m:
        return False
    v = int(m.group(1))
    return 1 <= v <= 110


def _is_dob(text: str) -> bool:
    t = normalize_text(text)
    return bool(DOB_DOT_RE.search(t) or DOB_MONTH_RE.search(t))


def _is_height_answer(text: str) -> bool:
    t = normalize_text(text)
    if M_RE.search(t):
        return True
    if METER_SPLIT_RE.search(t):
        return True
    # allow "1,80" without unit if line contains "gross" / "groß"
    if ("gross" in t or "gro" in t) and re.search(r"\b1[,.]\d{2}\b|\b1\d{2}\b", t):
        return True
    return False


def _is_weight_answer(text: str) -> bool:
    t = normalize_text(text)
    if KG_RE.search(t):
        return True
    if ("kilo" in t or "kg" in t or "gewicht" in t) and re.search(r"\b\d{2,3}\b", t):
        return True
    return False


def _is_height_q(text: str, thr: int) -> bool:
    return fuzzy_hit(text, HEIGHT_Q_PHRASES, thr) is not None


def _is_weight_q(text: str, thr: int) -> bool:
    return fuzzy_hit(text, WEIGHT_Q_PHRASES, thr) is not None


def pick_demog_window(
    entries: List[SrtEntry],
    first_minutes: float,
    tail_seconds: float,
    fuzzy_threshold: int = 84,
) -> Optional[Tuple[float, float, str]]:
    max_search = first_minutes * 60.0

    # Find the earliest plausible start
    start_i: Optional[int] = None
    start_hit: str = ""
    for i, e in enumerate(entries):
        if e.start > max_search:
            break
        hit = fuzzy_hit(e.text, START_PHRASES, fuzzy_threshold)
        if hit:
            start_i = i
            start_hit = hit
            break
    if start_i is None:
        return None

    found_age = False
    found_dob = False
    found_h = False
    found_w = False

    last_demog_time = entries[start_i].end
    end_time = entries[start_i].end

    for j in range(start_i, len(entries)):
        e = entries[j]
        if e.start > max_search:
            break

        t = e.text.strip()
        nt = normalize_text(t)

        # Track questions
        hq = _is_height_q(t, fuzzy_threshold)
        wq = _is_weight_q(t, fuzzy_threshold)

        # Track answers/signals
        if _is_age(t):
            found_age = True
            last_demog_time = e.end
        if _is_dob(t):
            found_dob = True
            last_demog_time = e.end
        if _is_height_answer(t):
            found_h = True
            last_demog_time = e.end
        if _is_weight_answer(t):
            found_w = True
            last_demog_time = e.end

        if hq or wq:
            last_demog_time = e.end

        # Decide when to stop:
        # If we already saw at least one of (height/weight) and a new non-demographic question starts, stop BEFORE it.
        if (found_h or found_w) and NON_DEMOG_Q_RE.search(t):
            break

        # If we already saw at least one of (height/weight) and we drift too far from last demog mention, stop.
        if (found_h or found_w) and (e.start - last_demog_time) > tail_seconds:
            break

        end_time = e.end

    # Score window; accept if we have at least age/dob + (height or weight), OR height+weight
    score = 0
    score += 2 if found_age else 0
    score += 2 if found_dob else 0
    score += 2 if found_h else 0
    score += 2 if found_w else 0

    if score < 4:
        return None

    reason = f"start={start_hit}; age={found_age}; dob={found_dob}; h={found_h}; w={found_w}"
    return entries[start_i].start, end_time, reason


def cut_clip_from_url(ffmpeg: str, url: str, start: float, end: float, out_mp3: Path) -> None:
    dur = max(0.05, end - start)
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
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
            url,
            "-vn",
            "-ac",
            "1",
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


def find_srt_exact(srt_dir: Path, stem: str) -> Optional[Path]:
    p = srt_dir / f"{stem}.srt"
    if p.exists():
        return p
    p2 = srt_dir / f"{stem}.SRT"
    if p2.exists():
        return p2
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identifier", required=True)
    ap.add_argument("--srt-dir", required=True)
    ap.add_argument("--out-dir", default="outputs")

    ap.add_argument("--limit-total", type=int, default=252)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=160)
    ap.add_argument("--max-failures", type=int, default=3)

    ap.add_argument("--first-minutes", type=float, default=6.0)
    ap.add_argument("--answer-tail-seconds", type=float, default=25.0)
    ap.add_argument("--pre-pad", type=float, default=0.25)
    ap.add_argument("--post-pad", type=float, default=0.25)

    ap.add_argument("--silence-seconds", type=float, default=0.6)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    ffmpeg = _which_ffmpeg()
    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    ubung_dir = out_dir / "ubung"
    debug_dir = out_dir / "debug"
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
    srt_dir = Path(args.srt_dir)

    pending: List[str] = []
    for fn in all_files:
        stem = Path(fn).with_suffix("").name
        if stem in done_stems:
            continue
        if is_perm_skipped(stem):
            continue
        pending.append(fn)

    state = {
        "identifier": args.identifier,
        "limit_total": args.limit_total,
        "done": len(done_stems),
        "pending": len(pending),
        "target_produced": args.batch_size,
        "max_attempts": args.max_attempts,
        "mode": "srt_exact_state_machine",
    }
    _save_json(out_dir / "state.json", state)

    if not pending:
        print("Nothing pending.")
        return 0

    produced: List[Path] = []
    tried = 0

    with tempfile.TemporaryDirectory(prefix="numubung_srt_") as _td:
        for fn in pending:
            if len(produced) >= args.batch_size:
                break
            if tried >= args.max_attempts:
                break
            tried += 1

            stem = Path(fn).with_suffix("").name
            url = ia_url(args.identifier, fn)

            srt_path = find_srt_exact(srt_dir, stem)
            if not srt_path:
                ent = failures.get(stem, {"attempts": 0, "permanent": False})
                ent["attempts"] = int(ent.get("attempts", 0)) + 1
                ent["last_reason"] = "srt_missing"
                if ent["attempts"] >= args.max_failures:
                    ent["permanent"] = True
                failures[stem] = ent
                _save_json(failures_path, failures)
                print(f"[{tried}] {fn}: !! SRT missing (skip)")
                continue

            entries = parse_srt(srt_path)
            picked = pick_demog_window(
                entries=entries,
                first_minutes=args.first_minutes,
                tail_seconds=args.answer_tail_seconds,
            )
            if not picked:
                ent = failures.get(stem, {"attempts": 0, "permanent": False})
                ent["attempts"] = int(ent.get("attempts", 0)) + 1
                ent["last_reason"] = "window_not_found_srt"
                if ent["attempts"] >= args.max_failures:
                    ent["permanent"] = True
                failures[stem] = ent
                _save_json(failures_path, failures)
                if args.debug:
                    _save_json(debug_dir / f"{stem}_fail.json", {"file": fn, "srt": str(srt_path)})
                print(f"[{tried}] {fn}: !! window not found (skip)")
                continue

            st, en, reason = picked
            st = max(0.0, st - args.pre_pad)
            en = en + args.post_pad

            out_mp3 = clips_dir / f"{stem}.mp3"
            out_txt = clips_dir / f"{stem}.txt"

            print(f"[{tried}] produce {len(produced)+1}/{args.batch_size}: {fn} ({st:.2f}-{en:.2f}) {reason}")
            try:
                cut_clip_from_url(ffmpeg, url, st, en, out_mp3)
            except subprocess.CalledProcessError:
                ent = failures.get(stem, {"attempts": 0, "permanent": False})
                ent["attempts"] = int(ent.get("attempts", 0)) + 1
                ent["last_reason"] = "ffmpeg_cut_failed"
                if ent["attempts"] >= args.max_failures:
                    ent["permanent"] = True
                failures[stem] = ent
                _save_json(failures_path, failures)
                print("  !! ffmpeg cut failed")
                continue

            parts = [e.text.strip() for e in entries if e.start >= st and e.end <= en and e.text.strip()]
            out_txt.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")

            if args.debug:
                _save_json(debug_dir / f"{stem}.json", {"file": fn, "srt": str(srt_path), "picked": {"start": st, "end": en, "reason": reason}})

            produced.append(out_mp3)

    if not produced:
        state["this_run_produced"] = 0
        state["this_run_tried"] = tried
        _save_json(out_dir / "state.json", state)
        _save_json(failures_path, failures)
        print("No clips produced.")
        return 0

    batch_idx = next_batch_index(ubung_dir)
    batch_mp3 = ubung_dir / f"batch_{batch_idx:03d}.mp3"

    silence_wav = out_dir / "_silence.wav"
    make_silence_wav(ffmpeg, silence_wav, args.silence_seconds, args.sample_rate)

    wavs: List[Path] = []
    for mp3 in produced:
        wav = out_dir / "_tmp" / f"{mp3.stem}.wav"
        mp3_to_wav(ffmpeg, mp3, wav, args.sample_rate)
        wavs.extend([wav, silence_wav])
    wavs = wavs[:-1]

    print(f"Creating {batch_mp3} (clips={len(produced)})")
    wavs_to_mp3_filter_concat(ffmpeg, wavs, batch_mp3)

    state["this_run_produced"] = len(produced)
    state["this_run_tried"] = tried
    state["last_batch_mp3"] = str(batch_mp3)
    _save_json(out_dir / "state.json", state)
    _save_json(failures_path, failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
