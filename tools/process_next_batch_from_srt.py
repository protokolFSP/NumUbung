# file: tools/process_next_batch_from_srt.py
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
]

HEIGHT_PHRASES = [
    "wie gross sind sie",
    "wie groß sind sie",
    "koerpergroesse",
    "körpergröße",
    "wie gross und wie schwer",
    "wie groß und wie schwer",
]

WEIGHT_PHRASES = [
    "wie viel wiegen sie",
    "wie viel wiegen sie zurzeit",
    "wie viel wiegen sie momentan",
    "wie schwer sind sie",
    "ihr gewicht",
    "gewicht",
    "kennen sie ihr gewicht",
    "aktuelles gewicht",
    "aktuelle gewicht",
]

TRANSITION_PHRASES = [
    "was fuehrt sie zu uns",
    "was fuehrt sie heute zu uns",
    "was fuehrt sie her",
    "was bringt sie zu uns",
    "was kann ich fuer sie tun",
    "womit kann ich ihnen helfen",
    "was fuer beschwerden haben sie",
    "welche beschwerden haben sie",
    "was sind ihre beschwerden",
]

NON_DEMOG_Q_RE = re.compile(
    r"\b(was\s+f(ü|ue)hrt\s+sie(\s+zu\s+uns|\s+heute|\s+her)?)\b|"
    r"\b(was\s+kann\s+ich\s+f(ü|ue)r\s+sie\s+tun)\b|"
    r"\b(was\s+bringt\s+sie\s+zu\s+uns)\b|"
    r"\b(beschwerden)\b",
    re.IGNORECASE,
)

KG_RE = re.compile(r"\b(\d{2,3})\s*(kg|kilo)\b", re.IGNORECASE)
M_RE = re.compile(r"\b(\d([,.]\d{1,2})?)\s*(m|meter)\b", re.IGNORECASE)
METER_SPLIT_RE = re.compile(r"\b1\s*meter\s*(\d{2})\b", re.IGNORECASE)
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


def fuzzy_any(text: str, phrases: List[str], threshold: int) -> bool:
    t = normalize_text(text)
    for p in phrases:
        if p in t:
            return True
    if fuzz is None:
        return False
    best = max(fuzz.partial_ratio(t, p) for p in phrases)
    return best >= threshold


def is_transition(text: str, threshold: int) -> bool:
    return bool(NON_DEMOG_Q_RE.search(text)) or fuzzy_any(text, TRANSITION_PHRASES, threshold)


def is_age(text: str) -> bool:
    t = normalize_text(text)
    if "jahre alt" in t:
        return True
    m = AGE_RE.search(t)
    return bool(m and 1 <= int(m.group(1)) <= 110)


def is_dob(text: str) -> bool:
    t = normalize_text(text)
    return bool(DOB_DOT_RE.search(t) or DOB_MONTH_RE.search(t))


def is_height(text: str, threshold: int) -> bool:
    t = normalize_text(text)
    if M_RE.search(t) or METER_SPLIT_RE.search(t):
        return True
    return fuzzy_any(text, HEIGHT_PHRASES, threshold)


def is_weight(text: str, threshold: int) -> bool:
    t = normalize_text(text)
    if KG_RE.search(t):
        return True
    return fuzzy_any(text, WEIGHT_PHRASES, threshold)


def pick_window_strict_first_minutes(
    entries: List[SrtEntry],
    first_minutes: float,
    max_clip_seconds: float,
    fuzzy_threshold: int,
    transition_fuzzy_threshold: int,
    demog_tail_seconds: float = 0.8,
) -> Optional[Tuple[float, float, str]]:
    max_search = first_minutes * 60.0

    start_i: Optional[int] = None
    for i, e in enumerate(entries):
        if e.start > max_search:
            break
        if fuzzy_any(e.text, START_PHRASES, fuzzy_threshold):
            start_i = i
            break
    if start_i is None:
        return None

    start_at = entries[start_i].start
    last_demog = entries[start_i].end
    end_at = entries[start_i].end

    found_age = False
    found_dob = False
    found_h_or_w = False

    for j in range(start_i, len(entries)):
        e = entries[j]
        if e.start > max_search:
            break

        # Transition => cut BEFORE transition line, still clamp to <= max_search
        if found_h_or_w and is_transition(e.text, transition_fuzzy_threshold):
            end_at = min(max_search, max(last_demog + 0.4, e.start - 0.05))
            break

        if is_age(e.text):
            found_age = True
            last_demog = e.end
        if is_dob(e.text):
            found_dob = True
            last_demog = e.end

        if is_height(e.text, fuzzy_threshold) or is_weight(e.text, fuzzy_threshold):
            found_h_or_w = True
            last_demog = e.end

        end_at = e.end

        # hard max clip length
        if (end_at - start_at) >= max_clip_seconds:
            end_at = start_at + max_clip_seconds
            break

    # Always clamp inside first_minutes window and near last demog
    end_at = min(max_search, min(end_at, last_demog + demog_tail_seconds))

    if not ((found_age or found_dob) and found_h_or_w):
        return None
    if end_at <= start_at:
        return None

    return start_at, end_at, f"age={found_age};dob={found_dob};h_or_w={found_h_or_w};max_search={max_search}"


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

    ap.add_argument("--first-minutes", type=float, default=3.0)
    ap.add_argument("--max-clip-seconds", type=float, default=70.0)

    ap.add_argument("--fuzzy-threshold", type=int, default=84)
    ap.add_argument("--transition-fuzzy-threshold", type=int, default=82)

    ap.add_argument("--pre-pad", type=float, default=0.10)
    ap.add_argument("--post-pad", type=float, default=0.10)

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
        if stem in done_stems or is_perm_skipped(stem):
            continue
        pending.append(fn)

    _save_json(
        out_dir / "state.json",
        {
            "identifier": args.identifier,
            "limit_total": args.limit_total,
            "done": len(done_stems),
            "pending": len(pending),
            "target_produced": args.batch_size,
            "max_attempts": args.max_attempts,
            "mode": "srt_strict_first_minutes",
            "first_minutes": args.first_minutes,
            "max_clip_seconds": args.max_clip_seconds,
        },
    )

    if not pending:
        print("Nothing pending.")
        return 0

    produced: List[Path] = []
    tried = 0

    with tempfile.TemporaryDirectory(prefix="numubung_srt_") as _td:
        for fn in pending:
            if len(produced) >= args.batch_size or tried >= args.max_attempts:
                break
            tried += 1

            stem = Path(fn).with_suffix("").name
            srt_path = find_srt_exact(srt_dir, stem)
            if not srt_path:
                ent = failures.get(stem, {"attempts": 0, "permanent": False})
                ent["attempts"] = int(ent.get("attempts", 0)) + 1
                ent["last_reason"] = "srt_missing"
                if ent["attempts"] >= args.max_failures:
                    ent["permanent"] = True
                failures[stem] = ent
                _save_json(failures_path, failures)
                continue

            entries = parse_srt(srt_path)
            picked = pick_window_strict_first_minutes(
                entries=entries,
                first_minutes=args.first_minutes,
                max_clip_seconds=args.max_clip_seconds,
                fuzzy_threshold=args.fuzzy_threshold,
                transition_fuzzy_threshold=args.transition_fuzzy_threshold,
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
                    _save_json(debug_dir / f"{stem}_fail.json", {"srt": str(srt_path)})
                continue

            st, en, reason = picked
            st = max(0.0, st - args.pre_pad)
            en = en + args.post_pad

            url = ia_url(args.identifier, fn)
            out_mp3 = clips_dir / f"{stem}.mp3"
            out_txt = clips_dir / f"{stem}.txt"

            print(f"[{tried}] produce {len(produced)+1}/{args.batch_size}: {stem} ({st:.2f}-{en:.2f}) {reason}")
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
                continue

            parts = [e.text.strip() for e in entries if e.start >= st and e.end <= en and e.text.strip()]
            out_txt.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")

            if args.debug:
                _save_json(debug_dir / f"{stem}.json", {"srt": str(srt_path), "picked": {"start": st, "end": en, "reason": reason}})

            produced.append(out_mp3)

    if not produced:
        _save_json(out_dir / "failures.json", failures)
        print("No clips produced.")
        return 0

    # build batch mp3
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
    wavs_to_mp3_filter_concat(ffmpeg, wavs, batch_mp3)

    _save_json(out_dir / "failures.json", failures)
    state = _load_json(out_dir / "state.json", {})
    state.update({"this_run_produced": len(produced), "this_run_tried": tried, "last_batch_mp3": str(batch_mp3)})
    _save_json(out_dir / "state.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
