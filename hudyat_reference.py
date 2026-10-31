"""
hudyat_reference.py - Reference videos for the FSL expert validation.

Before the signers record their 15 takes, whoever can sign records ONE reference
take of every sign (all 108) with  python record_tool.py --reference .
A sign can also get a second version (V in the recorder): another way of
signing it. The FSL validators then check these videos in a Google Form; for a
sign with two versions they tick which version is correct.

This module holds everything that does not need the camera or the window, so
it can be tested on its own:

  * the sign order and numbering (reference_signs),
  * file names (clean_filename) and paths,
  * the export of a kept take: trimmed H.264 .mp4 + looping .gif (export_take),
  * archiving an older version before a sign is re-recorded (archive_old),
  * manifest.csv, form_list.md, build_validation_form.gs and FORM_SETUP.md.

Everything is written under references/. Nothing here reads or writes dataset/.

Run  python hudyat_reference.py  to (re)write the form material without the camera.

    references/validation/<NN Category>/<NN> <Sign Label>.mp4   (and .gif)
    references/validation/<NN Category>/<NN> <Sign Label> (2).mp4   version 2, if recorded
    references/validation/manifest.csv, form_list.md, build_validation_form.gs, FORM_SETUP.md
    references/_old/<NN Category>/<NN> <Sign Label> <timestamp>.mp4   earlier versions, never deleted
    references/_tmp/            the raw take while it waits for Keep / Redo
    references/reference_log.txt, reference_state.json
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hudyat_common as hc

# =============================================================================
# SETTINGS
# =============================================================================
REF_DIR = hc.REFERENCES_DIR            # tests point this at a temporary folder
REF_SECONDS = 4.0                      # length of a reference take (time-based); --ref-seconds
EXPECTED_TOTAL = 108                   # stop if signs.csv does not give exactly this many signs
TRIM_PAD_S = 0.3                       # kept before / after the sign window (hudyat_common.sign_window)
MP4_SIZE = (1280, 720)
MP4_CRF = 20                           # H.264 quality (lower = better, bigger)
GIF_MAX_BYTES = 2_000_000              # every .gif stays under 2 MB (Google Forms image limit)
# GIF (width px, fps) tried in order until the file is under GIF_MAX_BYTES.
GIF_SETTINGS = [(480, 12), (480, 10), (400, 10), (360, 8), (320, 8), (240, 6)]
GIF_COLORS = 128

# (folder name, signs.csv categories that go in it), in form order.
CATEGORY_FOLDERS: list[tuple[str, tuple[str, ...]]] = [
    ("01 Alphabet", ("alphabet", "alphabet_static", "alphabet_static_test", "alphabet_dynamic")),
    ("02 Numbers", ("numbers",)),
    ("03 Greetings", ("greetings",)),
    ("04 Colors", ("colors",)),
    ("05 Shapes", ("shapes",)),
    ("06 Family", ("family",)),
    ("07 Calendar", ("calendar",)),
    ("08 WH-Questions", ("wh_questions",)),
    ("09 Word Concepts", ("word_concepts",)),
]
ALPHABET_FOLDER = "01 Alphabet"        # merged letters are sorted A to Z
SKIP = ("none",)                       # the "none" row (category or sign_id) is not a sign

MANIFEST_FIELDS = [
    "no", "category", "folder", "sign_id", "sign_label", "file_stem", "mp4", "gif",
    "seconds", "gif_kb", "hand_detect_rate", "hand", "recorded_at",
    "gif_settings", "version",
]
VERSIONS = (1, 2)                      # version 1 = the reference video; 2 = another way of signing it
FORM_TITLE = "HUDYAT – FSL Gesture Validation Form"
CONTACT_LINE = "[Contact: name, Messenger/phone]"
WINDOWS_FORBIDDEN = set('<>:"/\\|?*')


# =============================================================================
# SIGN LIST
# =============================================================================
@dataclass(frozen=True)
class RefSign:
    """One sign in the reference list."""
    index: int          # 0..109 in the whole list
    no: int             # 1.. inside its folder
    folder: str         # e.g. "03 Greetings"
    sign: hc.Sign

    @property
    def file_stem(self) -> str:
        """e.g. "06 I'm Fine" """
        return f"{self.no:02d} {clean_filename(self.sign.sign_label)}"

    @property
    def title(self) -> str:
        """e.g. "06 I'm Fine" with the exact label (used in the form)."""
        return f"{self.no:02d} {self.sign.sign_label}"

    @property
    def category_title(self) -> str:
        return folder_title(self.folder)


def folder_title(folder: str) -> str:
    """"03 Greetings" -> "Greetings" """
    return folder.split(" ", 1)[1]


def clean_filename(label: str) -> str:
    """The sign label as a file name: only characters Windows forbids are replaced.

    Capitals, spaces and apostrophes stay ("I'm Fine" stays "I'm Fine").
    Windows also refuses names ending in a space or a dot.
    """
    name = "".join("_" if c in WINDOWS_FORBIDDEN or ord(c) < 32 else c for c in label)
    return name.rstrip(" .") or "_"


def reference_signs(signs: list[hc.Sign]) -> list[RefSign]:
    """All signs in form order, numbered within their folder.

    Raises ValueError if a category is unknown or the total is not EXPECTED_TOTAL.
    """
    folder_of = {cat: folder for folder, cats in CATEGORY_FOLDERS for cat in cats}
    buckets: dict[str, list[hc.Sign]] = {folder: [] for folder, _ in CATEGORY_FOLDERS}
    unknown: list[str] = []
    for s in signs:
        if s.category in SKIP or s.sign_id in SKIP:
            continue
        if s.category not in folder_of:
            unknown.append(f"{s.category}/{s.sign_id}")
            continue
        buckets[folder_of[s.category]].append(s)
    if unknown:
        raise ValueError(f"signs.csv has signs in unknown categories: {', '.join(unknown)}")
    buckets[ALPHABET_FOLDER].sort(key=lambda s: s.sign_label.upper())
    out: list[RefSign] = []
    for folder, _ in CATEGORY_FOLDERS:
        for no, s in enumerate(buckets[folder], start=1):
            out.append(RefSign(len(out), no, folder, s))
    if len(out) != EXPECTED_TOTAL:
        counts = ", ".join(f"{folder_title(f)} {len(b)}" for f, b in buckets.items())
        raise ValueError(f"signs.csv gives {len(out)} reference signs, expected {EXPECTED_TOTAL} ({counts}).")
    return out


# =============================================================================
# PATHS
# =============================================================================
def base_dir(base: Path | None = None) -> Path:
    return base or REF_DIR


def validation_dir(base: Path | None = None) -> Path:
    return base_dir(base) / "validation"


def old_dir(base: Path | None = None) -> Path:
    return base_dir(base) / "_old"


def tmp_dir(base: Path | None = None) -> Path:
    return base_dir(base) / "_tmp"


def version_stem(r: RefSign, version: int = 1) -> str:
    """ "06 I'm Fine" for version 1, "06 I'm Fine (2)" for version 2."""
    return r.file_stem if version == 1 else f"{r.file_stem} ({version})"


def mp4_path(r: RefSign, base: Path | None = None, version: int = 1) -> Path:
    return validation_dir(base) / r.folder / f"{version_stem(r, version)}.mp4"


def gif_path(r: RefSign, base: Path | None = None, version: int = 1) -> Path:
    return validation_dir(base) / r.folder / f"{version_stem(r, version)}.gif"


def log(message: str, base: Path | None = None) -> None:
    """Append a line to references/reference_log.txt."""
    path = base_dir(base) / "reference_log.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")


def load_state(base: Path | None = None) -> dict:
    """Remembered hand and last sign (references/reference_state.json)."""
    try:
        with open(base_dir(base) / "reference_state.json", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state: dict, base: Path | None = None) -> None:
    path = base_dir(base) / "reference_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# =============================================================================
# RE-RECORDING: move the old version away (never deleted)
# =============================================================================
def archive_old(r: RefSign, base: Path | None = None, now: datetime | None = None,
                version: int = 1) -> list[Path]:
    """Move this version's existing .mp4 / .gif to references/_old/<folder>/ with a
    timestamp in the name. Returns the new paths. Nothing is ever deleted."""
    stamp = f"{now or datetime.now():%Y-%m-%d_%H%M%S}"
    moved: list[Path] = []
    for src in (mp4_path(r, base, version), gif_path(r, base, version)):
        if not src.exists():
            continue
        dest_dir = old_dir(base) / r.folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{src.stem} {stamp}{src.suffix}"
        n = 2
        while dest.exists():                                  # never overwrite an archived file
            dest = dest_dir / f"{src.stem} {stamp} ({n}){src.suffix}"
            n += 1
        os.replace(src, dest)
        moved.append(dest)
    return moved


# =============================================================================
# EXPORT: trimmed H.264 .mp4 + looping .gif
# =============================================================================
def ffmpeg_exe() -> str:
    """The ffmpeg binary bundled with imageio-ffmpeg (in the venv)."""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        raise RuntimeError(f"ffmpeg failed: {lines[-1] if lines else result.returncode}")


def trim_range(window_start: int, window_end: int, fps: float, n_frames: int,
               pad: float = TRIM_PAD_S) -> tuple[float, float]:
    """(start_s, end_s) of the clip to keep: the sign window (frames, end
    inclusive) plus `pad` seconds on each side, inside the take."""
    start = max(0.0, window_start / fps - pad)
    end = min(n_frames / fps, (window_end + 1) / fps + pad)
    return start, end


def encode_mp4(src: Path, dest: Path, start_s: float, end_s: float) -> None:
    """H.264 (yuv420p), 1280x720, no audio, trimmed to start_s..end_s.

    The frames are NOT flipped: the video shows the signer as the camera sees
    them, like watching a person face to face. A camera that is not 16:9 is
    letterboxed rather than stretched.
    """
    w, h = MP4_SIZE
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    _run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
          "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-vf", vf,
          "-c:v", "libx264", "-preset", "medium", "-crf", str(MP4_CRF), "-pix_fmt", "yuv420p",
          "-an", "-movflags", "+faststart", str(dest)])


def encode_gif(src: Path, dest: Path, width: int, fps: int) -> None:
    """Looping GIF of the (already trimmed) clip, with its own colour palette."""
    vf = (f"fps={fps},scale={width}:-2:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors={GIF_COLORS}:stats_mode=diff[p];"
          f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    _run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
          "-vf", vf, "-loop", "0", str(dest)])


def make_gif(src: Path, dest: Path, max_bytes: int = GIF_MAX_BYTES,
             settings: list[tuple[int, int]] | None = None) -> tuple[int, int, int, bool]:
    """Try GIF_SETTINGS in order until the file is under max_bytes.

    Returns (width, fps, bytes, under_limit). If even the smallest setting is
    too big, that last file is kept and under_limit is False.
    """
    width = fps = size = 0
    for width, fps in settings or GIF_SETTINGS:
        encode_gif(src, dest, width, fps)
        size = dest.stat().st_size
        if size < max_bytes:
            return width, fps, size, True
    return width, fps, size, False


@dataclass
class ExportResult:
    mp4: Path
    gif: Path
    seconds: float          # length of the trimmed clip
    gif_bytes: int
    gif_width: int
    gif_fps: int
    gif_under_limit: bool
    archived: list[Path]    # earlier versions moved to references/_old

    @property
    def gif_settings(self) -> str:
        return f"{self.gif_width}px {self.gif_fps}fps"


def export_take(raw: Path, r: RefSign, window_start: int, window_end: int, fps: float,
                n_frames: int, base: Path | None = None, version: int = 1) -> ExportResult:
    """Encode the raw take into the final .mp4 and .gif of sign r (version 1 or 2).

    Both files are first made in references/_tmp; only when both succeeded is
    an older version moved to references/_old and the new files moved in place.
    """
    start_s, end_s = trim_range(window_start, window_end, fps, n_frames)
    work = tmp_dir(base)
    work.mkdir(parents=True, exist_ok=True)
    part_mp4 = work / f"{r.sign.sign_id}.v{version}.part.mp4"
    part_gif = work / f"{r.sign.sign_id}.v{version}.part.gif"
    try:
        encode_mp4(raw, part_mp4, start_s, end_s)
        width, gif_fps, size, ok = make_gif(part_mp4, part_gif)
        archived = archive_old(r, base, version=version)
        final_mp4, final_gif = mp4_path(r, base, version), gif_path(r, base, version)
        final_mp4.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part_mp4, final_mp4)
        os.replace(part_gif, final_gif)
    finally:
        part_mp4.unlink(missing_ok=True)
        part_gif.unlink(missing_ok=True)
    return ExportResult(final_mp4, final_gif, end_s - start_s, size, width, gif_fps, ok, archived)


# =============================================================================
# MANIFEST
# =============================================================================
def manifest_path(base: Path | None = None) -> Path:
    return validation_dir(base) / "manifest.csv"


def read_manifest(base: Path | None = None) -> dict[tuple[str, int], dict[str, str]]:
    """manifest.csv rows by (sign_id, version) (empty if there is no manifest yet).

    A manifest from before versions existed has no version column: its rows are version 1.
    """
    path = manifest_path(base)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {(row["sign_id"], int(row.get("version") or 1)): row
                for row in csv.DictReader(f) if row.get("sign_id")}


def write_manifest(rows: dict[tuple[str, int], dict[str, str]], refs: list[RefSign],
                   base: Path | None = None) -> None:
    """Write the rows in form order, version 1 before version 2 (temporary file + rename).

    Written with a BOM so Excel shows the apostrophes and dashes correctly.
    """
    path = manifest_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in refs:
            for v in VERSIONS:
                if (r.sign.sign_id, v) in rows:
                    writer.writerow(rows[(r.sign.sign_id, v)])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def manifest_row(r: RefSign, result: ExportResult, hand_detect_rate: float, hand: str,
                 recorded_at: datetime | None = None, version: int = 1) -> dict[str, str]:
    stem = version_stem(r, version)
    return {
        "no": f"{r.no:02d}",
        "category": r.sign.category,
        "folder": r.folder,
        "sign_id": r.sign.sign_id,
        "sign_label": r.sign.sign_label,
        "file_stem": stem,
        "mp4": f"{r.folder}/{stem}.mp4",
        "gif": f"{r.folder}/{stem}.gif",
        "seconds": f"{result.seconds:.2f}",
        "gif_kb": f"{result.gif_bytes / 1024:.0f}",
        "hand_detect_rate": f"{hand_detect_rate:.3f}",
        "hand": hand,
        "recorded_at": f"{recorded_at or datetime.now():%Y-%m-%dT%H:%M:%S}",
        "gif_settings": result.gif_settings,
        "version": str(version),
    }


def recorded_ids(refs: list[RefSign], version: int, base: Path | None = None) -> set[str]:
    """Signs whose given version has a manifest row AND its .mp4 on disk."""
    rows = read_manifest(base)
    return {r.sign.sign_id for r in refs
            if (r.sign.sign_id, version) in rows and mp4_path(r, base, version).exists()}


def done_ids(refs: list[RefSign], base: Path | None = None) -> set[str]:
    """Signs that have their reference video (version 1)."""
    return recorded_ids(refs, 1, base)


def version2_ids(refs: list[RefSign], base: Path | None = None) -> set[str]:
    """Signs that also have a second version."""
    return recorded_ids(refs, 2, base)


# =============================================================================
# KEEP: export + manifest + form files, in that order
# =============================================================================
def keep_take(raw: Path, r: RefSign, refs: list[RefSign], *, window_start: int, window_end: int,
              fps: float, n_frames: int, hand_detect_rate: float, hand: str,
              base: Path | None = None, version: int = 1) -> ExportResult:
    """Everything that happens when K is pressed. The raw take is deleted only
    after the .mp4 and .gif are in place and the manifest is written."""
    result = export_take(raw, r, window_start, window_end, fps, n_frames, base, version)
    rows = read_manifest(base)
    rows[(r.sign.sign_id, version)] = manifest_row(r, result, hand_detect_rate, hand, version=version)
    write_manifest(rows, refs, base)
    write_form_files(refs, base)
    raw.unlink(missing_ok=True)
    for p in result.archived:
        log(f"ARCHIVED {p.relative_to(base_dir(base)).as_posix()}", base)
    log(f"KEPT {r.folder}/{version_stem(r, version)} seconds={result.seconds:.2f} hand_detect={hand_detect_rate:.0%} "
        f"gif={result.gif_bytes / 1024:.0f}KB ({result.gif_settings}"
        f"{'' if result.gif_under_limit else ', STILL OVER 2 MB'}) hand={hand}", base)
    return result


# =============================================================================
# FORM MATERIAL
# =============================================================================
def form_sections(refs: list[RefSign], base: Path | None = None) -> list[dict]:
    """The signs the validators check: one entry per folder, one per sign WITH a
    reference video (a folder without any is left out).

    A sign with a second version also has "mp4_2" and "gif_2".
    """
    done, two = done_ids(refs, base), version2_ids(refs, base)
    sections: list[dict] = []
    for folder, _ in CATEGORY_FOLDERS:
        signs = []
        for r in refs:
            if r.folder != folder or r.sign.sign_id not in done:
                continue
            s = {"no": f"{r.no:02d}", "label": r.sign.sign_label, "sign_id": r.sign.sign_id,
                 "mp4": f"{r.file_stem}.mp4", "gif": f"{r.file_stem}.gif"}
            if r.sign.sign_id in two:
                s["mp4_2"], s["gif_2"] = f"{version_stem(r, 2)}.mp4", f"{version_stem(r, 2)}.gif"
            signs.append(s)
        if signs:
            sections.append({"folder": folder, "title": folder_title(folder), "signs": signs})
    return sections


def request_signs(refs: list[RefSign], base: Path | None = None) -> list[RefSign]:
    """Signs without a reference video: the form asks the validators to send one."""
    done = done_ids(refs, base)
    return [r for r in refs if r.sign.sign_id not in done]


def form_list_md(refs: list[RefSign], base: Path | None = None) -> str:
    done, two = done_ids(refs, base), version2_ids(refs, base)
    lines = [f"# {FORM_TITLE}: sign list", "",
             f"{len(refs)} signs in {len(CATEGORY_FOLDERS)} categories. "
             f"Reference videos recorded: {len(done)} / {len(refs)}. "
             f"Signs with a second version: {len(two)}.", ""]
    for folder, _ in CATEGORY_FOLDERS:
        lines += [f"## {folder}", ""]
        lines += [r.title + (" (2 versions)" if r.sign.sign_id in two else "")
                  for r in refs if r.folder == folder]
        lines.append("")
    missing = request_signs(refs, base)
    if missing:
        lines += ["## Not in the form yet: validators are asked to send a video", ""]
        lines += [r.sign.sign_label for r in missing]
        lines.append("")
    return "\n".join(lines)


def build_form_gs(refs: list[RefSign], base: Path | None = None) -> str:
    data = form_sections(refs, base)
    requests = [r.sign.sign_label for r in request_signs(refs, base)]
    return (GS_TEMPLATE
            .replace("__GENERATED__", f"{datetime.now():%Y-%m-%d %H:%M}")
            .replace("__TITLE__", FORM_TITLE)
            .replace("__CONTACT__", CONTACT_LINE)
            .replace("__TOTAL__", str(sum(len(s["signs"]) for s in data)))
            .replace("__GROUPS__", str(len(data)))
            .replace("__REQUESTS__", json.dumps(requests, ensure_ascii=False))
            .replace("__SECTIONS__", json.dumps(data, indent=2, ensure_ascii=False)))


def form_setup_md(refs: list[RefSign]) -> str:
    return FORM_SETUP_TEMPLATE.replace("__TOTAL__", str(len(refs))).replace("__TITLE__", FORM_TITLE)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def write_form_files(refs: list[RefSign], base: Path | None = None) -> None:
    """(Re)write form_list.md, build_validation_form.gs and FORM_SETUP.md."""
    folder = validation_dir(base)
    _write_text(folder / "form_list.md", form_list_md(refs, base))
    _write_text(folder / "build_validation_form.gs", build_form_gs(refs, base))
    _write_text(folder / "FORM_SETUP.md", form_setup_md(refs))


# -----------------------------------------------------------------------------
# Google Apps Script template (placeholders: __GENERATED__ __TITLE__ __CONTACT__
# __TOTAL__ __GROUPS__ __REQUESTS__ __SECTIONS__)
# -----------------------------------------------------------------------------
GS_TEMPLATE = r"""/**
 * __TITLE__ : builder (Google Apps Script)
 *
 * Generated by record_tool.py --reference on __GENERATED__ from references/validation/manifest.csv.
 * It is rewritten after every kept reference take, so do not edit it by hand.
 *
 * How to use it (details in FORM_SETUP.md):
 *   1. Upload the "validation" folder to Google Drive and paste its ID in FOLDER_ID below.
 *      If some signs have no video yet, also paste the link of the upload folder in UPLOAD_LINK.
 *   2. Choose buildForm and click Run. It always creates a NEW form; an existing form is never changed.
 *   3. Google stops a run after about 6 minutes. If the log says "Run continueBuild", choose
 *      continueBuild and click Run until the log says FINISHED.
 *   4. The log ends with the edit link and the link for the validators.
 *
 * Each sign shows its name and its GIF (copied into the form; Drive sharing is not changed).
 */

var FOLDER_ID = 'PASTE_DRIVE_FOLDER_ID_HERE';
// Link of a Google Drive folder where validators can upload videos of the signs we have no
// reference for (Share -> "Anyone with the link" -> Editor). See FORM_SETUP.md.
var UPLOAD_LINK = 'PASTE_UPLOAD_FOLDER_LINK_HERE';

var FORM_TITLE = '__TITLE__';
var CHOICES = ['Correct', 'Other version', 'Needs fix', 'Wrong'];
var FOLLOW_UP = 'If not Correct, what should change?';
var VERSION_QUESTION = 'Which version is correct?';
var VERSION_CHOICES = ['Version 1', 'Version 2', 'Neither is correct'];
var VERSION_FOLLOW_UP = 'Comments (for example, which version people use in your area):';
var HAND_QUESTION = 'Can it also be signed with the other hand (left or right)?';
var HAND_CHOICES = ['Yes, either hand is fine', 'No, only with this hand', 'Not sure'];
var IMAGE_WIDTH = 480;
var TIME_LIMIT_MS = 4.5 * 60 * 1000;

var INTRO = [
  'HUDYAT is a learning app for Deaf and hard-of-hearing learners. It helps them practise ' +
  'Filipino Sign Language (FSL) and tells them if their sign is right. The app learns each sign ' +
  'from videos, so every sign must be correct before we use it.',
  '',
  'What you will do: for each sign, watch the moving picture. Then choose one answer:',
  '• Correct: the sign is right and used here',
  '• Other version: the sign is right, but people here also use a different sign',
  '• Needs fix: almost right, something should change',
  '• Wrong: this is not the sign',
  'If you did not choose Correct, please write what should change in the box under the question.',
  '',
  'Then tell us if the sign can also be signed with the other hand (left-handed or right-handed ' +
  'learners), or if it must be signed with the hand shown.',
  '',
  'Some signs have two versions (two ways of signing the same word). For these, watch both ' +
  'and tick every version that is correct, or "Neither is correct".',
  '',
  'There are __TOTAL__ signs in __GROUPS__ groups. It takes about 30–45 minutes. You do not need to finish ' +
  'in one sitting: if you are signed in to Google, the form keeps your answers and you can come back later.',
  '',
  'At the end, we ask for your help with a few signs we could not find a good reference for.',
  '',
  'Your answers are used only for the HUDYAT project.'
].join('\n');

var THANK_YOU =
  'Thank you very much for your time and your knowledge. Your answers help make sure that HUDYAT ' +
  'teaches Filipino Sign Language correctly and respectfully to Deaf learners. Salamat po!';
var CONTACT = '__CONTACT__';

// Signs we have no reference video for: the last page asks the validators to send a video.
var REQUEST_SIGNS = __REQUESTS__;
var KNOW_CHOICES = ['I know it', 'Not sure', "I don't know"];
var SEND_CHOICES = ['Yes, on Messenger', 'Yes, I will upload it to the Google Drive folder', 'No'];

// One section per category, in form order. Files are looked up by name in <FOLDER_ID>/<folder>/.
// Only signs with a reference video are listed.
var SECTIONS = __SECTIONS__;

function buildForm() {
  if (!FOLDER_ID || FOLDER_ID === 'PASTE_DRIVE_FOLDER_ID_HERE') {
    Logger.log('Paste the ID of the uploaded "validation" folder in FOLDER_ID first (see FORM_SETUP.md).');
    return;
  }
  var root = DriveApp.getFolderById(FOLDER_ID);   // stops here with a clear error if the ID is wrong
  var missing = SECTIONS.filter(function (sec) { return !subFolder_(root, sec.folder); });
  if (missing.length) {   // check before creating anything: a form without videos is useless
    var seen = [];
    for (var it = root.getFolders(); it.hasNext();) seen.push(it.next().getName());
    Logger.log('STOPPED, no form was created. The folder "' + root.getName() + '" does not contain: ' +
               missing.map(function (sec) { return sec.folder; }).join(', '));
    Logger.log('Folders it does contain: ' + (seen.length ? seen.join(', ') : '(none)'));
    Logger.log('FOLDER_ID must be the ID of the "validation" folder that holds 01 Alphabet, 02 Numbers, ... ' +
               'If you see a "validation" folder above, open it and use its ID. If the upload is still ' +
               'running, wait until it has finished.');
    return;
  }

  var form = FormApp.create(FORM_TITLE);   // always a NEW form
  form.setDescription(INTRO);
  form.setProgressBar(true);
  form.setConfirmationMessage(THANK_YOU + '\n\n' + CONTACT);

  form.addPageBreakItem().setTitle('Validator information');
  form.addTextItem().setTitle('Name').setRequired(true);
  form.addMultipleChoiceItem().setTitle('Organization')
    .setChoiceValues(['Tacloban Association of the Deaf (TAD)', 'Leyte Association of the Deaf (LAD)'])
    .showOtherOption(true);
  form.addTextItem().setTitle('Role or position');

  Logger.log('Created a new form: ' + form.getEditUrl());
  saveState_({ formId: form.getId(), sec: 0, sign: 0, opened: false, signs: 0, gifs: 0,
               problems: [] });
  continueBuild();
}

function continueBuild() {
  var started = Date.now();
  var st = loadState_();
  if (!st) { Logger.log('Nothing to continue. Run buildForm first.'); return; }
  var form = FormApp.openById(st.formId);   // the form buildForm created
  var root = DriveApp.getFolderById(FOLDER_ID);

  for (; st.sec < SECTIONS.length; st.sec++, st.sign = 0, st.opened = false) {
    var sec = SECTIONS[st.sec];
    var folder = subFolder_(root, sec.folder);
    if (!st.opened) {
      if (!folder) st.problems.push('Folder not found in Drive: ' + sec.folder);
      form.addPageBreakItem().setTitle(sec.title).setHelpText(
        'Watch each sign (the picture moves by itself), then answer the questions under it.');
      st.opened = true;
    }
    for (; st.sign < sec.signs.length; st.sign++) {
      if (Date.now() - started > TIME_LIMIT_MS) {
        saveState_(st);
        Logger.log('Done so far: ' + st.signs + ' signs. Google\'s time limit is near. ' +
                   '>>> Choose continueBuild and click Run to go on. <<<');
        return;
      }
      addSign_(form, folder, sec, sec.signs[st.sign], st);
    }
  }

  if (REQUEST_SIGNS.length) addRequestPage_(form);
  form.addPageBreakItem().setTitle('Thank you').setHelpText(THANK_YOU + '\n\n' + CONTACT);

  PropertiesService.getScriptProperties().deleteProperty('BUILD_STATE');
  Logger.log('==== FINISHED ====');
  Logger.log('Signs in the form: ' + st.signs + ' (moving pictures: ' + st.gifs + ')');
  Logger.log('Problems: ' + st.problems.length);
  st.problems.forEach(function (m) { Logger.log('  - ' + m); });
  Logger.log('Edit link: ' + form.getEditUrl());
  Logger.log('Live link for validators: ' + form.getPublishedUrl());
}

function addSign_(form, folder, sec, s, st) {
  var title = s.label;
  if (s.gif_2) {   // two versions: both pictures, then one question
    addMedia_(form, folder, sec, title + ' – Version 1', s.gif, st);
    addMedia_(form, folder, sec, title + ' – Version 2', s.gif_2, st);
    form.addCheckboxItem().setTitle(title + ': ' + VERSION_QUESTION)
      .setChoiceValues(VERSION_CHOICES).setRequired(true);
    form.addTextItem().setTitle(VERSION_FOLLOW_UP);
  } else {
    addMedia_(form, folder, sec, title, s.gif, st);
    form.addMultipleChoiceItem().setTitle(title).setChoiceValues(CHOICES).setRequired(true);
    form.addTextItem().setTitle(FOLLOW_UP);
  }
  form.addMultipleChoiceItem().setTitle(title + ': ' + HAND_QUESTION)
    .setChoiceValues(HAND_CHOICES).setRequired(true);
  st.signs++;
}

// The sign name and its moving picture (GIF) only; just the name if the GIF is missing.
function addMedia_(form, folder, sec, title, gifName, st) {
  var gif = folder ? findFile_(folder, gifName) : null;
  if (gif) {
    try {
      form.addImageItem().setTitle(title).setImage(gif.getBlob())
        .setWidth(IMAGE_WIDTH).setAlignment(FormApp.Alignment.CENTER);
      st.gifs++;
      return;
    } catch (e) {
      st.problems.push(title + ': could not add the GIF (' + e + ')');
    }
  } else {
    st.problems.push(title + ': GIF not found (' + sec.folder + '/' + gifName + ')');
  }
  form.addSectionHeaderItem().setTitle(title);
}

// Last page: the signs without a reference video. Who knows them, and can they send a video?
function addRequestPage_(form) {
  var upload = (UPLOAD_LINK && UPLOAD_LINK !== 'PASTE_UPLOAD_FOLDER_LINK_HERE')
    ? UPLOAD_LINK : '(we will send you the link)';
  form.addPageBreakItem().setTitle('Help us with ' + REQUEST_SIGNS.length + ' more signs').setHelpText(
    'We could not find a good reference video for these signs. If you know them, please help us: ' +
    'a short video of you signing each one is the best way to show us.');
  form.addGridItem().setTitle('Which of these signs do you know?')
    .setRows(REQUEST_SIGNS).setColumns(KNOW_CHOICES);
  form.addCheckboxItem().setTitle('Can you send us a video of the signs you know?')
    .setHelpText('Messenger: ' + CONTACT + '\nGoogle Drive folder: ' + upload +
                 '\nPlease put the name of the sign in the message or in the file name.')
    .setChoiceValues(SEND_CHOICES);
  form.addTextItem().setTitle('Your Messenger name or phone number (so we can reach you)');
  form.addParagraphTextItem().setTitle('Notes (for example, if a sign has different versions in your area)');
}

function saveState_(st) {
  PropertiesService.getScriptProperties().setProperty('BUILD_STATE', JSON.stringify(st));
}

function loadState_() {
  var s = PropertiesService.getScriptProperties().getProperty('BUILD_STATE');
  return s ? JSON.parse(s) : null;
}

function subFolder_(parent, name) {
  var it = parent.getFoldersByName(name);
  return it.hasNext() ? it.next() : null;
}

function findFile_(folder, name) {
  var it = folder.getFilesByName(name);
  return it.hasNext() ? it.next() : null;
}
"""

FORM_SETUP_TEMPLATE = """# Building the __TITLE__

The `validation` folder (this folder) holds one `.mp4` and one `.gif` per
recorded sign, `manifest.csv`, `form_list.md` and `build_validation_form.gs`.
Only signs with a reference video are in the form. The others (listed at the
end of `form_list.md`) are on the last page, where the validators are asked
to send a video of them on Messenger or to a Google Drive upload folder.
A sign with a second version also has `<NN> <Sign Label> (2).mp4` / `.gif`; in
the form, both versions are shown and the validator ticks which one is correct.
The script and lists are refreshed by the recorder after every kept reference
take, so record first, then build the form.

## Steps

1. **Upload the folder to Google Drive.** Drag the whole `validation` folder
   into Google Drive in the browser, or let **Drive for desktop** sync it (copy
   it into the `My Drive` folder that Drive for desktop shows in File
   Explorer). Wait until every file has finished uploading. Keep the folder
   and file names exactly as they are: the script finds the files by name.
2. **Copy the folder ID.** Open the uploaded `validation` folder in Drive. The
   address bar shows `https://drive.google.com/drive/folders/<FOLDER ID>`.
   Copy the part after `folders/` (a long code of letters and numbers).
3. **Create a script project.** Go to <https://script.google.com>, sign in with
   the same Google account, and click **New project**.
4. **Paste the script.** Open `build_validation_form.gs` in Notepad, select
   all (Ctrl+A), copy, and paste it over everything in the editor's `Code.gs`.
   On the line `var FOLDER_ID = 'PASTE_DRIVE_FOLDER_ID_HERE';` replace
   `PASTE_DRIVE_FOLDER_ID_HERE` with the folder ID (keep the quotes).
   If some signs have no video yet, make the **upload folder**: in Google
   Drive, create a new folder (for example `HUDYAT sign videos from
   validators`), click **Share**, set **General access** to **Anyone with the
   link** and the role to **Editor**, and click **Copy link**. On the line
   `var UPLOAD_LINK = 'PASTE_UPLOAD_FOLDER_LINK_HERE';` replace
   `PASTE_UPLOAD_FOLDER_LINK_HERE` with that link. (Uploading needs a Google
   account, and anyone with the link can see the folder, so use it only for
   these videos.) Also replace `[Contact: name, Messenger/phone]` on the line
   `var CONTACT = ...` with your name and Messenger name or phone number:
   validators see it on the last pages and use it to send videos. Click
   **Save** (the disk icon).
5. **Run `buildForm` and authorize.** In the toolbar, choose `buildForm` in the
   function list and click **Run**. The first time, Google asks for
   permission: click **Review permissions**, choose your account, then
   **Advanced → Go to (project name) (unsafe)** and **Allow**. (It is your own
   script; it needs Forms to create the form and Drive to read the GIFs.)
6. **Wait for it to finish.** Google stops a run after about 6 minutes. If the
   log says **"Choose continueBuild and click Run"**, choose `continueBuild`
   and click **Run**; repeat until the log says **FINISHED**.
7. **Open the link in the log.** The log (View → Logs, or the Execution log
   panel) ends with:
   - **Edit link**: for you, to check and adjust the form;
   - **Live link for validators**: the one you send to the validators.
   It also lists any problems (for example a missing GIF, in which case only
   the sign name is shown).

## Before sending the link

- Check that your contact (step 4) shows on the last pages, not
  `[Contact: name, Messenger/phone]`.
- Open the live link in a private browser window and check that the GIFs move.
- Running `buildForm` again makes another new form; delete test forms you do
  not need in Google Drive.
"""


# =============================================================================
# python hudyat_reference.py : write the form material now (no camera needed)
# =============================================================================
def main() -> int:
    try:
        refs = reference_signs(hc.load_signs())
    except ValueError as exc:
        print(f"Stopped: {exc}")
        return 1
    write_form_files(refs)
    done = done_ids(refs)
    folder = validation_dir()
    print(f"Form material written to {folder}")
    for name in ("build_validation_form.gs", "FORM_SETUP.md", "form_list.md"):
        print(f"  {name}")
    print(f"Reference videos recorded: {len(done)} / {len(refs)}")
    missing = request_signs(refs)
    if missing:
        print(f"Not in the form ({len(missing)}, the validators are asked to send a video): "
              + ", ".join(r.sign.sign_label for r in missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
