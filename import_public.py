"""
import_public.py - Import the FSL-105 public dataset as signer "PUB".

PUB clips are used for TRAINING only, in both the random split and the
leave-one-signer-out test (README section 4). They are never
recorded with record_tool.py.

Usage:
    python import_public.py --source "D:\\FSL-105\\clips"            (import)
    python import_public.py --source "D:\\FSL-105\\clips" --dry-run  (show the plan only)
    python import_public.py --source ... --flip                     (source videos are mirrored)

public_mapping.csv (next to this script) says which FSL-105 label becomes
which of our signs:
    fsl105_id,fsl105_label,category,sign_id
    0,GOOD MORNING,greetings,good_morning
A source video belongs to a label when its top folder under --source is named
after the fsl105_id or the fsl105_label (case-insensitive). If the source
folder has FSL-105's own labels.csv (id,label columns), it is used to link
ids and labels, so filling in either one in the mapping is enough.
Labels that are not in the mapping are skipped.

Each video (.mov/.mp4, HEVC or H.264, phone rotation metadata applied) is
converted to H.264 .mp4 with ffmpeg (system ffmpeg, else the copy bundled
with the imageio-ffmpeg package) and saved as
    dataset/raw/<category>/<sign_id>/PUB/<sign_id>_PUB_<NN>.mp4
with one metadata.csv row (signer PUB, status kept). The source path is
stored in the camera_name column ("FSL-105:<path>"), so running the script
again never imports the same file twice. Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

import hudyat_store as store

PUB = "PUB"
SOURCE_TAG = "FSL-105:"      # camera_name of a PUB row = SOURCE_TAG + source path (used to skip re-imports)
ORIENTATION = "camera_raw_unmirrored"   # what the saved PUB clips are (use --flip if the source is mirrored)


@dataclass(frozen=True)
class PlannedImport:
    source: str          # path relative to the FSL-105 folder, '/'-separated
    category: str
    sign_id: str
    take: int
    dest: str            # project-relative output path

    @property
    def fsl105_key(self) -> str:
        return self.source.split("/")[0]


def plan_imports(mapping: list[dict[str, str]], sources: list[str],
                 metadata_rows: list[dict[str, str]]) -> tuple[list[PlannedImport], list[str]]:
    """Decide which source files become which PUB clips.

    A source file's top folder is matched (case-insensitive) against the
    mapping's fsl105_id or fsl105_label. Returns (planned imports, skipped
    sources); files of unmapped labels are skipped.
    """
    by_key: dict[str, dict[str, str]] = {}
    for m in mapping:
        for key in (m.get("fsl105_id", ""), m.get("fsl105_label", "")):
            if key.strip():
                by_key[key.strip().lower()] = m
    done = {r.get("camera_name", "")[len(SOURCE_TAG):] for r in metadata_rows
            if r.get("signer") == PUB and r.get("camera_name", "").startswith(SOURCE_TAG)}
    next_take: dict[str, int] = defaultdict(lambda: 1)
    for r in metadata_rows:
        if r.get("signer") == PUB and str(r.get("take", "")).isdigit():
            next_take[r["sign_id"]] = max(next_take[r["sign_id"]], int(r["take"]) + 1)
    plans: list[PlannedImport] = []
    skipped: list[str] = []
    for source in sorted(sources, key=lambda s: (s.split("/")[0], s.lower())):
        m = by_key.get(PurePosixPath(source).parts[0].strip().lower())
        if m is None:
            skipped.append(source)
            continue
        if source in done:
            continue
        sign_id, category = m["sign_id"].strip(), m["category"].strip()
        take = next_take[sign_id]
        next_take[sign_id] += 1
        dest = f"dataset/raw/{category}/{sign_id}/{PUB}/{sign_id}_{PUB}_{take:02d}.mp4"
        plans.append(PlannedImport(source, category, sign_id, take, dest))
    return plans, skipped


def metadata_row(plan: PlannedImport, codec: str, width: int = 0, height: int = 0,
                 fps: float = 0.0, n_frames: int = 0, label: str = "") -> dict[str, object]:
    """The metadata.csv row for one imported clip (same columns as record_tool.py).
    Quality columns stay empty; split_dataset.py takes them from landmarks_index.csv."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "signer": PUB, "signing_hand": "", "category": plan.category, "sign_id": plan.sign_id,
        "sign_label": label, "hands_required": "", "take": plan.take, "file_path": plan.dest,
        "camera_index": "", "camera_name": SOURCE_TAG + plan.source,
        "width": width, "height": height, "fps_actual": f"{fps:.2f}", "n_frames": n_frames,
        "duration_s": f"{(n_frames / fps) if fps else 0:.3f}", "codec": codec,
        "orientation": ORIENTATION, "hand_detect_rate": "", "both_hands_rate": "",
        "mean_brightness": "", "status": store.STATUS_KEPT,
    }


# =============================================================================
# COMMAND LINE
# =============================================================================
MAPPING_CSV = Path(__file__).resolve().parent / "public_mapping.csv"
MAPPING_FIELDS = ["fsl105_id", "fsl105_label", "category", "sign_id"]
SOURCE_EXTENSIONS = {".mov", ".mp4", ".m4v"}
OUTPUT_CODEC = "h264"


def find_ffmpeg() -> str | None:
    """System ffmpeg if installed, else the binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def read_mapping(path: Path, source_dir: Path) -> list[dict[str, str]]:
    """public_mapping.csv rows, with ids/labels completed from FSL-105's labels.csv if present."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in MAPPING_FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name} is missing column(s): {', '.join(missing)}")
        rows = [r for r in reader if (r.get("sign_id") or "").strip()]
    labels_csv = source_dir / "labels.csv"
    if labels_csv.exists():
        with open(labels_csv, newline="", encoding="utf-8-sig") as f:
            pairs = [(str(r.get("id", "")).strip(), str(r.get("label", "")).strip()) for r in csv.DictReader(f)]
        id_of = {label.lower(): i for i, label in pairs if i and label}
        label_of = {i: label for i, label in pairs if i and label}
        for r in rows:
            if not r["fsl105_id"].strip():
                r["fsl105_id"] = id_of.get(r["fsl105_label"].strip().lower(), "")
            if not r["fsl105_label"].strip():
                r["fsl105_label"] = label_of.get(r["fsl105_id"].strip(), "")
    return rows


def convert(ffmpeg: str, src: Path, dest: Path, flip: bool) -> None:
    """Re-encode to H.264 .mp4. ffmpeg applies the rotation metadata itself
    (autorotate) and decodes HEVC; audio is dropped. Written to a .part file
    and renamed, so a crash never leaves a half-written clip."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.stem + ".tmp.mp4")     # ".tmp" = ignored by extract_landmarks.py
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if flip:
        cmd += ["-vf", "hflip"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart", str(part)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not part.exists():
        part.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "ffmpeg failed")
    os.replace(part, dest)


def video_info(path: Path) -> tuple[int, int, float, int]:
    """(width, height, fps, frame count) of a saved clip, read back with OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = 0
        width = height = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            n += 1
        return width, height, fps, n
    finally:
        cap.release()


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Append rows to metadata.csv (header first if the file is new), forced to disk."""
    new_file = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=store.METADATA_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def main() -> int:
    import hudyat_common as hc

    parser = argparse.ArgumentParser(description="Import the FSL-105 public dataset as signer PUB")
    parser.add_argument("--source", required=True, type=Path, help="FSL-105 folder (one sub-folder per label)")
    parser.add_argument("--mapping", type=Path, default=MAPPING_CSV, help="mapping CSV (default public_mapping.csv)")
    parser.add_argument("--flip", action="store_true", help="mirror the videos (only if the source is mirrored)")
    parser.add_argument("--dry-run", action="store_true", help="print what would be imported, change nothing")
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"Source folder not found: {args.source}")
        return 1
    if not args.mapping.exists():
        print(f"Mapping file not found: {args.mapping}")
        return 1
    mapping = read_mapping(args.mapping, args.source)
    if not mapping:
        print(f"{args.mapping.name} has no rows yet. Add one row per FSL-105 label you want "
              "(fsl105_id,fsl105_label,category,sign_id).")
        return 1
    signs = {s.sign_id: s for s in hc.load_signs()}
    bad = [m["sign_id"] for m in mapping if m["sign_id"].strip() not in signs
           or signs[m["sign_id"].strip()].category != m["category"].strip()]
    if bad:
        print(f"Mapping rows whose sign_id/category are not in signs.csv: {', '.join(bad)}")
        return 1

    sources = sorted(p.relative_to(args.source).as_posix() for p in args.source.rglob("*")
                     if p.is_file() and p.suffix.lower() in SOURCE_EXTENSIONS and len(p.relative_to(args.source).parts) >= 2)
    if not sources:
        print(f"No .mov/.mp4 files in sub-folders of {args.source}.")
        return 1
    plans, skipped = plan_imports(mapping, sources, store.read_metadata())
    already = len(sources) - len(plans) - len(skipped)
    print(f"{len(sources)} source video(s): {len(plans)} to import, {already} already imported, "
          f"{len(skipped)} with unmapped labels (skipped).")
    if skipped:
        labels = sorted({s.split('/')[0] for s in skipped})
        print("Unmapped label folders: " + ", ".join(labels[:30]) + (" ..." if len(labels) > 30 else ""))
    if args.dry_run:
        for p in plans:
            print(f"  {p.source}  ->  {p.dest}")
        return 0
    if not plans:
        return 0
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print("ffmpeg not found. Install it, or run:  pip install imageio-ffmpeg")
        return 1

    done = failed = 0
    for k, plan in enumerate(plans, 1):
        dest = hc.PROJECT_DIR / plan.dest
        if dest.exists():
            print(f"[{k}/{len(plans)}] SKIPPED {plan.dest}: a file is already there but has no "
                  "metadata row (left over from an interrupted import?). Check it by hand.")
            failed += 1
            continue
        try:
            convert(ffmpeg, args.source / plan.source, dest, args.flip)
            width, height, fps, n = video_info(dest)
            if n == 0:
                raise RuntimeError("converted file has no readable frames")
        except Exception as exc:
            print(f"[{k}/{len(plans)}] FAILED {plan.source}: {exc}")
            failed += 1
            continue
        append_rows(hc.METADATA_CSV, [metadata_row(plan, OUTPUT_CODEC, width, height, fps, n,
                                                   signs[plan.sign_id].sign_label)])
        done += 1
        print(f"[{k}/{len(plans)}] {plan.source} -> {plan.dest} ({width}x{height}, {n} frames)")
    print(f"\nImported {done}, failed/skipped {failed}. Next: python extract_landmarks.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Already imported clips are complete; run again to continue.")
        sys.exit(1)
