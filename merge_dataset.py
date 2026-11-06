"""
merge_dataset.py - Bring recordings made on ANOTHER laptop into this dataset.

Each researcher may record on their own laptop. Their clips have the same
kind of names as ours (circle_S3_01.mp4 ...), and take numbers start at 01 on
every laptop, so copying folders by hand can overwrite a recording. This
script copies them safely:

  * Every clip of the other laptop that has a metadata.csv row is copied to
        dataset/raw/<category>/<sign_id>/<signer>/<sign_id>_<signer>_<NN>.<ext>
    and its row is appended to this dataset/metadata.csv (status kept /
    needs_rerecord is preserved).
  * The take number stays the same when it is free here. If it is already
    used (in metadata.csv or on disk), the clip gets the next free number.
    Existing files are NEVER overwritten.
  * Running it twice is safe: a clip already merged (same signer, sign,
    timestamp, frame count and camera) is skipped.
  * PUB rows are skipped (import the public dataset with import_public.py).
  * A clip whose sign_id is not in signs.csv, or whose video file is missing,
    is skipped and listed.
  * If the other laptop saved the signer under the wrong code (e.g. Russel
    recorded as S1 on his laptop but is S2 here), use --rename S1=S2.
  * --categories family wh_questions merges only those categories.

Usage (PowerShell, venv active, from this folder):
    python merge_dataset.py --source "E:\\Russel\\HUDYAT - hand tracker" --dry-run
    python merge_dataset.py --source "E:\\Russel\\HUDYAT - hand tracker"
    python merge_dataset.py --source "E:\\Christian\\dataset" --only S3
    python merge_dataset.py --source "E:\\Russel\\dataset" --rename S1=S2
    python merge_dataset.py --source "E:\\Russel\\dataset" --categories family wh_questions

--source can be the other laptop's project folder or its dataset folder
(the one that contains metadata.csv and raw\\).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable

import hudyat_common as hc
import hudyat_store as store
from import_public import append_rows

PUB = "PUB"


@dataclass(frozen=True)
class PlannedCopy:
    src: str              # file_path as written in the source metadata.csv
    dest: str             # project-relative destination, '/'-separated
    row: dict             # metadata row to append here (signer / take / file_path updated)


def row_key(row: dict) -> tuple:
    """Identity of a recording, independent of its file name / take number."""
    return (row.get("signer", ""), row.get("sign_id", ""), row.get("timestamp", ""),
            row.get("n_frames", ""), row.get("camera_name", ""))


def dest_path(category: str, sign_id: str, signer: str, take: int, ext: str) -> str:
    return f"dataset/raw/{category}/{sign_id}/{signer}/{sign_id}_{signer}_{take:02d}{ext}"


def plan_merge(source_rows: list[dict], local_rows: list[dict],
               valid_signs: dict[str, str],
               disk_takes: Callable[[str, str, str], set[int]] = lambda c, s, g: set(),
               rename: dict[str, str] | None = None,
               only: set[str] | None = None,
               categories: set[str] | None = None) -> tuple[list[PlannedCopy], list[tuple[str, str]], int]:
    """Decide what to copy. Returns (plans, skipped [(file, reason)], n_already_merged).

    valid_signs: sign_id -> category (from signs.csv).
    disk_takes(category, sign_id, signer): take numbers already present as files here.
    categories: merge only these; rows of other categories are ignored, not reported.
    """
    rename = rename or {}
    seen = {row_key(r) for r in local_rows}
    used: dict[tuple[str, str], set[int]] = {}
    for r in local_rows:
        if r.get("take", "").isdigit():
            used.setdefault((r.get("signer", ""), r.get("sign_id", "")), set()).add(int(r["take"]))

    plans: list[PlannedCopy] = []
    skipped: list[tuple[str, str]] = []
    already = 0
    for src_row in source_rows:
        src = src_row.get("file_path", "")
        signer = rename.get(src_row.get("signer", ""), src_row.get("signer", ""))
        if signer == PUB or (only and signer not in only):
            continue
        sign_id = src_row.get("sign_id", "")
        if categories and valid_signs.get(sign_id, src_row.get("category", "")) not in categories:
            continue
        if sign_id not in valid_signs:
            skipped.append((src, f"sign_id '{sign_id}' is not in signs.csv"))
            continue
        category = valid_signs[sign_id]
        row = dict(src_row, signer=signer, category=category)
        key = row_key(row)
        if key in seen:
            already += 1
            continue

        taken = used.setdefault((signer, sign_id), set())
        taken |= disk_takes(category, sign_id, signer)
        orig = int(src_row["take"]) if src_row.get("take", "").isdigit() else 0
        take = orig if orig >= 1 and orig not in taken else max(taken, default=0) + 1
        ext = PurePosixPath(src).suffix or ".mp4"
        dest = dest_path(category, sign_id, signer, take, ext)
        row.update(take=str(take), file_path=dest)
        plans.append(PlannedCopy(src, dest, row))
        taken.add(take)
        seen.add(key)
    return plans, skipped, already


def find_source_root(source: Path) -> Path | None:
    """The folder that file_path values are relative to (it contains dataset/metadata.csv)."""
    if (source / "dataset" / "metadata.csv").exists():
        return source
    if (source / "metadata.csv").exists() and source.name.lower() == "dataset":
        return source.parent
    return None


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def takes_on_disk(category: str, sign_id: str, signer: str) -> set[int]:
    folder = hc.RAW_DIR / category / sign_id / signer
    prefix = f"{sign_id}_{signer}_"
    out: set[int] = set()
    if folder.is_dir():
        for p in folder.iterdir():
            stem = p.name.split(".")[0]
            if stem.startswith(prefix) and stem[len(prefix):].isdigit():
                out.add(int(stem[len(prefix):]))
    return out


def parse_rename(values: list[str]) -> dict[str, str]:
    out = {}
    for v in values:
        old, sep, new = v.partition("=")
        if not sep or not old.strip() or not new.strip():
            raise ValueError(f"--rename needs OLD=NEW, got '{v}'")
        out[old.strip().upper()] = new.strip().upper()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge recordings from another laptop into this dataset")
    parser.add_argument("--source", required=True, type=Path,
                        help="the other laptop's project folder (or its dataset folder)")
    parser.add_argument("--only", nargs="+", metavar="SIGNER", help="merge only these signers, e.g. --only S2 S3")
    parser.add_argument("--rename", nargs="+", default=[], metavar="OLD=NEW",
                        help="signer saved under another code there, e.g. --rename S1=S2")
    parser.add_argument("--categories", nargs="+", metavar="CATEGORY",
                        help="merge only these categories, e.g. --categories family wh_questions")
    parser.add_argument("--dry-run", action="store_true", help="show what would be merged, change nothing")
    args = parser.parse_args()

    root = find_source_root(args.source)
    if root is None:
        print(f"No dataset\\metadata.csv found in {args.source}.\n"
              "Point --source at the other laptop's project folder or its 'dataset' folder.")
        return 1
    if root.resolve() == hc.PROJECT_DIR.resolve():
        print("--source is this project itself; nothing to merge.")
        return 1
    try:
        rename = parse_rename(args.rename)
    except ValueError as exc:
        print(exc)
        return 1
    only = {s.upper() for s in args.only} if args.only else None

    categories = set(args.categories) if args.categories else None
    source_rows = read_rows(root / "dataset" / "metadata.csv")
    valid = {s.sign_id: s.category for s in hc.load_signs()}
    if categories and categories - set(valid.values()):
        print(f"Unknown categories: {', '.join(sorted(categories - set(valid.values())))}. "
              f"signs.csv has: {', '.join(sorted(set(valid.values())))}")
        return 1
    plans, skipped, already = plan_merge(source_rows, store.read_metadata(), valid,
                                         takes_on_disk, rename, only, categories)

    missing = [p for p in plans if not (root / p.src).is_file()]
    plans = [p for p in plans if (root / p.src).is_file()]
    skipped += [(p.src, "video file not found in the source folder") for p in missing]

    renumbered = sum(1 for p in plans if PurePosixPath(p.src).name != PurePosixPath(p.dest).name)
    print(f"Source: {root}")
    print(f"{len(plans)} clip(s) to merge ({renumbered} get a new take number), "
          f"{already} already merged, {len(skipped)} skipped.")
    for signer, n in sorted(Counter(p.row["signer"] for p in plans).items()):
        print(f"  {signer}: {n} clip(s)")

    # A signer's hand must be the same everywhere.
    signers = store.load_signers()
    for signer, hands in sorted({(p.row["signer"], p.row.get("signing_hand", "")) for p in plans}):
        here = signers.get(signer)
        if here and here.hand and hands and hands != here.hand:
            print(f"  WARNING: {signer} is '{hands}'-handed in the source but '{here.hand}' in signers.csv here.")

    for src, why in skipped[:40]:
        print(f"  SKIPPED {src}: {why}")
    if len(skipped) > 40:
        print(f"  ... and {len(skipped) - 40} more skipped")

    if args.dry_run:
        for p in plans:
            note = "" if PurePosixPath(p.src).name == PurePosixPath(p.dest).name else "   (renumbered)"
            print(f"  {p.src}  ->  {p.dest}{note}")
        print("\nDry run: nothing was changed.")
        return 0
    if not plans:
        return 0

    log_lines = [f"merge from {root} at {datetime.now():%Y-%m-%d %H:%M}"]
    done = failed = 0
    for k, p in enumerate(plans, 1):
        dest = hc.PROJECT_DIR / p.dest
        if dest.exists():      # should not happen (planned around disk files), never overwrite
            print(f"[{k}/{len(plans)}] SKIPPED {p.dest}: a file is already there.")
            failed += 1
            continue
        tmp = dest.with_name(dest.stem + ".merge.tmp")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / p.src, tmp)
            tmp.replace(dest)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            print(f"[{k}/{len(plans)}] FAILED {p.src}: {exc}")
            failed += 1
            continue
        append_rows(hc.METADATA_CSV, [{f: p.row.get(f, "") for f in store.METADATA_FIELDS}])
        log_lines.append(f"{p.src} -> {p.dest}")
        done += 1
    hc.SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = hc.SESSION_LOG_DIR / f"merge_{datetime.now():%Y-%m-%d_%H%M%S}.txt"
    log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"\nMerged {done}, failed {failed}. Log: {log}")
    print("Next: re-run landmark extraction so the new clips are included.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Clips merged so far are complete; run again to continue.")
        sys.exit(1)
