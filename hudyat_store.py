"""
hudyat_store.py - Small data files used by the recorder's dashboard.

  * signers.csv     - signer code, signing hand, optional display name, notes.
                      Nothing else about a signer is stored (privacy).
  * settings.json   - remembered camera index / camera name / backend, so they
                      don't have to be typed every session.
  * Progress        - kept / required counts per signer, sign and category,
                      computed from dataset/metadata.csv (the single source of
                      truth for what has been recorded).
  * set_take_status - flags a kept take as "needs_rerecord" (or back to
                      "kept") by rewriting ONLY the status column of that row.
                      Video files are never deleted or renamed.

Status values in metadata.csv:
    kept            - counts toward the required takes (normal)
    needs_rerecord  - flagged during review; does NOT count, so the sign needs
                      another take. The file stays on disk and its take number
                      is never reused.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import hudyat_common as hc

# =============================================================================
# SETTINGS
# =============================================================================
SIGNER_CODES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
HANDS = ("left", "right")

SIGNERS_CSV = hc.PROJECT_DIR / "signers.csv"
SIGNERS_FIELDS = ["code", "hand", "display_name_optional", "notes"]
SETTINGS_JSON = hc.PROJECT_DIR / "settings.json"
DEFAULT_SETTINGS = {"camera": 0, "camera_name": "unknown", "backend": hc.DEFAULT_CAMERA_BACKEND}

# Columns of dataset/metadata.csv (record_tool.py and import_public.py both append to it).
METADATA_FIELDS = [
    "timestamp", "signer", "signing_hand", "category", "sign_id", "sign_label",
    "hands_required", "take", "file_path", "camera_index", "camera_name",
    "width", "height", "fps_actual", "n_frames", "duration_s", "codec",
    "orientation", "hand_detect_rate", "both_hands_rate", "mean_brightness", "status",
]

STATUS_KEPT = "kept"
STATUS_NEEDS_RERECORD = "needs_rerecord"

# Display names for the category tiles / pills, and short names for the overview table.
CATEGORY_TITLES = {
    "greetings": "Greetings", "calendar": "Calendar", "family": "Family",
    "wh_questions": "WH-Questions", "numbers": "Numbers", "shapes": "Shapes",
    "colors": "Colors", "word_concepts": "Word Concepts", "alphabet": "Alphabet",
    "all": "All Categories",
}
CATEGORY_SHORT = {
    "greetings": "Greetings", "calendar": "Calendar", "family": "Family",
    "wh_questions": "WH-Q", "numbers": "Numbers", "shapes": "Shapes", "colors": "Colors",
    "word_concepts": "Words", "alphabet": "A–Z",
}


def category_title(category: str) -> str:
    return CATEGORY_TITLES.get(category, category.replace("_", " ").title())


# =============================================================================
# SIGNERS
# =============================================================================
@dataclass
class SignerInfo:
    code: str
    hand: str = ""                  # "left" / "right" / "" (not set yet)
    display_name: str = ""          # optional, e.g. a first name or nickname
    notes: str = ""


def load_signers(path: Path | None = None) -> dict[str, SignerInfo]:
    """Read signers.csv (creating it with S1-S7 if missing)."""
    path = path or SIGNERS_CSV
    signers = {code: SignerInfo(code) for code in SIGNER_CODES}
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                code = (row.get("code") or "").strip().upper()
                if code in signers:
                    hand = (row.get("hand") or "").strip().lower()
                    signers[code] = SignerInfo(code, hand if hand in HANDS else "",
                                               (row.get("display_name_optional") or "").strip(),
                                               (row.get("notes") or "").strip())
    else:
        save_signers(signers, path)
    return signers


def save_signers(signers: dict[str, SignerInfo], path: Path | None = None) -> None:
    """Write signers.csv safely (temporary file + rename)."""
    path = path or SIGNERS_CSV
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SIGNERS_FIELDS)
        for code in SIGNER_CODES:
            s = signers[code]
            w.writerow([s.code, s.hand, s.display_name, s.notes])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# =============================================================================
# SETTINGS FILE
# =============================================================================
def load_settings(path: Path | None = None) -> dict:
    path = path or SETTINGS_JSON
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        settings.update({k: data[k] for k in DEFAULT_SETTINGS if k in data})
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings: dict, path: Path | None = None) -> None:
    path = path or SETTINGS_JSON
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({k: settings[k] for k in DEFAULT_SETTINGS}, f, indent=2)
    os.replace(tmp, path)


# =============================================================================
# METADATA / PROGRESS
# =============================================================================
def read_metadata() -> list[dict[str, str]]:
    """All rows of dataset/metadata.csv (empty list if it does not exist yet)."""
    if not hc.METADATA_CSV.exists():
        return []
    with open(hc.METADATA_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Progress:
    """Kept / flagged takes per (signer, sign_id), built from metadata.csv."""

    def __init__(self, rows: list[dict[str, str]] | None = None) -> None:
        self.reload(rows)

    def reload(self, rows: list[dict[str, str]] | None = None) -> None:
        rows = read_metadata() if rows is None else rows
        self.rows = rows
        self.kept: dict[tuple[str, str], int] = {}
        self.flagged: dict[tuple[str, str], int] = {}
        self.max_take: dict[tuple[str, str], int] = {}    # over ALL rows: never reuse a number
        for r in rows:
            key = (r.get("signer", ""), r.get("sign_id", ""))
            status = r.get("status", "")
            if status == STATUS_KEPT:
                self.kept[key] = self.kept.get(key, 0) + 1
            elif status == STATUS_NEEDS_RERECORD:
                self.flagged[key] = self.flagged.get(key, 0) + 1
            if r.get("take", "").isdigit():
                self.max_take[key] = max(self.max_take.get(key, 0), int(r["take"]))

    def add_kept(self, signer: str, sign_id: str, take: int) -> None:
        """Update counts after a take is kept (metadata row already written)."""
        key = (signer, sign_id)
        self.kept[key] = self.kept.get(key, 0) + 1
        self.max_take[key] = max(self.max_take.get(key, 0), take)

    def kept_for(self, signer: str, sign: hc.Sign) -> int:
        return self.kept.get((signer, sign.sign_id), 0)

    def flagged_for(self, signer: str, sign: hc.Sign) -> int:
        return self.flagged.get((signer, sign.sign_id), 0)

    def done_required(self, signer: str, signs: list[hc.Sign]) -> tuple[int, int]:
        """(kept, required) over a list of signs; extra takes beyond reps don't count."""
        done = sum(min(self.kept_for(signer, s), s.reps) for s in signs)
        return done, sum(s.reps for s in signs)

    def takes(self, signer: str, sign_id: str) -> list[dict[str, str]]:
        """Metadata rows (kept and flagged) for one signer + sign, by take number."""
        out = [r for r in self.rows if r.get("signer") == signer and r.get("sign_id") == sign_id
               and r.get("status") in (STATUS_KEPT, STATUS_NEEDS_RERECORD)]
        return sorted(out, key=lambda r: int(r["take"]) if r.get("take", "").isdigit() else 0)


def set_take_status(file_path: str, status: str) -> bool:
    """Change the status of the metadata row for file_path. Returns True if found.

    The whole file is rewritten to a temporary file and swapped in with
    os.replace, so a crash can never leave a half-written metadata.csv.
    Only the status column changes; the video file is not touched.
    """
    if status not in (STATUS_KEPT, STATUS_NEEDS_RERECORD):
        raise ValueError(f"unknown status {status!r}")
    if not hc.METADATA_CSV.exists():
        return False
    with open(hc.METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    found = False
    for r in rows:
        if r.get("file_path") == file_path:
            r["status"] = status
            found = True
    if not found:
        return False
    tmp = hc.METADATA_CSV.with_name(hc.METADATA_CSV.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, hc.METADATA_CSV)
    return True
