"""
hudyat_common.py - Shared settings and helpers for the HUDYAT Hand Tracker.

record_tool.py, extract_landmarks.py, import_public.py and the training scripts import this file,
so the recording standard, the orientation/handedness convention and the
signs.csv loader are defined in ONE place and can never disagree.
The on-screen interface (display only) lives in hudyat_ui.py.

Script-specific settings (countdown length, window sizes, ...) live at the top
of each script. Settings shared by all scripts live here.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Makes the Windows Media Foundation camera backend open much faster.
# Must be set before the first camera is opened.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# =============================================================================
# PATHS (everything is relative to the folder this file lives in)
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
SIGNS_CSV = PROJECT_DIR / "signs.csv"
REFERENCES_DIR = PROJECT_DIR / "references"
DATASET_DIR = PROJECT_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
LANDMARKS_DIR = DATASET_DIR / "landmarks"
METADATA_CSV = DATASET_DIR / "metadata.csv"
LANDMARKS_INDEX_CSV = DATASET_DIR / "landmarks_index.csv"
SESSION_LOG_DIR = DATASET_DIR / "session_logs"

# =============================================================================
# RECORDING STANDARD
# =============================================================================
TARGET_WIDTH = 1280          # pixels (16:9)
TARGET_HEIGHT = 720          # pixels
TARGET_FPS = 30              # frames per second requested from the camera
CLIP_SECONDS = 3.0           # every take lasts exactly this long (TIME-based, whatever the fps)
CLIP_FRAMES = round(CLIP_SECONDS * TARGET_FPS)   # frames in a take at the target fps (90); informational
MIN_ACCEPTABLE_FPS = 25.0    # warn (never block) if the measured fps is below this
MIN_CLIP_FRAMES = 10         # a take with fewer frames than this is discarded as failed

# Windows camera backends. Which one reaches 30 fps depends on the camera:
#   "msmf"  = Media Foundation (the modern Windows camera API). DEFAULT.
#             On the development laptop: 1280x720 at 29.6 fps.
#   "dshow" = DirectShow (older API). On the same laptop it fell back to
#             uncompressed YUY2 frames at only 8 fps. Some USB webcams do
#             better with it; "record_tool.py --list-cameras --backend dshow" measures it.
# Pick per session with --backend in record_tool.py.
CAMERA_BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}
DEFAULT_CAMERA_BACKEND = "msmf"
DSHOW_FOURCC = "MJPG"        # DirectShow only: ask for compressed frames (needed for 720p30 on many USB webcams)
MAX_CAMERA_INDEX = 5         # camera indexes 0..MAX_CAMERA_INDEX are probed

# Video codecs tried in order: (fourcc, file extension).
# mp4v in .mp4 works on stock OpenCV for Windows; XVID/.avi is the fallback.
VIDEO_CODECS = [("mp4v", ".mp4"), ("XVID", ".avi")]
VIDEO_EXTENSIONS = (".mp4", ".avi")
TMP_TAG = ".tmp"             # unreviewed takes are saved as <name>.tmp.mp4

# =============================================================================
# MEDIAPIPE HAND LANDMARKER
# =============================================================================
NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# =============================================================================
# ORIENTATION AND HANDEDNESS CONVENTION  (read this carefully)
# =============================================================================
# 1. SAVED VIDEO: frames are saved exactly as the camera delivers them,
#    NOT mirrored. The camera looks AT the signer, so the signer's real RIGHT
#    hand appears on the LEFT side of the saved image.
# 2. LANDMARKS: MediaPipe always runs on these same unmirrored frames (live
#    preview, take review and extract_landmarks.py). Landmark x/y coordinates
#    are therefore in unmirrored camera-image coordinates.
# 3. PREVIEW: only the on-screen preview is flipped (selfie / mirror view),
#    for the signer's comfort. Mirroring is never applied to saved data.
# 4. HANDEDNESS: a mirror turns a right hand into something that LOOKS like a
#    left hand, so a handedness label only means something if you know whether
#    the image was mirrored.
#      - The OLD MediaPipe "Hands" solution assumed mirrored (selfie) input and
#        told apps to swap the labels for unmirrored images.
#      - The Tasks API HandLandmarker used here (mediapipe 0.10.21, and
#        @mediapipe/tasks-vision in the browser, same model) labels the hand
#        as it APPEARS in the image you give it. We checked this on MediaPipe's
#        own sample photo: a palm-facing hand with the thumb down and fingers
#        pointing left (anatomically a left hand) is labelled "Left".
#    Because we feed UNMIRRORED camera frames, what a hand looks like in the
#    frame is the signer's real hand, so MediaPipe's label is used AS-IS:
#    the signer's real right hand is labelled "Right".
#    (If we fed it the mirrored preview instead, every label would come out
#    reversed. That is exactly why detection never runs on mirrored frames.)
#
#    The handedness test in the README (raise only your real right hand)
#    verifies this on your camera. If it ever shows the opposite, set
#    SWAP_HANDEDNESS_LABELS = True and re-run extract_landmarks.py --overwrite.
#    Never change it in the middle of data collection.
#
# Left-handed signers: data is NEVER flipped. Their dominant hand is
# simply recorded as "left" in metadata.csv (signing_hand column).
ORIENTATION = "camera_raw_unmirrored"
SWAP_HANDEDNESS_LABELS = False

# =============================================================================
# SIGN WINDOW (trimming idle frames; README section 6.4)
# =============================================================================
# A clip is trimmed to the frames between the FIRST and LAST frame in which a
# hand is detected, plus a small margin, before resampling to 32 frames.
# Hand PRESENCE only, never "no movement": static signs (numbers, letters)
# hold the hand still on purpose. Used by extract_landmarks.py and by the
# recorder's take review, and the browser app must apply the same rule.
TRIM_MARGIN_S = 0.1              # keep this much before the first / after the last hand frame
TRIM_MIN_FRAMES = 8              # never keep fewer frames than this (window widened around its centre)
EDGE_CHECK_S = 0.3               # recorder: warn if a hand is visible in the first / last 0.3 s


def _round_half_up(x: float) -> int:
    """Like JavaScript Math.round (Python's round() rounds .5 to even)."""
    return int(np.floor(x + 0.5))


def sign_window(hand_present: "np.ndarray | list[bool]", fps: float) -> tuple[int, int, bool]:
    """(start, end, trimmed) frame indexes (end INCLUSIVE) of the sign in a clip.

    hand_present[i] is True when at least one hand was detected in frame i.
    No hand at all -> the whole clip (0, T-1, False).
    """
    present = np.asarray(hand_present, dtype=bool)
    total = len(present)
    if total == 0:
        raise ValueError("empty clip")
    if not present.any():
        return 0, total - 1, False
    margin = _round_half_up(TRIM_MARGIN_S * fps)
    first = int(np.argmax(present))
    last = int(total - 1 - np.argmax(present[::-1]))
    start, end = max(0, first - margin), min(total - 1, last + margin)
    need = min(TRIM_MIN_FRAMES, total)
    if end - start + 1 < need:                      # too short: widen around the centre, inside the clip
        centre = (start + end) / 2
        start = _round_half_up(centre - (need - 1) / 2)
        start = max(0, min(start, total - need))
        end = start + need - 1
    return start, end, (start > 0 or end < total - 1)


def hands_at_edges(hand_present: "np.ndarray | list[bool]", fps: float,
                   seconds: float = EDGE_CHECK_S) -> tuple[bool, bool]:
    """(hand visible in the first `seconds`, hand visible in the last `seconds`)."""
    present = np.asarray(hand_present, dtype=bool)
    k = max(1, _round_half_up(seconds * fps))
    return bool(present[:k].any()), bool(present[-k:].any())


# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================
# Detection rates are measured inside the SIGN WINDOW (see above), not over the
# whole clip, because signers start and end each take with their hands down.
HAND_DETECT_WARN_RATE = 0.80     # warn if < 80 % of sign-window frames contain a hand
BOTH_HANDS_WARN_RATE = 0.80      # for 2-hand signs: warn if < 80 % of sign-window frames have 2 hands

# Mean grayscale brightness (0-255) of the frame.
BRIGHTNESS_TOO_DARK = 70
BRIGHTNESS_TOO_BRIGHT = 190

# =============================================================================
# FRAMING GUIDE (fractions of the frame width/height, in the preview)
# =============================================================================
# Head-to-waist area, marked on screen by corner brackets (hudyat_ui.py):
# top of head near the top edge, waist near the bottom edge, body centred.
GUIDE_LEFT = 0.22
GUIDE_RIGHT = 0.78
GUIDE_TOP = 0.04
GUIDE_BOTTOM = 0.98

# The 21-point hand skeleton (same as MediaPipe's HAND_CONNECTIONS).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
]


# =============================================================================
# SIGN LIST
# =============================================================================
@dataclass(frozen=True)
class Sign:
    """One row of signs.csv."""
    category: str
    sign_id: str
    sign_label: str
    reps: int
    hands: int
    mirror_ok: int = 0      # 1 = FSL experts approved a mirrored (left-handed) version


def load_signs(path: Path = SIGNS_CSV) -> list[Sign]:
    """Read signs.csv and validate it (unique sign_ids, hands 1 or 2, reps > 0)."""
    if not path.exists():
        raise FileNotFoundError(f"Sign list not found: {path}")
    signs: list[Sign] = []
    seen: set[str] = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            sign = Sign(
                category=row["category"].strip(),
                sign_id=row["sign_id"].strip(),
                sign_label=row["sign_label"].strip(),
                reps=int(row["reps"]),
                hands=int(row["hands"]),
                mirror_ok=int((row.get("mirror_ok") or "0").strip() or 0),
            )
            if sign.sign_id in seen:
                raise ValueError(f"signs.csv line {line_no}: duplicate sign_id '{sign.sign_id}'")
            if sign.hands not in (1, 2):
                raise ValueError(f"signs.csv line {line_no}: hands must be 1 or 2")
            if sign.mirror_ok not in (0, 1):
                raise ValueError(f"signs.csv line {line_no}: mirror_ok must be 0 or 1")
            if sign.reps < 1:
                raise ValueError(f"signs.csv line {line_no}: reps must be >= 1")
            seen.add(sign.sign_id)
            signs.append(sign)
    return signs


def list_categories(signs: list[Sign]) -> list[str]:
    """Categories in the order they first appear in signs.csv."""
    return list(dict.fromkeys(s.category for s in signs))


# =============================================================================
# MEDIAPIPE
# =============================================================================
@dataclass
class DetectedHand:
    """One detected hand, with the handedness label already corrected."""
    label: str              # "Left" or "Right" = the signer's REAL hand
    score: float            # handedness confidence (0..1)
    landmarks: np.ndarray   # shape (21, 3): normalised x, y, z in UNMIRRORED image coords


def actual_hand_label(mediapipe_label: str) -> str:
    """Convert MediaPipe's handedness label into the signer's real hand.

    See the ORIENTATION AND HANDEDNESS CONVENTION section above: our frames are
    unmirrored and the Tasks HandLandmarker labels hands as they appear, so by
    default the label is returned unchanged.
    """
    if not SWAP_HANDEDNESS_LABELS:
        return mediapipe_label
    return {"Left": "Right", "Right": "Left"}.get(mediapipe_label, mediapipe_label)


def create_hand_landmarker(num_hands: int = NUM_HANDS) -> vision.HandLandmarker:
    """Create a HandLandmarker in VIDEO running mode (works fully offline)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found at {MODEL_PATH}\n"
            "Download it first (see README.md, Setup step 4)."
        )
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker: vision.HandLandmarker, frame_bgr: np.ndarray,
                 timestamp_ms: int) -> list[DetectedHand]:
    """Run MediaPipe on one UNMIRRORED BGR frame and return the hands found.

    timestamp_ms must increase strictly from call to call for the same
    landmarker (a VIDEO-mode requirement).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # MediaPipe expects RGB
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, int(timestamp_ms))
    hands: list[DetectedHand] = []
    for lms, handedness in zip(result.hand_landmarks, result.handedness):
        category = handedness[0]
        hands.append(DetectedHand(
            label=actual_hand_label(category.category_name),
            score=float(category.score),
            landmarks=np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32),
        ))
    return hands


# =============================================================================
# CAMERA
# =============================================================================
def open_camera(index: int, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT,
                fps: int = TARGET_FPS,
                backend: str = DEFAULT_CAMERA_BACKEND) -> cv2.VideoCapture | None:
    """Open a camera and ask for the given resolution/fps.

    The camera may silently deliver something else; always check the real
    size with read_actual_size() and the real fps with measure_fps().
    Returns None if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(index, CAMERA_BACKENDS[backend])
    if not cap.isOpened():
        cap.release()
        return None
    if backend == "dshow":
        # FOURCC must be set before the size for DirectShow to honour it.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*DSHOW_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def read_actual_size(cap: cv2.VideoCapture, attempts: int = 10) -> tuple[int, int] | None:
    """Read real frames and return their (width, height), or None on failure."""
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame.shape[1], frame.shape[0]
    return None


def measure_fps(cap: cv2.VideoCapture, seconds: float = 3.0) -> tuple[float, int]:
    """Read frames for `seconds` and return (measured fps, frames read)."""
    for _ in range(5):                      # warm-up: auto-exposure settles
        cap.read()
    count = 0
    start = time.perf_counter()
    first_t = last_t = start
    while time.perf_counter() - start < seconds:
        ok, _ = cap.read()
        if not ok:
            break
        last_t = time.perf_counter()
        if count == 0:
            first_t = last_t
        count += 1
    if count < 2 or last_t <= first_t:
        return 0.0, count
    return (count - 1) / (last_t - first_t), count


def list_cameras(max_index: int = MAX_CAMERA_INDEX,
                 backend: str = DEFAULT_CAMERA_BACKEND) -> list[int]:
    """Return the camera indexes that open AND deliver a frame."""
    found: list[int] = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, CAMERA_BACKENDS[backend])
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(index)
        cap.release()
    return found


# =============================================================================
# VIDEO WRITER
# =============================================================================
def open_video_writer(path_stem: Path, fps: float,
                      size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path, str]:
    """Open a video writer, trying mp4v/.mp4 first, then XVID/.avi.

    path_stem is the output path WITHOUT extension. Returns
    (writer, actual file path, codec name). Raises RuntimeError if no codec works.
    """
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fourcc, ext in VIDEO_CODECS:
        path = path_stem.with_name(path_stem.name + ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path, fourcc
        writer.release()
        path.unlink(missing_ok=True)
    raise RuntimeError("No working video codec (tried mp4v and XVID).")


# =============================================================================
# IMAGE HELPERS (all on-screen drawing lives in hudyat_ui.py)
# =============================================================================
def mean_brightness(frame_bgr: np.ndarray) -> float:
    """Average grayscale brightness 0-255 (computed on a small copy for speed)."""
    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def fit_into(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize img to fit inside width x height, letterboxed on black."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    scale = min(width / iw, height / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas
