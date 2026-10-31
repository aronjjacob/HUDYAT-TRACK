"""
record_tool.py - HUDYAT FSL data-collection recorder.

Records fixed-length (3-second, time-based) clips of each sign, shows a quality review
after every take, and saves kept clips + metadata.

Start it with no arguments to open the dashboard (all in one window):
    Signers (S1-S7 cards) -> Categories -> Signs -> Recording screen
                                               \\-> saved takes -> replay / flag
    "Progress overview" (button at the top): signers x categories table.

Optional command-line shortcuts:
    python record_tool.py --signer S1                     (opens S1's categories)
    python record_tool.py --signer S1 --category shapes   (straight to recording)
    python record_tool.py --camera 1 --camera-name "Logitech C920" --backend msmf
        (camera settings are remembered in settings.json; type them once)
    --hand left|right   sets the signer's hand in signers.csv
    python record_tool.py --list-cameras [--backend dshow]
        prints every camera index with its resolution and measured fps, then exits
    python record_tool.py --reference [--ref-seconds 4]
        Reference mode (also the "Reference videos" button on the dashboard):
        one 4-second reference take per sign (all 108) for the FSL expert
        validation form. Saved under references/ (see hudyat_reference.py);
        dataset/ is never touched. Review keys: K keep, R redo, N next, B previous.
        V on a done sign records a second version (another way of signing it):
        "<NN> <Sign Label> (2).mp4"; the form then asks which version is correct.

Keys on the recording screen (click the window first so it has keyboard focus):
    SPACE = start the 3-2-1 countdown and record one take
    K     = keep the take just recorded
    R     = retake (discard the take just recorded and redo it)
    S     = skip this sign
    P     = pause / resume
    G     = show / hide the framing guides (corner brackets)
    Esc   = back (recording screen -> sign list -> categories -> signers)
    Q     = save progress and quit (from any screen)
Every on-screen button, card, chip and breadcrumb can also be clicked.

How it stays safe:
  * A separate capture thread reads EVERY camera frame and hands it to a
    separate writer thread during recording. MediaPipe preview detection runs
    in the main thread on a small copy, so it can never cause recorded frames
    to be dropped. The camera stays open while you move between screens.
  * Each take is first written as <name>.tmp.mp4. It is renamed to its final
    name only when you press K, and the metadata.csv row is written only after
    that rename. Retakes, crashes, Ctrl+C and camera disconnects delete the
    .tmp file, so a half-written clip is never left as a "real" clip.
  * Existing clips are never overwritten: take numbers continue from the
    highest take already in metadata.csv (any status) or on disk.
  * You cannot leave the recording screen while a take is recording or waiting
    for Keep / Retake.
  * Flagging a saved take ("needs re-record") only changes its status in
    metadata.csv; no file is ever deleted from the interface.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import hudyat_common as hc
import hudyat_reference as ref
import hudyat_store as store
import hudyat_ui as ui

# =============================================================================
# SETTINGS (recording standard, thresholds and paths are in hudyat_common.py;
# signer codes and data files are in hudyat_store.py)
# =============================================================================
COUNTDOWN_SECONDS = 3
STARTUP_FPS_SECONDS = 2.0          # fps is measured for this long at startup
LIST_FPS_SECONDS = 3.0             # --list-cameras measures each camera this long
CAMERA_FAIL_LIMIT = 30             # this many failed reads in a row = disconnected

PREVIEW_DETECT_WIDTH = 640         # preview MediaPipe runs on a copy this wide
PREVIEW_DETECT_EVERY_N_WHILE_RECORDING = 2   # lighter preview while recording

# On-screen interface (drawn by hudyat_ui.py on a 1280x720 canvas; the window
# is resizable and scales it). All sizes below are in canvas pixels.
WINDOW_NAME = "HUDYAT Recorder"
GUIDES_ON_BY_DEFAULT = True        # corner brackets (toggle with G)
LIGHT_UI_WHILE_RECORDING = True    # during the 3 s take: only REC pill + progress, no preview detection
REVIEW_FRAME_WIDTH = 960           # replay frames are kept in memory at this width
TITLE_MAX_WIDTH = 700              # the sign name shrinks to fit this width
PROGRESS_W = 340                   # kept-takes progress bar under the sign name
CARD_W = 330                       # take-quality card
KEEP_W, KEEP_H = 196, 58           # Keep button (the retake button is KEEP_H round)
REC_BAR_W = 320                    # recording progress bar
TOAST_SECONDS = 4.0                # how long a status message stays visible
REF_SIGN_COLS = 3                  # reference videos: signs per row in a category

METADATA_FIELDS = store.METADATA_FIELDS


# =============================================================================
# CAPTURE AND WRITING THREADS
# =============================================================================
class RecordingJob:
    """Writes every frame captured during duration_s seconds to a video file,
    on its own thread.

    Recording is TIME-based: a frame belongs to the take if it was captured
    less than duration_s after the first frame. A 30 fps camera gives 90
    frames, a 15 fps camera 45; either way the take covers the same 3 s.

    The capture thread calls push() for every frame; push() only puts the frame
    on a queue (instant), so slow disk/encoding can never stall the camera.
    """

    def __init__(self, writer: cv2.VideoWriter, path: Path, codec: str, duration_s: float) -> None:
        self.writer = writer
        self.path = path
        self.codec = codec
        self.duration = duration_s
        self.timestamps: list[float] = []      # capture time of each frame
        self.frames_captured = 0
        self.frames_written = 0
        self.failed = False
        self.error = ""
        self.done = threading.Event()          # set when the file is closed
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def push(self, frame: np.ndarray, timestamp: float) -> bool:
        """Queue one frame. Returns True once the take's time is up (that
        frame, captured at or after the end, is not part of the take)."""
        if self.timestamps and timestamp - self.timestamps[0] >= self.duration:
            self._queue.put(None)              # sentinel: no more frames
            return True
        self._queue.put(frame)
        self.timestamps.append(timestamp)
        self.frames_captured += 1
        return False

    def elapsed_fraction(self, now: float) -> float:
        """0..1 progress of the take, for the on-screen bar."""
        if not self.timestamps:
            return 0.0
        return min(1.0, (now - self.timestamps[0]) / self.duration)

    def abort(self, reason: str) -> None:
        """Stop the job; the (incomplete) file must then be deleted."""
        self.failed = True
        self.error = reason
        self._queue.put(None)

    def _write_loop(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                if not self.failed:
                    self.writer.write(frame)
                    self.frames_written += 1
        except Exception as exc:              # disk full, etc.
            self.failed = True
            self.error = f"Video write error: {exc}"
        finally:
            self.writer.release()              # file is fully closed before done is set
            self.done.set()

    def measured_fps(self) -> float:
        """Real fps from the capture timestamps of this clip."""
        if len(self.timestamps) < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / span if span > 0 else 0.0

    def duration_s(self) -> float:
        """Real length of the take: frames / measured fps."""
        fps = self.measured_fps()
        return len(self.timestamps) / fps if fps else 0.0


class CameraWorker(threading.Thread):
    """Reads camera frames continuously; feeds the active RecordingJob."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        super().__init__(daemon=True)
        self.cap = cap
        self.error: str | None = None
        self.frame_id = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest: np.ndarray | None = None
        self._times: deque[float] = deque(maxlen=60)
        self._job: RecordingJob | None = None

    def run(self) -> None:
        fails = 0
        while not self._stop_event.is_set():
            ok, frame = self.cap.read()
            ts = time.perf_counter()
            if not ok or frame is None:
                fails += 1
                if fails >= CAMERA_FAIL_LIMIT:
                    self.error = "Camera disconnected or stopped delivering frames."
                    break
                time.sleep(0.01)
                continue
            fails = 0
            with self._lock:
                self._latest = frame
                self.frame_id += 1
                self._times.append(ts)
                job = self._job
            # Every captured frame goes to the recording job (no skipping).
            if job is not None and job.push(frame, ts):
                with self._lock:
                    self._job = None
        # Leaving the loop while recording: abort so the partial file is removed.
        with self._lock:
            job, self._job = self._job, None
        if job is not None:
            job.abort(self.error or "Recording stopped.")

    def start_job(self, job: RecordingJob) -> None:
        with self._lock:
            self._job = job

    def latest(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            return self._latest, self.frame_id

    def fps(self) -> float:
        with self._lock:
            times = list(self._times)
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=3.0)


# =============================================================================
# METADATA, FILE NAMES, RESUME
# =============================================================================
def read_metadata() -> list[dict[str, str]]:
    """All rows of dataset/metadata.csv (empty list if it does not exist yet)."""
    if not hc.METADATA_CSV.exists():
        return []
    with open(hc.METADATA_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_metadata(row: dict[str, object]) -> None:
    """Append one row and force it to disk (flush + fsync)."""
    new_file = not hc.METADATA_CSV.exists() or hc.METADATA_CSV.stat().st_size == 0
    with open(hc.METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def clip_dir(sign: hc.Sign, signer: str) -> Path:
    return hc.RAW_DIR / sign.category / sign.sign_id / signer


def clip_stem(sign: hc.Sign, signer: str, take: int) -> str:
    """e.g. circle_S1_07"""
    return f"{sign.sign_id}_{signer}_{take:02d}"


def takes_on_disk(sign: hc.Sign, signer: str) -> set[int]:
    """Take numbers of finished clip files already in the clip folder."""
    folder = clip_dir(sign, signer)
    prefix = f"{sign.sign_id}_{signer}_"
    takes: set[int] = set()
    if folder.exists():
        for p in folder.iterdir():
            if p.suffix.lower() in hc.VIDEO_EXTENSIONS and hc.TMP_TAG not in p.name \
                    and p.stem.startswith(prefix) and p.stem[len(prefix):].isdigit():
                takes.add(int(p.stem[len(prefix):]))
    return takes


def cleanup_tmp_files(signer: str) -> list[Path]:
    """Delete unreviewed/half-written .tmp clips left by a crash (this signer only)."""
    removed: list[Path] = []
    if hc.RAW_DIR.exists():
        for p in hc.RAW_DIR.glob(f"*/*/{signer}/*{hc.TMP_TAG}.*"):
            try:
                p.unlink()
                removed.append(p)
            except OSError:
                pass
    return removed


# =============================================================================
# SESSION LOG
# =============================================================================
class SessionLog:
    """dataset/session_logs/<signer>_<YYYY-MM-DD_HHMM>.txt, written as we go."""

    def __init__(self, signer: str) -> None:
        hc.SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.start = datetime.now()
        self.path = hc.SESSION_LOG_DIR / f"{signer}_{self.start:%Y-%m-%d_%H%M}.txt"
        self.counts = {"kept": 0, "retaken": 0, "skipped": 0}
        self._closed = False
        self.write(f"SESSION START {self.start:%Y-%m-%d %H:%M:%S}")

    def write(self, message: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%H:%M:%S}] {message}\n")

    def close(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        end = datetime.now()
        self.write(f"SESSION END {end:%Y-%m-%d %H:%M:%S} ({reason})")
        self.write(f"Duration: {str(end - self.start).split('.')[0]}")
        self.write(f"Kept: {self.counts['kept']}  Retaken: {self.counts['retaken']}  "
                   f"Skipped: {self.counts['skipped']}")


# =============================================================================
# TAKE ANALYSIS
# =============================================================================
@dataclass
class ClipReview:
    frames: list[np.ndarray]     # mirrored, REVIEW_FRAME_WIDTH wide, for replay only
    hands: list[list[hc.DetectedHand]]   # hands found in each frame (drawn at replay)
    n_frames_read: int
    hand_detect_rate: float      # 0..1, inside the sign window
    both_hands_rate: float       # 0..1, inside the sign window
    mean_brightness: float
    warnings: list[str]
    window_start: int = 0        # sign window kept by the trim (frame indexes, end inclusive)
    window_end: int = 0
    fps: float = float(hc.TARGET_FPS)
    face_visible_rate: float | None = None   # 0..1 over the hands-down frames; None = not checked


def analyze_clip(path: Path, sign: hc.Sign, fps_actual: float, expected_frames: int,
                 clip_seconds: float = hc.CLIP_SECONDS) -> ClipReview:
    """Re-open the SAVED file (proves it is readable), run MediaPipe on every
    full-resolution frame and compute the quality numbers.

    Uses a fresh VIDEO-mode HandLandmarker and unmirrored frames, the same way
    extract_landmarks.py does, so the numbers match the later extraction.

    On the frames without a hand, a face model checks that the whole face is in
    the picture (nothing about the face is saved).
    """
    cap = cv2.VideoCapture(str(path))
    file_fps = cap.get(cv2.CAP_PROP_FPS) or hc.TARGET_FPS
    landmarker = hc.create_hand_landmarker()
    face_landmarker = hc.create_face_landmarker()          # None if the model is not downloaded
    frames: list[np.ndarray] = []
    hands_per_frame: list[list[hc.DetectedHand]] = []
    face_visible: list[bool | None] = []
    n = 0
    brightness_sum = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = round(n * 1000 / file_fps)
            hands = hc.detect_hands(landmarker, frame, ts)
            face_visible.append(hc.face_fully_visible(face_landmarker, frame, ts)
                                if face_landmarker is not None and not hands else None)
            n += 1
            brightness_sum += hc.mean_brightness(frame)
            display = cv2.flip(frame, 1)                       # mirrored COPY, for replay only
            h = int(display.shape[0] * REVIEW_FRAME_WIDTH / display.shape[1])
            frames.append(cv2.resize(display, (REVIEW_FRAME_WIDTH, h), interpolation=cv2.INTER_AREA))
            hands_per_frame.append(hands)
    finally:
        cap.release()
        landmarker.close()
        if face_landmarker is not None:
            face_landmarker.close()

    q = take_quality([len(h) for h in hands_per_frame], file_fps, sign, expected_frames, fps_actual, clip_seconds,
                     face_visible if face_landmarker is not None else None)
    if face_landmarker is None:
        q.warnings.append("Face check skipped: models\\face_landmarker.task not found (README Setup step 4).")
    return ClipReview(frames, hands_per_frame, n, q.hand_detect_rate, q.both_hands_rate,
                      brightness_sum / max(n, 1), q.warnings, q.window_start, q.window_end, file_fps,
                      q.face_visible_rate)


@dataclass
class TakeQuality:
    hand_detect_rate: float      # inside the sign window
    both_hands_rate: float       # inside the sign window
    window_start: int            # sign window (hudyat_common.sign_window), end inclusive
    window_end: int
    warnings: list[str]
    face_visible_rate: float | None = None   # over the hands-down frames; None = not checked


def take_quality(hand_counts: list[int], fps: float, sign: hc.Sign, expected_frames: int,
                 fps_actual: float, clip_seconds: float = hc.CLIP_SECONDS,
                 face_visible: list[bool | None] | None = None) -> TakeQuality:
    """Quality numbers + warnings for one take, from the number of hands per frame.

    The detection rates are measured inside the sign window, exactly as
    extract_landmarks.py does, because signers start and end with hands down.
    face_visible: per frame, whether the whole face was in the picture (None =
    not checked, e.g. a hand was in the picture); None for the whole take = no face check.
    """
    counts = np.asarray(hand_counts)
    n = len(counts)
    warnings: list[str] = []
    if n == 0:
        return TakeQuality(0.0, 0.0, 0, 0, ["Saved file has no frames. Retake."])
    start, end, _ = hc.sign_window(counts > 0, fps)
    window = counts[start:end + 1]
    hdr, bhr = float((window >= 1).mean()), float((window >= 2).mean())
    if n != expected_frames:
        warnings.append(f"Saved file has {n} frames, {expected_frames} were recorded. Retake.")
    if not (counts > 0).any():
        warnings.append("No hands detected at all. Retake suggested.")
    elif hdr < hc.HAND_DETECT_WARN_RATE:
        warnings.append(f"Low hand detection ({hdr:.0%}). Retake suggested.")
    if sign.hands == 2 and bhr < hc.BOTH_HANDS_WARN_RATE:
        warnings.append(f"2-hand sign: both hands only {bhr:.0%}. Retake suggested.")
    # The trim needs idle frames at both ends (README "Recording rule").
    at_start, at_end = hc.hands_at_edges(counts > 0, fps)
    if at_start or at_end:
        where = "start and end" if at_start and at_end else "start" if at_start else "end"
        warnings.append(f"Hand visible at the {where}. Start and end with hands down.")
    if 0 < fps_actual < hc.MIN_ACCEPTABLE_FPS:
        warnings.append(f"Low fps during take ({fps_actual:.1f}); still a full {clip_seconds:g} s.")
    checked = [v for v in face_visible or [] if v is not None]
    face_rate = sum(checked) / len(checked) if checked else None
    if face_rate is not None and face_rate < hc.FACE_VISIBLE_WARN_RATE:
        warnings.append(f"Face not fully visible ({face_rate:.0%}). Keep the forehead and chin in the picture.")
    return TakeQuality(hdr, bhr, start, end, warnings, face_rate)


# =============================================================================
# SAVED-TAKE PLAYER (Takes screen -> click a take)
# =============================================================================
class TakePlayer:
    """Loops one saved clip at its own frame rate (read-only; never modifies it)."""

    def __init__(self, row: dict[str, str]) -> None:
        self.row = row
        self.path = hc.PROJECT_DIR / row.get("file_path", "")
        self.cap: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.interval = 1 / hc.TARGET_FPS
        self.last = 0.0
        if self.path.exists():
            cap = cv2.VideoCapture(str(self.path))
            if cap.isOpened():
                self.cap = cap
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.interval = 1 / fps if fps and fps > 1 else 1 / hc.TARGET_FPS

    def current(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        now = time.perf_counter()
        if self.frame is None or now - self.last >= self.interval:
            self.last = now
            ok, frame = self.cap.read()
            if not ok:                                   # end of clip: loop
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if ok:
                self.frame = frame
        return self.frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None


# =============================================================================
# THE APP: dashboard screens + recording screen in one window
# =============================================================================
class App:
    """One window, several screens:

        HOME (signer cards) -> CATEGORIES -> SIGNS -> RECORD
                                                   -> TAKES -> PLAYER
        HOME / CATEGORIES / SIGNS -> OVERVIEW (signers x categories)
        HOME -> REF_LIST (reference-video categories) -> REF_SIGNS -> RECORD (reference mode)

    The camera thread runs the whole time, so moving between screens never
    reopens the camera. On the RECORD screen the recorder works exactly as
    before: LIVE -> COUNTDOWN -> RECORDING -> REVIEW -> LIVE ... -> FINISHED.
    In reference mode (self.ref_mode) the same RECORD screen records one
    take per sign into references/ instead of dataset/.
    """

    def __init__(self, all_signs: list[hc.Sign], cap: cv2.VideoCapture,
                 frame_size: tuple[int, int], settings: dict) -> None:
        self.all_signs = all_signs
        self.categories = hc.list_categories(all_signs)
        self.cap = cap
        self.width, self.height = frame_size
        self.settings = settings                        # camera, camera_name, backend
        self.signers = store.load_signers()
        self.progress = store.Progress()
        self.worker = CameraWorker(cap)
        self.landmarker = hc.create_hand_landmarker()
        self.camera_lines: list[str] = []                # written at the top of every session log
        self.totals = {"kept": 0, "retaken": 0, "skipped": 0}

        # --- navigation ---
        self.screen = "HOME"
        self.modal: str | None = None                    # "hand": ask a signer's hand
        self.modal_signer = ""
        self.pending_category: str | None = None         # --category waiting for the hand answer
        self.overview_return = "HOME"
        self.signer: str | None = None
        self.category: str | None = None                 # category code, or "all"
        self.signs: list[hc.Sign] = []                   # signs of the open category
        self.view_sign: hc.Sign | None = None            # TAKES / PLAYER
        self.take_rows: list[dict[str, str]] = []
        self.player: TakePlayer | None = None
        self.scroll: dict[str, int] = {}                 # list name -> first visible row
        self.log: SessionLog | None = None

        # --- recorder (RECORD screen) ---
        self.state = "LIVE"
        self.paused = False
        self.sign_index = 0
        self.take = 1
        self.job: RecordingJob | None = None
        self.review: ClipReview | None = None
        self.review_start = 0.0
        self.countdown_start = 0.0
        self.session_kept = 0
        self.standard_warnings: list[str] = []
        self.writer_fps: float = float(hc.TARGET_FPS)
        self.hands: list[hc.DetectedHand] = []
        self.brightness = 0.0
        self._last_frame_id = -1
        self._detect_counter = 0
        self._t0 = time.perf_counter()
        self._last_ts_ms = -1

        # --- reference mode (REF_LIST + REF_SIGNS + RECORD; files under references/) ---
        self.ref_mode = False
        self.ref_seconds = ref.REF_SECONDS
        self.refs: list[ref.RefSign] = []
        self.ref_done: set[str] = set()
        self.ref_v2: set[str] = set()                    # signs that also have a version 2
        self.ref_version = 1                             # version the take being recorded is for
        self.ref_hand = ""
        self.ref_folder = ""                             # category folder open in REF_SIGNS / RECORD

        # --- interface ---
        self.message = ""
        self.message_color: tuple[int, int, int] = ui.WHITE
        self.message_until = 0.0
        self.show_guides = GUIDES_ON_BY_DEFAULT
        self.hits = ui.HitMap()                          # clickable areas of the current frame
        self.scroll_regions: list[tuple[int, int, int, int, str]] = []
        self._clicked: str | None = None
        self._wheel: list[tuple[str, int]] = []
        self._mouse = (-1, -1)
        self._last_canvas: np.ndarray | None = None
        self._chips_follow = False                       # scroll the chips to the current sign

    # ----- startup ----------------------------------------------------------
    def start_camera(self) -> None:
        """Start the capture thread, measure real fps, and build warnings."""
        self.worker.start()
        time.sleep(STARTUP_FPS_SECONDS)
        fps = self.worker.fps()
        if (self.width, self.height) != (hc.TARGET_WIDTH, hc.TARGET_HEIGHT):
            self.standard_warnings.append(
                f"CAMERA GIVES {self.width}x{self.height}, NOT {hc.TARGET_WIDTH}x{hc.TARGET_HEIGHT}!")
        if fps < hc.MIN_ACCEPTABLE_FPS:
            self.standard_warnings.append(f"LOW FPS: {fps:.1f} (< {hc.MIN_ACCEPTABLE_FPS:.0f})")
        # Tag the video with the fps the camera really runs at if it is far from 30,
        # so the saved clips play back at real speed.
        self.writer_fps = float(hc.TARGET_FPS) if abs(fps - hc.TARGET_FPS) <= 2 or fps <= 0 else round(fps, 1)
        info = (f"Camera {self.settings['camera']} ({self.settings['camera_name']}, {self.settings['backend']}): "
                f"{self.width}x{self.height} @ {fps:.1f} fps measured")
        print(info)
        self.camera_lines = [info, f"Video file fps tag: {self.writer_fps}",
                             *("WARNING: " + w for w in self.standard_warnings)]
        for w in self.standard_warnings:
            print("WARNING:", w)

    # ----- helpers ----------------------------------------------------------
    @property
    def sign(self) -> hc.Sign | None:
        return self.signs[self.sign_index] if self.sign_index < len(self.signs) else None

    @property
    def hand(self) -> str:
        return self.signers[self.signer].hand if self.signer else ""

    def kept_for(self, sign: hc.Sign) -> int:
        return self.progress.kept_for(self.signer or "", sign)

    def category_signs(self, category: str) -> list[hc.Sign]:
        return select_signs(self.all_signs, category)

    def set_message(self, text: str, color: tuple[int, int, int] = ui.WHITE) -> None:
        """Short status message shown above the caption for TOAST_SECONDS."""
        self.message, self.message_color = text, color
        self.message_until = time.perf_counter() + TOAST_SECONDS

    def log_write(self, text: str) -> None:
        if self.ref_mode:                                # reference mode: references/reference_log.txt
            ref.log(text)
        elif self.log is not None:
            self.log.write(text)

    @property
    def clip_seconds(self) -> float:
        return self.ref_seconds if self.ref_mode else hc.CLIP_SECONDS

    @property
    def current_ref(self) -> ref.RefSign | None:
        return self.refs[self.sign_index] if self.ref_mode and self.sign_index < len(self.refs) else None

    def is_complete(self, sign: hc.Sign) -> bool:
        """Dataset: all takes kept. Reference mode: the reference video exists."""
        if self.ref_mode:
            return sign.sign_id in self.ref_done
        return self.kept_for(sign) >= sign.reps

    # ----- reference mode -----------------------------------------------------
    def enter_reference(self) -> None:
        """'Reference videos' button / --reference: ask which hand signs (anyone can sign)."""
        if not self.leave_record_ok():
            return
        try:
            self.refs = ref.reference_signs(self.all_signs)
        except ValueError as exc:
            self.set_message(str(exc), ui.DANGER)
            return
        state = ref.load_state()
        self.ref_hand = state.get("hand", "") if state.get("hand") in store.HANDS else ""
        self.modal = "ref_setup"

    def start_reference(self, hand: str) -> None:
        """Hand button clicked: switch to reference mode and show the sign list."""
        self.modal = None
        self.ref_mode = True
        self.ref_hand = hand
        state = ref.load_state()
        state["hand"] = hand
        ref.save_state(state)
        self.signs = [r.sign for r in self.refs]
        last = state.get("last_sign_id")
        self.sign_index = next((r.index for r in self.refs if r.sign.sign_id == last), 0)
        self.ref_done = ref.done_ids(self.refs)
        self.ref_v2 = ref.version2_ids(self.refs)
        ref.write_form_files(self.refs)                  # form material always matches the current list
        ref.log(f"REFERENCE SESSION hand={self.ref_hand} seconds={self.ref_seconds:g} "
                f"done={len(self.ref_done)}/{len(self.refs)}")
        for line in self.camera_lines:
            ref.log(line)
        self.open_ref_list()

    def open_ref_list(self) -> None:
        """Category tiles; the category of the last sign used is outlined."""
        self.ref_done = ref.done_ids(self.refs)
        self.ref_v2 = ref.version2_ids(self.refs)
        self.screen = "REF_LIST"
        if self.sign_index < len(self.refs):
            self.ref_folder = self.refs[self.sign_index].folder

    def open_ref_signs(self, folder: str) -> None:
        """The signs of one category, scrolled to the last sign used."""
        self.ref_done = ref.done_ids(self.refs)
        self.ref_v2 = ref.version2_ids(self.refs)
        self.ref_folder = folder
        self.screen = "REF_SIGNS"
        items = self.ref_folder_indices(folder)
        pos = items.index(self.sign_index) if self.sign_index in items else 0
        self.scroll["ref_signs"] = max(0, pos // REF_SIGN_COLS - 1)

    def ref_folder_indices(self, folder: str) -> list[int]:
        """Indexes (into self.refs / self.signs) of the signs in one category folder."""
        return [r.index for r in self.refs if r.folder == folder]

    def exit_reference(self) -> None:
        if not self.ref_mode:
            return
        ref.log(f"REFERENCE SESSION END done={len(self.ref_done)}/{len(self.refs)}")
        self.ref_mode = False
        self.signs = self.category_signs(self.category) if self.category else []
        self.sign_index = 0

    def ref_step(self, step: int) -> None:
        """N / B: next or previous sign (discarding a take that is waiting for review)."""
        if self.review is not None or self.job is not None:
            self.discard_pending_take()
            self.log_write(f"DISCARDED take of {self.sign.sign_id if self.sign else ''} ({'next' if step > 0 else 'previous'})")
        new = self.sign_index + step
        if new < 0:
            self.set_message("This is the first sign.", ui.WARNING)
            new = 0
        if new >= len(self.signs):
            self.sign_index = len(self.signs)
            self.state = "FINISHED"
            return
        self.select_record_sign(new)

    def keep_reference(self) -> None:
        """K in reference mode: export .mp4 + .gif, update the manifest and the form files."""
        job, review, r, version = self.job, self.review, self.current_ref, self.ref_version
        assert job is not None and review is not None and r is not None
        self.render_busy("Saving reference video (MP4 + GIF)...")
        stem = ref.version_stem(r, version)
        try:
            result = ref.keep_take(job.path, r, self.refs, window_start=review.window_start,
                                   window_end=review.window_end, fps=review.fps, n_frames=review.n_frames_read,
                                   hand_detect_rate=review.hand_detect_rate,
                                   hand=self.ref_hand, version=version)
        except (RuntimeError, OSError) as exc:            # stays in review: K again, or R
            self.set_message(f"Could not save: {exc}", ui.DANGER)
            ref.log(f"SAVE FAILED {stem}: {exc}")
            return
        self.job, self.review = None, None
        (self.ref_done if version == 1 else self.ref_v2).add(r.sign.sign_id)
        self.session_kept += 1
        self.totals["kept"] += 1
        gif = f"GIF {result.gif_bytes / 1024:.0f} KB, {result.gif_settings}"
        if not result.gif_under_limit:
            self.set_message(f"Saved {stem}, but the GIF is still over 2 MB ({gif})", ui.WARNING)
        else:
            old = " · old version moved to references\\_old" if result.archived else ""
            self.set_message(f"Saved {stem} ({result.seconds:.1f} s · {gif}){old}", ui.SUCCESS)
        self.ref_step(1)

    # ----- navigation -------------------------------------------------------
    def select_signer(self, code: str) -> None:
        """Signer card clicked: ask the hand once if unknown, then show categories."""
        if not self.signers[code].hand:
            self.modal, self.modal_signer = "hand", code
            return
        self._enter_signer(code)
        self.screen = "CATEGORIES"

    def set_hand(self, code: str, hand: str) -> None:
        self.signers[code].hand = hand
        store.save_signers(self.signers)
        self.modal = None
        self._enter_signer(code)
        self.screen = "CATEGORIES"
        if self.pending_category:                       # command-line shortcut continues
            self.open_category(self.pending_category)
            self.open_record(self.first_incomplete_index())
            self.pending_category = None

    def _enter_signer(self, code: str) -> None:
        """Switch the active signer: own session log, leftover .tmp clean-up."""
        if code == self.signer:
            return
        if self.log is not None:
            self.log.close("switched signer")
        self.signer = code
        self.log = SessionLog(code)
        self.log.write(f"signer={code} hand={self.hand} orientation={hc.ORIENTATION}")
        for line in self.camera_lines:
            self.log.write(line)
        for p in cleanup_tmp_files(code):
            print(f"Removed leftover unsaved take: {p.relative_to(hc.PROJECT_DIR)}")

    def open_category(self, category: str) -> None:
        self.category = category
        self.signs = self.category_signs(category)
        self.scroll["signs"] = 0
        self.screen = "SIGNS"
        self.log_write(f"OPEN category {category}")

    def first_incomplete_index(self) -> int:
        for i, s in enumerate(self.signs):
            if self.kept_for(s) < s.reps:
                return i
        return 0

    def open_record(self, index: int) -> None:
        """Open the recording screen at a given sign of the open category."""
        if not self.signs:
            return
        self.screen = "RECORD"
        self.paused = False
        self.select_record_sign(index)

    def select_record_sign(self, index: int) -> None:
        """Jump to a sign on the recording screen (chips / sign list)."""
        self.sign_index = max(0, min(index, len(self.signs) - 1))
        sign = self.signs[self.sign_index]
        self.state = "LIVE"
        self._chips_follow = True
        if self.ref_mode:                                # remembered for the next session
            self.ref_folder = self.refs[self.sign_index].folder
            state = ref.load_state()
            state["last_sign_id"] = sign.sign_id
            ref.save_state(state)
            self.log_write(f"SIGN {sign.sign_id} ({'done' if self.is_complete(sign) else 'pending'})")
            return
        self.take = self.next_free_take(sign)
        self.log_write(f"SIGN {sign.sign_id} ({self.kept_for(sign)}/{sign.reps} kept)")

    def open_takes(self, sign: hc.Sign) -> None:
        self.view_sign = sign
        self.take_rows = self.progress.takes(self.signer or "", sign.sign_id)
        self.scroll["takes"] = 0
        self.screen = "TAKES"

    def open_player(self, row_index: int) -> None:
        if 0 <= row_index < len(self.take_rows):
            self.close_player()
            self.player = TakePlayer(self.take_rows[row_index])
            self.screen = "PLAYER"

    def close_player(self) -> None:
        if self.player is not None:
            self.player.release()
        self.player = None

    def open_overview(self) -> None:
        self.progress.reload()                           # always fresh from metadata.csv
        self.overview_return = self.screen
        self.screen = "OVERVIEW"

    def leave_record_ok(self) -> bool:
        """The recording screen can only be left when no take is in progress."""
        if self.screen != "RECORD":
            return True
        if self.state == "RECORDING":
            self.set_message("Wait for the take to finish.", ui.WARNING)
            return False
        if self.state == "REVIEW":
            self.set_message("Keep or retake this take first.", ui.WARNING)
            return False
        self.state = "LIVE"                              # cancels a running countdown
        self.paused = False
        return True

    def navigate(self, screen: str) -> None:
        if not self.leave_record_ok():
            return
        if self.screen == "PLAYER" and screen != "PLAYER":
            self.close_player()
        if screen == "REF_LIST":
            self.open_ref_list()
            return
        if screen == "REF_SIGNS":
            self.open_ref_signs(self.ref_folder)
            return
        if self.ref_mode:                                # any other screen ends reference mode
            self.exit_reference()
        if screen == "OVERVIEW":
            self.open_overview()
            return
        if screen in ("HOME", "CATEGORIES"):
            self.progress.reload()
        if screen == "TAKES" and self.view_sign is not None:
            self.open_takes(self.view_sign)
            return
        self.screen = screen

    def go_back(self) -> None:
        back = {"RECORD": "REF_SIGNS" if self.ref_mode else "SIGNS", "PLAYER": "TAKES", "TAKES": "SIGNS",
                "SIGNS": "CATEGORIES", "CATEGORIES": "HOME", "OVERVIEW": self.overview_return,
                "REF_SIGNS": "REF_LIST", "REF_LIST": "HOME"}
        if self.screen in back:
            self.navigate(back[self.screen])

    def toggle_flag(self) -> None:
        """Flag a kept take as 'needs re-record' (or undo). Files are never deleted."""
        if self.player is None:
            return
        row = self.player.row
        new = store.STATUS_NEEDS_RERECORD if row.get("status") == store.STATUS_KEPT else store.STATUS_KEPT
        if store.set_take_status(row["file_path"], new):
            row["status"] = new
            self.progress.reload()
            self.log_write(f"STATUS {row['file_path']} -> {new}")
            self.set_message("Flagged: needs re-record (does not count any more)" if new != store.STATUS_KEPT
                             else "Unflagged: counts as kept again", ui.WARNING if new != store.STATUS_KEPT else ui.SUCCESS)
        else:
            self.set_message("Could not find this take in metadata.csv", ui.DANGER)

    # ----- recorder ---------------------------------------------------------
    def go_to_next_incomplete(self) -> None:
        """Move to the first sign (from the current index) that still needs takes."""
        while self.sign is not None and self.kept_for(self.sign) >= self.sign.reps:
            self.sign_index += 1
        sign = self.sign
        if sign is None:
            self.state = "FINISHED"
            self.log_write("End of the sign list reached.")
            return
        self.take = self.next_free_take(sign)
        self.state = "LIVE"
        self._chips_follow = True

    def next_free_take(self, sign: hc.Sign, start: int | None = None) -> int:
        """Next take number that is not in metadata (any status) and has no file on disk."""
        on_disk = takes_on_disk(sign, self.signer or "")
        take = start or max([self.progress.max_take.get((self.signer or "", sign.sign_id), 0), *on_disk, 0]) + 1
        while take in on_disk:
            take += 1
        return take

    def start_recording(self) -> None:
        sign = self.sign
        if self.ref_mode:                                # raw reference take: references/_tmp/
            assert sign is not None
            stem = ref.tmp_dir() / (sign.sign_id + hc.TMP_TAG)
        else:
            assert sign is not None and self.signer is not None
            folder = clip_dir(sign, self.signer)
            folder.mkdir(parents=True, exist_ok=True)
            stem = folder / (clip_stem(sign, self.signer, self.take) + hc.TMP_TAG)
        try:
            writer, path, codec = hc.open_video_writer(stem, self.writer_fps, (self.width, self.height))
        except RuntimeError as exc:
            self.set_message(str(exc), ui.DANGER)
            self.state = "LIVE"
            return
        self.job = RecordingJob(writer, path, codec, self.clip_seconds)
        self.worker.start_job(self.job)
        self.state = "RECORDING"

    def finish_recording(self) -> None:
        """Called once the writer thread has closed the file."""
        job = self.job
        assert job is not None and self.sign is not None
        if job.failed or job.frames_written < hc.MIN_CLIP_FRAMES or job.frames_written != job.frames_captured:
            job.path.unlink(missing_ok=True)
            self.job = None
            self.set_message(f"Take failed and was discarded: {job.error or 'incomplete'}", ui.DANGER)
            self.log_write(f"FAILED take discarded ({job.error})")
            self.state = "LIVE"
            return
        self.render_busy("Analyzing take...")
        self.review = analyze_clip(job.path, self.sign, job.measured_fps(), job.frames_written, self.clip_seconds)
        self.review_start = time.perf_counter()
        self.state = "REVIEW"

    def discard_pending_take(self) -> None:
        if self.job is not None:
            self.job.path.unlink(missing_ok=True)
        self.job = None
        self.review = None

    def keep_take(self) -> None:
        if self.ref_mode:
            self.keep_reference()
            return
        job, review, sign, signer = self.job, self.review, self.sign, self.signer
        assert job is not None and review is not None and sign is not None and signer is not None
        # Never overwrite: if the final name is taken somehow, use the next free take.
        take = self.take
        final = clip_dir(sign, signer) / (clip_stem(sign, signer, take) + job.path.suffix)
        while final.exists() or take in takes_on_disk(sign, signer):
            take += 1
            final = final.with_name(clip_stem(sign, signer, take) + job.path.suffix)
        os.replace(job.path, final)          # tmp -> final (file is already closed)

        fps_actual = job.measured_fps()
        n_frames = job.frames_written
        append_metadata({                    # only AFTER the clip is saved and closed
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "signer": signer,
            "signing_hand": self.hand,
            "category": sign.category,
            "sign_id": sign.sign_id,
            "sign_label": sign.sign_label,
            "hands_required": sign.hands,
            "take": take,
            "file_path": final.relative_to(hc.PROJECT_DIR).as_posix(),
            "camera_index": self.settings["camera"],
            "camera_name": self.settings["camera_name"],
            "width": self.width,
            "height": self.height,
            "fps_actual": f"{fps_actual:.2f}",
            "n_frames": n_frames,
            "duration_s": f"{job.duration_s():.3f}",
            "codec": job.codec,
            "orientation": hc.ORIENTATION,
            "hand_detect_rate": f"{review.hand_detect_rate:.3f}",
            "both_hands_rate": f"{review.both_hands_rate:.3f}",
            "mean_brightness": f"{review.mean_brightness:.1f}",
            "status": "kept",
        })
        self.progress.add_kept(signer, sign.sign_id, take)
        self.progress.reload()               # keep the dashboard rows in sync with the file
        self.session_kept += 1
        self.totals["kept"] += 1
        if self.log is not None:
            self.log.counts["kept"] += 1
        self.log_write(f"KEPT {final.name} hand_detect={review.hand_detect_rate:.0%} "
                       f"both={review.both_hands_rate:.0%} fps={fps_actual:.1f} codec={job.codec}")
        self.job, self.review = None, None
        self.set_message(f"Saved {final.name}", ui.SUCCESS)
        if self.kept_for(sign) >= sign.reps:
            self.set_message(f"'{sign.sign_label}' complete ({sign.reps} takes). Next sign.", ui.SUCCESS)
            self.sign_index += 1
        self.go_to_next_incomplete()      # also picks the next free take number

    def retake(self) -> None:
        self.discard_pending_take()
        self.totals["retaken"] += 1
        if self.log is not None and not self.ref_mode:
            self.log.counts["retaken"] += 1
        self.log_write(f"RETAKE {self.sign.sign_id if self.sign else ''} take {self.take:02d} discarded")
        self.set_message("Take discarded. Press SPACE to record again.", ui.WARNING)
        self.state = "LIVE"

    def skip_sign(self) -> None:
        sign = self.sign
        if self.review is not None:
            self.discard_pending_take()
        self.totals["skipped"] += 1
        if self.log is not None:
            self.log.counts["skipped"] += 1
        self.log_write(f"SKIP {sign.sign_id if sign else ''} "
                       f"({self.kept_for(sign) if sign else 0}/{sign.reps if sign else 0} kept)")
        self.set_message(f"Skipped '{sign.sign_label if sign else ''}'. Click it on the right to come back.",
                         ui.WARNING)
        self.sign_index += 1
        self.go_to_next_incomplete()

    # ----- main loop --------------------------------------------------------
    def run(self) -> str:
        """Run until Q / window closed / camera lost. Returns the quit reason."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)       # resizable; the UI scales
        cv2.resizeWindow(WINDOW_NAME, ui.UI_W, ui.UI_H)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        while True:
            if self.worker.error:
                if self.job is not None:
                    self.job.done.wait(3.0)
                    self.discard_pending_take()
                print("\nERROR:", self.worker.error)
                print("Kept clips are safe. Reconnect the camera and run again - it will resume.")
                return "camera disconnected"

            frame, frame_id = self.worker.latest()
            if (self.screen == "RECORD" and frame is not None and frame_id != self._last_frame_id
                    and self.state != "REVIEW"):
                self._last_frame_id = frame_id
                self.update_detection(frame)

            if self.screen == "RECORD":
                if self.state == "COUNTDOWN" and time.perf_counter() - self.countdown_start >= COUNTDOWN_SECONDS:
                    self.start_recording()
                if self.state == "RECORDING" and self.job is not None and self.job.done.is_set():
                    self.finish_recording()

            cv2.imshow(WINDOW_NAME, self.render(frame))
            key = cv2.waitKey(1) & 0xFF
            actions: list[str] = []
            if key != 255:
                actions.append(self.key_action(key))
            if self._clicked:                             # an on-screen button was clicked
                actions.append(self._clicked)
                self._clicked = None
            while self._wheel:
                name, step = self._wheel.pop(0)
                actions.append(f"scroll:{name}:{step}")
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                actions = ["quit"]                        # window closed with the X button
            for action in actions:
                if action and self.do(action):
                    return "operator quit (Q)"

    def update_detection(self, frame: np.ndarray) -> None:
        """Preview-only MediaPipe on a small unmirrored copy (never affects saving)."""
        self._detect_counter += 1
        if self.state == "RECORDING" and (LIGHT_UI_WHILE_RECORDING or
                                          self._detect_counter % PREVIEW_DETECT_EVERY_N_WHILE_RECORDING):
            return
        h = int(frame.shape[0] * PREVIEW_DETECT_WIDTH / frame.shape[1])
        small = cv2.resize(frame, (PREVIEW_DETECT_WIDTH, h), interpolation=cv2.INTER_AREA)
        ts_ms = max(int((time.perf_counter() - self._t0) * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        self.hands = hc.detect_hands(self.landmarker, small, ts_ms)
        self.brightness = hc.mean_brightness(small)

    def key_action(self, key: int) -> str:
        """Keyboard -> action. Q quits everywhere, Esc goes back."""
        if key in (ord("q"), ord("Q")):
            return "quit"
        if key == 27:
            return "back"
        return "key:" + chr(key).lower()

    def do(self, action: str) -> bool:
        """Run one action (from a key or a click). Returns True to quit."""
        if action == "quit":
            if self.state == "RECORDING" and self.job is not None:
                self.job.abort("Quit during recording")
                self.job.done.wait(3.0)
            if self.job is not None:
                self.log_write("Unsaved take discarded on quit.")
            self.discard_pending_take()
            return True
        if self.modal == "hand":                         # the hand question blocks everything else
            choice = {"hand:left": "left", "hand:right": "right", "key:l": "left", "key:r": "right"}.get(action)
            if choice:
                self.set_hand(self.modal_signer, choice)
            elif action in ("hand:cancel", "back"):
                self.modal = None
                self.pending_category = None
            return False
        if self.modal == "ref_setup":
            self.ref_setup_action(action)
            return False

        kind, _, arg = action.partition(":")
        if action == "back":
            self.go_back()
        elif action == "home":
            self.navigate("HOME")
        elif action == "overview":
            self.navigate("OVERVIEW")
        elif action == "reference":
            self.enter_reference()
        elif action == "ref_list":
            self.navigate("REF_LIST")
        elif action == "ref_signs":
            self.navigate("REF_SIGNS")
        elif kind == "refcat":                            # REF_LIST: open a category
            self.open_ref_signs(ref.CATEGORY_FOLDERS[int(arg)][0])
        elif kind == "ref":                               # REF_SIGNS: open a sign
            self.open_record(int(arg))
        elif action == "crumb:signer" and self.signer:
            self.navigate("CATEGORIES")
        elif action == "crumb:category" and self.category:
            self.navigate("SIGNS")
        elif action == "crumb:takes":
            self.navigate("TAKES")
        elif kind == "signer" and self.leave_record_ok():
            if self.screen == "RECORD":
                self.navigate("HOME")
            self.select_signer(arg)
        elif kind == "open":                              # overview cell: open:S3:shapes
            code, category = arg.split(":")
            self.select_signer(code)
            if self.modal:
                self.pending_category = None
            else:
                self.open_category(category)
        elif kind == "cat":
            self.open_category(arg)
        elif kind == "sign":
            self.open_record(int(arg))
        elif kind == "takes":
            self.open_takes(self.signs[int(arg)])
        elif kind == "take":
            self.open_player(int(arg))
        elif action == "record_this" and self.view_sign in self.signs:
            self.close_player()
            self.open_record(self.signs.index(self.view_sign))
        elif action == "flag":
            self.toggle_flag()
        elif kind == "chip":
            if self.state in ("LIVE", "FINISHED"):
                self.select_record_sign(int(arg))
            elif self.state == "REVIEW":
                self.set_message("Keep or retake this take first.", ui.WARNING)
        elif kind == "scroll":
            name, step = arg.split(":")
            self.scroll[name] = max(0, self.scroll.get(name, 0) + int(step))
            if name == "chips":
                self._chips_follow = False
        elif kind == "key" and self.screen == "RECORD":
            self.handle_record_key(arg)
        elif kind == "key" and arg == "g":
            self.show_guides = not self.show_guides
        return False

    def ref_setup_action(self, action: str) -> None:
        """Setup dialog of reference mode: a hand button (or L / R) starts at once."""
        hand = {"refhand:left": "left", "refhand:right": "right", "key:l": "left", "key:r": "right"}.get(action)
        if hand:
            self.start_reference(hand)
        elif action in ("refcancel", "back"):
            self.modal = None

    def handle_ref_key(self, key: str) -> None:
        """Reference mode keys: SPACE record, V record version 2; review K keep, R redo;
        N next, B previous; P pause."""
        if self.state == "REVIEW":
            if key == "k":
                self.keep_take()
            elif key == "r":
                self.retake()
            elif key in ("n", "b"):
                self.ref_step(1 if key == "n" else -1)
            return
        if key in ("n", "b") and self.state in ("LIVE", "FINISHED"):
            self.ref_step(1 if key == "n" else -1)
        elif key == "p" and self.state in ("LIVE", "COUNTDOWN"):
            self.paused = not self.paused
            self.state = "LIVE"
            self.set_message("PAUSED - press P to resume" if self.paused else "Resumed", ui.WARNING)
        elif key in (" ", "v") and self.state == "LIVE" and not self.paused and self.sign is not None:
            version = 2 if key == "v" else 1
            if version == 2 and not self.is_complete(self.sign):
                self.set_message("Record the first version with SPACE before adding a second one.", ui.WARNING)
                return
            if version == 2 and self.sign.sign_id in self.ref_v2:
                self.set_message("Recording version 2 again: the current version 2 will be moved to "
                                 "references\\_old when you keep the new one.", ui.WARNING)
            elif version == 2:
                self.set_message("Recording version 2: another way of signing it. Version 1 stays.", ui.PRIMARY)
            elif self.is_complete(self.sign):
                self.set_message("Recording again: the current version will be moved to references\\_old "
                                 "when you keep the new one.", ui.WARNING)
            self.ref_version = version
            self.countdown_start = time.perf_counter()
            self.state = "COUNTDOWN"

    def handle_record_key(self, key: str) -> None:
        """The recorder's keys (unchanged): SPACE K R S P G."""
        if key == "g":
            self.show_guides = not self.show_guides
            return
        if self.ref_mode:
            self.handle_ref_key(key)
            return
        if self.state == "FINISHED":
            return
        if self.state == "REVIEW":
            if key == "k":
                self.keep_take()
            elif key == "r":
                self.retake()
            elif key == "s":
                self.skip_sign()
            return
        if key == "p" and self.state in ("LIVE", "COUNTDOWN"):
            self.paused = not self.paused
            self.state = "LIVE"
            self.set_message("PAUSED - press P to resume" if self.paused else "Resumed", ui.WARNING)
            self.log_write("PAUSED" if self.paused else "RESUMED")
        elif key == " " and self.state == "LIVE" and not self.paused:
            sign = self.sign
            if sign is not None and self.kept_for(sign) >= sign.reps:
                self.set_message(f"'{sign.sign_label}' is already complete. Pick another sign.", ui.WARNING)
                return
            self.countdown_start = time.perf_counter()
            self.state = "COUNTDOWN"
            self.message_until = 0.0
        elif key == "s" and self.state == "LIVE":
            self.skip_sign()

    # ----- mouse ------------------------------------------------------------
    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Clicks on buttons/cards become actions; the wheel scrolls lists."""
        if event == cv2.EVENT_MOUSEMOVE:
            self._mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse = (x, y)
            key = self.hits.key_at(x, y)
            if key:
                self._clicked = key
        elif event == cv2.EVENT_MOUSEWHEEL:
            # On Windows the wheel event carries screen coordinates, so use the
            # last known position inside the window instead.
            mx, my = self._mouse
            for x0, y0, x1, y1, name in self.scroll_regions:
                if x0 <= mx <= x1 and y0 <= my <= y1:
                    self._wheel.append((name, -1 if cv2.getMouseWheelDelta(flags) > 0 else 1))
                    break

    # ----- drawing: shared pieces --------------------------------------------
    # Everything is drawn on DISPLAY COPIES (ui.backdrop / ui.compose_video).
    # The frames given to the RecordingJob are never touched, so the saved
    # video stays clean (no text, landmarks, buttons or mirroring).
    def render_busy(self, text: str) -> None:
        """Dimmed last screen with a centred message (e.g. while analysing)."""
        canvas = self._last_canvas.copy() if self._last_canvas is not None else ui.blank_canvas()[0]
        ui.dim(canvas, 0.5)
        p = ui.pill(text, 20, "medium", height=54, pad=30, bg_alpha=0.7)
        p.draw(canvas, (ui.UI_W - p.width) // 2, (ui.UI_H - p.height) // 2)
        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(1)

    def render(self, frame: np.ndarray | None) -> np.ndarray:
        """Build the display image for the current screen."""
        self.hits.clear()
        self.scroll_regions.clear()
        if self.screen == "RECORD":
            canvas = self.render_record(frame)
        elif self.screen == "PLAYER":
            canvas = self.render_player()
        else:
            canvas = ui.backdrop(frame)
            {"HOME": self.draw_home, "CATEGORIES": self.draw_categories, "SIGNS": self.draw_signs,
             "TAKES": self.draw_takes, "OVERVIEW": self.draw_overview,
             "REF_LIST": self.draw_ref_list, "REF_SIGNS": self.draw_ref_signs}[self.screen](canvas)
            self.draw_notices(canvas, ui.UI_H - ui.MARGIN - 46)
        if self.modal in ("hand", "ref_setup"):
            self.hits.clear()                            # only the dialog is clickable
            self.scroll_regions.clear()
            if self.modal == "hand":
                self.draw_hand_modal(canvas)
            else:
                self.draw_ref_setup(canvas)
                self.draw_notices(canvas, ui.UI_H - ui.MARGIN - 46)
        self.draw_hover(canvas)
        self._last_canvas = canvas
        return canvas

    def draw_hover(self, canvas: np.ndarray) -> None:
        """Thin outline around the clickable item under the mouse."""
        area = self.hits.area_at(*self._mouse)
        if area and self.state != "RECORDING":
            x0, y0, x1, y1, _ = area
            w, h = x1 - x0, y1 - y0
            ui.hover_outline(w, h, min(18, h // 2)).draw(canvas, x0, y0)

    def crumbs(self) -> list[tuple[str, str | None]]:
        """Breadcrumb (label, action) list for the current screen."""
        out: list[tuple[str, str | None]] = [("Signers", "home")]
        if self.ref_mode:
            out.append(("Reference videos", "ref_list"))
            if self.screen in ("REF_SIGNS", "RECORD") and self.ref_folder:
                out.append((ref.folder_title(self.ref_folder), "ref_signs"))
            if self.screen == "RECORD" and self.sign is not None:
                out.append((self.sign.sign_label, None))
            return out
        if self.screen == "OVERVIEW":
            return out + [("Progress overview", None)]
        if self.screen == "HOME" or not self.signer:
            return out
        out.append((self.signer, "crumb:signer"))
        if self.screen == "CATEGORIES" or not self.category:
            return out
        out.append((store.category_title(self.category), "crumb:category"))
        if self.screen == "RECORD" and self.sign is not None:
            out.append((self.sign.sign_label, None))
        elif self.screen in ("TAKES", "PLAYER") and self.view_sign is not None:
            out.append((self.view_sign.sign_label, "crumb:takes"))
            if self.screen == "PLAYER" and self.player is not None:
                out.append((f"Take {int(self.player.row.get('take', '0') or 0):02d}", None))
        return out

    def draw_crumbs(self, canvas: np.ndarray, x: int, y: int) -> None:
        items = self.crumbs()
        items[-1] = (items[-1][0], None)                 # the current page is not a link
        sep = ui.pill("›", 18, "semibold", ui.MUTED, ui.DARK, 0.0, height=36, pad=4)
        for i, (label, action) in enumerate(items):
            current = i == len(items) - 1
            p = ui.pill(label, 15, "semibold" if current else "medium", ui.WHITE,
                        ui.PRIMARY if current else ui.DARK, 0.9 if current else ui.PANEL_ALPHA, height=36, pad=16)
            p.draw(canvas, x, y)
            if action:
                self.hits.add(x, y, p.width, p.height, action)
            x += p.width
            if not current:
                sep.draw(canvas, x + 2, y)
                x += sep.width + 4

    def draw_header(self, canvas: np.ndarray, title: str, subtitle: str = "", back: bool = True) -> int:
        """Back button, breadcrumbs, top-right buttons, big title. Returns content top y."""
        m = ui.MARGIN
        x = m
        if back:
            ui.round_button("back", 44).draw(canvas, m, m)
            ui.key_badge("Esc").draw(canvas, m + 26, m + 30)
            self.hits.add(m, m, 44, 44, "back")
            x = m + 60
        self.draw_crumbs(canvas, x, m + 4)
        # top-right: overview + quit
        q = ui.pill("Quit", 15, "medium", height=40, pad=18, bg_alpha=0.6)
        qx = ui.UI_W - m - q.width
        q.draw(canvas, qx, m + 2)
        ui.key_badge("Q").draw(canvas, qx + q.width - 14, m + 28)
        self.hits.add(qx, m + 2, q.width, q.height, "quit")
        if self.screen not in ("OVERVIEW", "REF_LIST", "REF_SIGNS"):
            o = ui.pill("Progress overview", 15, "semibold", ui.WHITE, ui.PRIMARY, 0.92, height=40, pad=18,
                        icon="chart")
            ox = qx - 12 - o.width
            o.draw(canvas, ox, m + 2)
            self.hits.add(ox, m + 2, o.width, o.height, "overview")
            if self.screen == "HOME":                    # reference videos for the expert validation
                rv = ui.pill("Reference videos", 15, "semibold", height=40, pad=18, bg_alpha=0.6, icon="play")
                rx = ox - 12 - rv.width
                rv.draw(canvas, rx, m + 2)
                self.hits.add(rx, m + 2, rv.width, rv.height, "reference")
        t = ui.title(title, ui.fit_title_size(title, 900, 46))
        ty = m + 58
        t.draw(canvas, m, ty)
        y = ty + t.height + 2
        if subtitle:
            s = ui.pill(subtitle, 15, "regular", ui.WHITE, ui.DARK, 0.45, height=30, pad=14, fg_alpha=0.85)
            s.draw(canvas, m, y)
            y += s.height
        return y + 18

    def draw_caption(self, canvas: np.ndarray, text: str) -> int:
        """Subtitle-style instruction bar, bottom centre. Returns its top y."""
        cap = ui.pill(text, 18, "medium", height=46, pad=26, bg_alpha=0.62)
        y = ui.UI_H - ui.MARGIN - cap.height
        cap.draw(canvas, (ui.UI_W - cap.width) // 2, y)
        return y

    def draw_notices(self, canvas: np.ndarray, cap_y: int, warnings: bool = False) -> None:
        """Latest message (and camera warnings on the record screen) above the caption."""
        notices: list[ui.Sprite] = []
        if self.message and time.perf_counter() < self.message_until:
            notices.append(ui.pill(self.message, 15, dot=self.message_color, bg_alpha=0.75))
        if warnings:
            for w in self.standard_warnings:
                notices.append(ui.pill(w, 15, "semibold", ui.DARK, ui.WARNING, 0.95))
        y = cap_y - 10
        for p in notices:
            y -= p.height
            p.draw(canvas, (ui.UI_W - p.width) // 2, y)
            y -= 8

    def draw_scroll_buttons(self, canvas: np.ndarray, name: str, x: int, y_top: int, y_bottom: int,
                            can_up: bool, can_down: bool) -> None:
        """Small round up/down buttons for a scrollable list."""
        d = 34
        for icon, yy, ok, step in (("up", y_top, can_up, -1), ("down", y_bottom - d, can_down, 1)):
            ui.round_button(icon, d, bg_alpha=0.7).draw(canvas, x, yy, 1.0 if ok else 0.3)
            if ok:
                self.hits.add(x, yy, d, d, f"scroll:{name}:{step}")

    def _scroll_window(self, name: str, total: int, visible: int) -> int:
        """Clamp a list's scroll position; returns the first visible row."""
        first = max(0, min(self.scroll.get(name, 0), max(0, total - visible)))
        self.scroll[name] = first
        return first

    # ----- drawing: HOME (signer selector) --------------------------------------
    def draw_home(self, canvas: np.ndarray) -> None:
        top = self.draw_header(canvas, "Choose a signer",
                               "Click a card to start", back=False)
        w, h, gap = 284, 170, 18
        x0 = (ui.UI_W - (4 * w + 3 * gap)) // 2
        for i, code in enumerate(store.SIGNER_CODES):
            x, y = x0 + (i % 4) * (w + gap), top + 10 + (i // 4) * (h + gap)
            info = self.signers[code]
            done, req = self.progress.done_required(code, self.all_signs)
            frac = done / req if req else 0.0
            complete = req > 0 and done >= req
            hand = f"{info.hand.title()} hand" if info.hand else "Hand not set — click to choose"
            texts = [(24, 42, code, 40, "bold", ui.WHITE, 1.0, "lm"),
                     (24, 92, hand, 15, "medium", ui.WHITE if info.hand else ui.WARNING, 1.0, "lm"),
                     (24, 126, f"{done:,} / {req:,} clips", 15, "medium", ui.WHITE, 0.9, "lm"),
                     (w - 24, 126, f"{frac:.0%}", 15, "semibold", ui.SUCCESS if complete else ui.WHITE, 1.0, "rm")]
            if info.display_name:
                texts.append((24 + ui.text_width(code, 40, "bold") + 12, 46, info.display_name[:18], 15,
                              "medium", ui.MUTED, 1.0, "lm"))
            outline = ui.SUCCESS if complete else (ui.PRIMARY if code == self.signer else None)
            ui.box(w, h, 22, ui.DARK, 0.62, outline=outline, texts=tuple(texts)).draw(canvas, x, y)
            ui.progress_bar(w - 48, 6, frac).draw(canvas, x + 24, y + h - 26)
            self.hits.add(x, y, w, h, f"signer:{code}")
        self.draw_caption(canvas, "Click a signer   ·   Q quit")

    def draw_hand_modal(self, canvas: np.ndarray) -> None:
        """Asks once which hand a signer signs with; saved in signers.csv."""
        ui.dim(canvas, 0.55)
        w, h = 560, 260
        x, y = (ui.UI_W - w) // 2, (ui.UI_H - h) // 2
        ui.box(w, h, 24, ui.DARK, 0.92, outline=ui.WHITE, outline_alpha=0.25, texts=(
            (w / 2, 52, f"Which hand does {self.modal_signer} sign with?", 22, "semibold", ui.WHITE, 1.0, "mm"),
            (w / 2, 88, "Asked once and saved in signers.csv. Data is never flipped.", 14, "regular",
             ui.MUTED, 1.0, "mm"))).draw(canvas, x, y)
        bw, bh = 220, 64
        for i, (label, action) in enumerate((("Left hand", "hand:left"), ("Right hand", "hand:right"))):
            bx = x + 40 + i * (bw + 40)
            ui.box(bw, bh, bh // 2, ui.PRIMARY, 0.95, texts=(
                (bw / 2, bh / 2, label, 20, "semibold", ui.WHITE, 1.0, "mm"),)).draw(canvas, bx, y + 120)
            self.hits.add(bx, y + 120, bw, bh, action)
        c = ui.pill("Cancel", 14, height=32, pad=16, bg_alpha=0.6)
        cx = x + (w - c.width) // 2
        c.draw(canvas, cx, y + h - 50)
        self.hits.add(cx, y + h - 50, c.width, c.height, "hand:cancel")

    # ----- drawing: CATEGORIES ----------------------------------------------------
    def draw_categories(self, canvas: np.ndarray) -> None:
        code = self.signer or ""
        done_all, req_all = self.progress.done_required(code, self.all_signs)
        top = self.draw_header(canvas, f"{code} · Categories",
                               f"{self.hand.title() or '?'} hand  ·  {done_all:,} / {req_all:,} clips overall")
        w, h, gap = 284, 132, 18
        x0 = (ui.UI_W - (4 * w + 3 * gap)) // 2
        for i, cat in enumerate(self.categories):
            signs = self.category_signs(cat)
            done, req = self.progress.done_required(code, signs)
            complete = req > 0 and done >= req
            x, y = x0 + (i % 4) * (w + gap), top + (i // 4) * (h + gap)
            icons = (("check", w - 52, 14, 30, ui.SUCCESS, 1.0),) if complete else ()
            ui.box(w, h, 22, ui.SUCCESS if complete else ui.DARK, 0.30 if complete else 0.62,
                   outline=ui.SUCCESS if complete else None, icons=icons, texts=(
                       (22, 32, store.category_title(cat), 19, "semibold", ui.WHITE, 1.0, "lm"),
                       (22, 76, f"{done} / {req}", 28, "bold", ui.SUCCESS if complete else ui.WHITE, 1.0, "lm"),
                       (w - 22, 78, f"{len(signs)} sign{'s' if len(signs) != 1 else ''}", 13, "regular",
                        ui.MUTED, 1.0, "rm"),
                   )).draw(canvas, x, y)
            ui.progress_bar(w - 44, 6, done / req if req else 0).draw(canvas, x + 22, y + h - 24)
            self.hits.add(x, y, w, h, f"cat:{cat}")
        self.draw_caption(canvas, "Click a category   ·   Esc back")

    # ----- drawing: SIGNS -----------------------------------------------------------
    def draw_signs(self, canvas: np.ndarray) -> None:
        code = self.signer or ""
        done, req = self.progress.done_required(code, self.signs)
        top = self.draw_header(canvas, store.category_title(self.category or ""),
                               f"{done} / {req} takes  ·  click a sign to record  ·  list icon = saved takes")
        cols, gap, rh, step = 3, 16, 52, 62
        right_pad = 50                                   # room for the scroll buttons
        cw = (ui.UI_W - 2 * ui.MARGIN - right_pad - (cols - 1) * gap) // cols
        bottom = ui.UI_H - ui.MARGIN - 46 - 14
        visible_rows = max(1, (bottom - top) // step)
        total_rows = (len(self.signs) + cols - 1) // cols
        first = self._scroll_window("signs", total_rows, visible_rows)
        self.scroll_regions.append((ui.MARGIN, top, ui.UI_W - ui.MARGIN, bottom, "signs"))
        for r in range(first, min(total_rows, first + visible_rows)):
            for c in range(cols):
                i = r * cols + c
                if i >= len(self.signs):
                    break
                s = self.signs[i]
                x, y = ui.MARGIN + c * (cw + gap), top + (r - first) * step
                kept, flagged = self.kept_for(s), self.progress.flagged_for(code, s)
                complete = kept >= s.reps
                texts = [(18, rh / 2, s.sign_label, 16, "medium", ui.WHITE, 1.0, "lm"),
                         (cw - 62, rh / 2, f"{min(kept, s.reps)} / {s.reps}", 15, "semibold",
                          ui.SUCCESS if complete else ui.WHITE, 1.0, "rm")]
                tx = 18 + ui.text_width(s.sign_label, 16, "medium") + 10
                if s.hands == 2:
                    texts.append((tx, rh / 2, "2 hands", 11, "semibold", ui.PRIMARY_LIGHT, 1.0, "lm"))
                    tx += ui.text_width("2 hands", 11, "semibold") + 8
                if flagged:
                    texts.append((tx, rh / 2, f"{flagged} flagged", 11, "semibold", ui.WARNING, 1.0, "lm"))
                icons = (("check", cw - 62 - ui.text_width(f"{min(kept, s.reps)} / {s.reps}", 15, "semibold") - 26,
                          rh / 2 - 10, 20, ui.SUCCESS, 1.0),) if complete else ()
                ui.box(cw, rh, 16, ui.SUCCESS if complete else ui.DARK, 0.28 if complete else 0.62,
                       outline=ui.SUCCESS if complete else None, outline_alpha=0.7,
                       texts=tuple(texts), icons=icons).draw(canvas, x, y)
                self.hits.add(x, y, cw - 48, rh, f"sign:{i}")
                ui.round_button("list", 36, bg_alpha=0.7).draw(canvas, x + cw - 44, y + 8)
                self.hits.add(x + cw - 44, y + 8, 36, 36, f"takes:{i}")
        if total_rows > visible_rows:
            self.draw_scroll_buttons(canvas, "signs", ui.UI_W - ui.MARGIN - 34, top, bottom,
                                     first > 0, first + visible_rows < total_rows)
        self.draw_caption(canvas, "Click a sign to record   ·   Esc back")

    # ----- drawing: TAKES (saved takes of one sign) ----------------------------------
    def draw_takes(self, canvas: np.ndarray) -> None:
        s = self.view_sign
        assert s is not None
        rows = self.take_rows
        kept = sum(1 for r in rows if r.get("status") == store.STATUS_KEPT)
        top = self.draw_header(canvas, f"{s.sign_label} — saved takes",
                               f"{kept} kept  ·  {len(rows) - kept} flagged  ·  {s.reps} required")
        rec = ui.pill("Record this sign", 15, "semibold", ui.WHITE, ui.DANGER, 0.9, height=40, pad=18, icon="record")
        rx = ui.UI_W - ui.MARGIN - rec.width
        rec.draw(canvas, rx, top - 58)
        self.hits.add(rx, top - 58, rec.width, rec.height, "record_this")
        if not rows:
            p = ui.pill("No kept takes yet for this sign", 17, height=48, pad=24, fg=ui.MUTED)
            p.draw(canvas, (ui.UI_W - p.width) // 2, top + 60)
            self.draw_caption(canvas, "Esc back")
            return
        rh, step = 50, 58
        w = ui.UI_W - 2 * ui.MARGIN - 50
        bottom = ui.UI_H - ui.MARGIN - 46 - 14
        visible = max(1, (bottom - top) // step)
        first = self._scroll_window("takes", len(rows), visible)
        self.scroll_regions.append((ui.MARGIN, top, ui.UI_W - ui.MARGIN, bottom, "takes"))
        for i in range(first, min(len(rows), first + visible)):
            r = rows[i]
            y = top + (i - first) * step
            flagged = r.get("status") == store.STATUS_NEEDS_RERECORD
            take = int(r.get("take") or 0)
            hdr, bhr = float(r.get("hand_detect_rate") or 0), float(r.get("both_hands_rate") or 0)
            texts = ((58, rh / 2, f"Take {take:02d}", 16, "semibold", ui.WHITE, 1.0, "lm"),
                     (170, rh / 2, (r.get("timestamp") or "").replace("T", "  "), 14, "regular", ui.MUTED, 1.0, "lm"),
                     (400, rh / 2, f"Hand {hdr:.0%}", 14, "medium",
                      ui.WARNING if hdr < hc.HAND_DETECT_WARN_RATE else ui.WHITE, 1.0, "lm"),
                     (520, rh / 2, f"Both hands {bhr:.0%}", 14, "medium",
                      ui.WARNING if s.hands == 2 and bhr < hc.BOTH_HANDS_WARN_RATE else ui.WHITE, 1.0, "lm"),
                     (690, rh / 2, f"{float(r.get('duration_s') or 0):.2f} s", 14, "medium", ui.WHITE, 1.0, "lm"),
                     (w - 22, rh / 2, "Needs re-record" if flagged else "Kept", 14, "semibold",
                      ui.WARNING if flagged else ui.SUCCESS, 1.0, "rm"))
            ui.box(w, rh, 16, ui.DARK, 0.62, outline=ui.WARNING if flagged else None, outline_alpha=0.7,
                   texts=texts, icons=(("play", 18, 13, 24, ui.WHITE, 0.9),)).draw(canvas, ui.MARGIN, y)
            self.hits.add(ui.MARGIN, y, w, rh, f"take:{i}")
        if len(rows) > visible:
            self.draw_scroll_buttons(canvas, "takes", ui.UI_W - ui.MARGIN - 34, top, bottom,
                                     first > 0, first + visible < len(rows))
        self.draw_caption(canvas, "Click a take to replay it   ·   Esc back")

    # ----- drawing: PLAYER (replay one saved take) -------------------------------------
    def render_player(self) -> np.ndarray:
        p, s = self.player, self.view_sign
        assert p is not None and s is not None
        frame = p.current()
        if frame is not None:
            canvas, _ = ui.compose_video(frame, mirror=True)
        else:
            canvas, _ = ui.blank_canvas()
            msg = ui.pill(f"Video file not found: {p.row.get('file_path', '')}", 16, fg=ui.WARNING, height=44, pad=22)
            msg.draw(canvas, (ui.UI_W - msg.width) // 2, ui.UI_H // 2)
        m = ui.MARGIN
        ui.round_button("back", 44).draw(canvas, m, m)
        ui.key_badge("Esc").draw(canvas, m + 26, m + 30)
        self.hits.add(m, m, 44, 44, "back")
        self.draw_crumbs(canvas, m + 60, m + 4)
        row = p.row
        take = int(row.get("take") or 0)
        label = f"{s.sign_label.upper()} · TAKE {take:02d}"
        t = ui.title(label, ui.fit_title_size(label, TITLE_MAX_WIDTH, 52))
        t.draw(canvas, m, m + 58)
        flagged = row.get("status") == store.STATUS_NEEDS_RERECORD
        two = s.hands == 2
        hdr, bhr = float(row.get("hand_detect_rate") or 0), float(row.get("both_hands_rate") or 0)
        rows = [
            ("Hand detection", f"{hdr:.0%}", ui.SUCCESS if hdr >= hc.HAND_DETECT_WARN_RATE else ui.WARNING, False),
            ("Both hands", f"{bhr:.0%}",
             (ui.SUCCESS if bhr >= hc.BOTH_HANDS_WARN_RATE else ui.WARNING) if two else ui.MUTED, two),
            ("Duration", f"{float(row.get('duration_s') or 0):.2f} s", ui.WHITE, False),
            ("Frames / fps", f"{row.get('n_frames', '?')} / {float(row.get('fps_actual') or 0):.1f}", ui.WHITE, False),
            ("Brightness", f"{float(row.get('mean_brightness') or 0):.0f}", ui.WHITE, False),
            ("Recorded", (row.get("timestamp") or "").replace("T", " ")[:16], ui.WHITE, False),
        ]
        y = m + 58 + t.height + 12
        c = ui.card(rows, "Take info", width=CARD_W + 30, tag="2-HAND SIGN" if two else None)
        c.draw(canvas, m, y)
        y += c.height + 10
        status = ui.pill("Needs re-record" if flagged else "Kept — counts toward the takes", 14, "semibold",
                         dot=ui.WARNING if flagged else ui.SUCCESS, bg_alpha=0.7)
        status.draw(canvas, m, y)
        y += status.height + 10
        if flagged:
            b = ui.pill("Unflag (count as kept)", 16, "semibold", ui.WHITE, ui.PRIMARY, 0.92, height=46, pad=22,
                        icon="check")
        else:
            b = ui.pill("Flag: needs re-record", 16, "semibold", ui.DARK, ui.WARNING, 0.95, height=46, pad=22,
                        icon="flag")
        b.draw(canvas, m, y)
        self.hits.add(m, y, b.width, b.height, "flag")
        cap_y = self.draw_caption(canvas, "Mirror view replay  ·  flagged takes don't count and are re-recorded"
                                          "  ·  Esc back")
        self.draw_notices(canvas, cap_y)
        return canvas

    # ----- drawing: OVERVIEW (signers x categories) ---------------------------------
    def draw_overview(self, canvas: np.ndarray) -> None:
        top = self.draw_header(canvas, "Progress overview",
                               "Kept / required clips from metadata.csv  ·  click a cell to open it")
        cats = self.categories
        name_w = 118
        cols = len(cats) + 1
        cw = (ui.UI_W - 2 * ui.MARGIN - name_w) // cols
        x0 = ui.MARGIN + name_w
        for j, cat in enumerate(cats + ["total"]):          # header row
            label = "Total" if cat == "total" else store.CATEGORY_SHORT.get(cat, cat)
            ui.box(cw - 6, 34, 12, ui.DARK, 0.5, texts=(
                ((cw - 6) / 2, 17, label, 13, "semibold", ui.MUTED, 1.0, "mm"),)).draw(canvas, x0 + j * cw, top)
        rh, step = 46, 52
        for i, code in enumerate(store.SIGNER_CODES):
            y = top + 44 + i * step
            info = self.signers[code]
            ui.box(name_w - 10, rh, 14, ui.DARK, 0.62, texts=(
                (16, rh / 2, code, 18, "bold", ui.WHITE, 1.0, "lm"),
                (name_w - 24, rh / 2, (info.hand or "?").title(), 12, "medium",
                 ui.MUTED if info.hand else ui.WARNING, 1.0, "rm"))).draw(canvas, ui.MARGIN, y)
            self.hits.add(ui.MARGIN, y, name_w - 10, rh, f"signer:{code}")
            for j, cat in enumerate(cats + ["total"]):
                x = x0 + j * cw
                if cat == "total":
                    done, req = self.progress.done_required(code, self.all_signs)
                else:
                    done, req = self.progress.done_required(code, self.category_signs(cat))
                frac = done / req if req else 0.0
                if req and done >= req:
                    bg, alpha, fg = ui.SUCCESS, 0.85, ui.WHITE
                elif done > 0:
                    bg, alpha, fg = ui.PRIMARY, round(0.25 + 0.55 * frac, 2), ui.WHITE
                else:
                    bg, alpha, fg = ui.DARK, 0.55, ui.MUTED
                ui.box(cw - 6, rh, 12, bg, alpha, texts=(
                    ((cw - 6) / 2, rh / 2, f"{done}/{req}", 13, "semibold", fg, 1.0, "mm"),)).draw(canvas, x, y)
                if cat != "total":
                    self.hits.add(x, y, cw - 6, rh, f"open:{code}:{cat}")
        self.draw_caption(canvas, "Green = complete  ·  blue = in progress  ·  Esc back")

    # ----- drawing: REF_LIST (reference videos: category tiles) ---------------------
    def draw_ref_list(self, canvas: np.ndarray) -> None:
        """Same layout as the CATEGORIES screen: one tile per category folder."""
        done_all, total_all = len(self.ref_done), len(self.refs)
        top = self.draw_header(canvas, "Reference videos",
                               f"{done_all} / {total_all} recorded  ·  {self.ref_hand.title()} hand  ·  "
                               f"one {self.ref_seconds:g}-second take per sign")
        w, h, gap = 284, 132, 18
        x0 = (ui.UI_W - (4 * w + 3 * gap)) // 2
        for i, (folder, _) in enumerate(ref.CATEGORY_FOLDERS):
            items = self.ref_folder_indices(folder)
            done = sum(1 for k in items if self.refs[k].sign.sign_id in self.ref_done)
            complete = done == len(items)
            last = folder == self.ref_folder                     # category of the last sign used
            x, y = x0 + (i % 4) * (w + gap), top + (i // 4) * (h + gap)
            icons = (("check", w - 52, 14, 30, ui.SUCCESS, 1.0),) if complete else ()
            ui.box(w, h, 22, ui.SUCCESS if complete else ui.DARK, 0.30 if complete else 0.62,
                   outline=ui.SUCCESS if complete else (ui.PRIMARY if last else None), icons=icons, texts=(
                       (22, 32, folder, 19, "semibold", ui.WHITE, 1.0, "lm"),
                       (22, 76, f"{done} / {len(items)}", 28, "bold", ui.SUCCESS if complete else ui.WHITE, 1.0, "lm"),
                       (w - 22, 78, f"{len(items)} sign{'s' if len(items) != 1 else ''}", 13, "regular",
                        ui.MUTED, 1.0, "rm"),
                   )).draw(canvas, x, y)
            ui.progress_bar(w - 44, 6, done / len(items) if items else 0).draw(canvas, x + 22, y + h - 24)
            self.hits.add(x, y, w, h, f"refcat:{i}")
        self.draw_caption(canvas, "Click a category   ·   blue outline = last used   ·   Esc back")

    # ----- drawing: REF_SIGNS (reference videos: signs of one category) ---------------
    def draw_ref_signs(self, canvas: np.ndarray) -> None:
        """Same layout as the SIGNS screen: each sign marked done or pending."""
        items = self.ref_folder_indices(self.ref_folder)
        done = sum(1 for k in items if self.refs[k].sign.sign_id in self.ref_done)
        top = self.draw_header(canvas, ref.folder_title(self.ref_folder),
                               f"{done} / {len(items)} recorded  ·  click a sign to record it")
        cols, gap, rh, step = REF_SIGN_COLS, 16, 52, 62
        right_pad = 50                                   # room for the scroll buttons
        cw = (ui.UI_W - 2 * ui.MARGIN - right_pad - (cols - 1) * gap) // cols
        bottom = ui.UI_H - ui.MARGIN - 46 - 14
        visible_rows = max(1, (bottom - top) // step)
        total_rows = (len(items) + cols - 1) // cols
        first = self._scroll_window("ref_signs", total_rows, visible_rows)
        self.scroll_regions.append((ui.MARGIN, top, ui.UI_W - ui.MARGIN, bottom, "ref_signs"))
        for row in range(first, min(total_rows, first + visible_rows)):
            for c in range(cols):
                p = row * cols + c
                if p >= len(items):
                    break
                i = items[p]
                r = self.refs[i]
                x, y = ui.MARGIN + c * (cw + gap), top + (row - first) * step
                is_done = r.sign.sign_id in self.ref_done
                last = i == self.sign_index
                label = r.title if len(r.title) <= 24 else r.title[:23] + "…"
                icons = (("check", cw - 18 - ui.text_width("done", 14, "semibold") - 26, rh / 2 - 10, 20,
                          ui.SUCCESS, 1.0),) if is_done else ()
                ui.box(cw, rh, 16, ui.SUCCESS if is_done else ui.DARK, 0.28 if is_done else 0.62,
                       outline=ui.PRIMARY if last else (ui.SUCCESS if is_done else None),
                       outline_alpha=0.9 if last else 0.7, icons=icons, texts=(
                           (18, rh / 2, label, 16, "medium", ui.WHITE, 1.0, "lm"),
                           (cw - 18, rh / 2, "done" if is_done else "pending", 14, "semibold",
                            ui.SUCCESS if is_done else ui.MUTED, 1.0, "rm"))).draw(canvas, x, y)
                self.hits.add(x, y, cw, rh, f"ref:{i}")
        if total_rows > visible_rows:
            self.draw_scroll_buttons(canvas, "ref_signs", ui.UI_W - ui.MARGIN - 34, top, bottom,
                                     first > 0, first + visible_rows < total_rows)
        self.draw_caption(canvas, "Click a sign to record it   ·   blue outline = last sign   ·   Esc back")

    def draw_ref_setup(self, canvas: np.ndarray) -> None:
        """Asked when reference mode starts: anyone can sign; click the hand to start."""
        ui.dim(canvas, 0.55)
        w, h = 620, 300
        x, y = (ui.UI_W - w) // 2, (ui.UI_H - h) // 2
        ui.box(w, h, 24, ui.DARK, 0.92, outline=ui.WHITE, outline_alpha=0.25, texts=(
            (w / 2, 48, "Reference videos", 24, "semibold", ui.WHITE, 1.0, "mm"),
            (w / 2, 84, "Anyone who knows the signs can record. Which hand will sign?", 14, "regular",
             ui.MUTED, 1.0, "mm"))).draw(canvas, x, y)
        bw, bh, gap = 250, 64, 40
        for i, hand in enumerate(store.HANDS):
            bx = x + (w - 2 * bw - gap) // 2 + i * (bw + gap)
            last = self.ref_hand == hand                 # the hand used last time
            ui.box(bw, bh, bh // 2, ui.PRIMARY, 0.95 if last else 0.7,
                   outline=ui.WHITE if last else None, outline_alpha=0.6, texts=(
                       (bw / 2, bh / 2, f"{hand.title()} hand · start", 18, "semibold", ui.WHITE, 1.0, "mm"),)
                   ).draw(canvas, bx, y + 120)
            self.hits.add(bx, y + 120, bw, bh, f"refhand:{hand}")
        c = ui.pill("Cancel", 15, height=40, pad=18, bg_alpha=0.6)
        cx = x + (w - c.width) // 2
        c.draw(canvas, cx, y + h - 76)
        self.hits.add(cx, y + h - 76, c.width, c.height, "refcancel")
        hint = ui.pill("L left hand  ·  R right hand  ·  Esc cancel", 13, height=28, pad=12,
                       fg=ui.MUTED, bg_alpha=0.0)
        hint.draw(canvas, x + (w - hint.width) // 2, y + h - 32)

    # ----- drawing: RECORD screen ---------------------------------------------------
    def render_record(self, frame: np.ndarray | None) -> np.ndarray:
        sign = self.sign
        if self.state == "REVIEW" and self.review and self.review.frames:
            idx = int((time.perf_counter() - self.review_start) * hc.TARGET_FPS) % len(self.review.frames)
            canvas, rect = ui.compose_video(self.review.frames[idx], mirror=False)  # stored pre-mirrored
            hands = self.review.hands[idx]
        elif frame is not None:
            canvas, rect = ui.compose_video(frame, mirror=True)                    # selfie view (copy)
            hands = self.hands
        else:
            canvas, rect = ui.blank_canvas()
            hands = []

        if self.state == "RECORDING" and LIGHT_UI_WHILE_RECORDING:
            self.draw_recording_hud(canvas)          # minimal UI during the 3-second take
            return canvas

        if self.show_guides and self.state in ("LIVE", "COUNTDOWN") and not self.paused:
            ui.draw_guides(canvas, rect)
        ui.draw_hands(canvas, hands, rect, mirrored=True)
        if self.paused or self.state == "FINISHED":
            ui.dim(canvas)
        self.draw_left_column(canvas, sign, hands)
        self.draw_right_column(canvas, sign)
        self.draw_center(canvas)
        self.draw_bottom(canvas)
        return canvas

    def draw_left_column(self, canvas: np.ndarray, sign: hc.Sign | None,
                         hands: list[hc.DetectedHand]) -> None:
        """Back button, breadcrumbs, sign name, progress bar, pills and the review card."""
        m = ui.MARGIN
        ui.round_button("back", 48).draw(canvas, m, m)
        ui.key_badge("Esc").draw(canvas, m + 30, m + 32)
        self.hits.add(m, m, 48, 48, "back")
        self.draw_crumbs(canvas, m + 64, m + 6)
        if sign is None:
            return

        label = sign.sign_label.upper()
        name = ui.title(label, ui.fit_title_size(label, TITLE_MAX_WIDTH))
        name_y = m + 62
        name.draw(canvas, m, name_y)
        if self.ref_mode:
            self.draw_ref_pills(canvas, sign, hands, name_y + name.height + 4)
            return
        kept = self.kept_for(sign)
        bar_y = name_y + name.height + 4
        ui.progress_bar(PROGRESS_W, 8, kept / sign.reps).draw(canvas, m + 2, bar_y)

        y = bar_y + 24
        enough = len(hands) >= sign.hands
        signer_pill = ui.pill(f"Signer {self.signer} · {self.hand}")
        signer_pill.draw(canvas, m, y)
        self.hits.add(m, y, signer_pill.width, signer_pill.height, "home")                 # signer pill -> signer selector
        y += signer_pill.height + 8
        pills = [
            ui.pill(f"Take {min(kept + 1, sign.reps)} / {sign.reps}" if kept < sign.reps
                    else f"Complete · {sign.reps} / {sign.reps}", dot=ui.SUCCESS if kept >= sign.reps else None),
            ui.pill(f"Hands: {sign.hands}", dot=ui.SUCCESS if enough else ui.MUTED),
        ]
        if self.state in ("LIVE", "COUNTDOWN"):
            text, dot, warn = ui.brightness_info(self.brightness)
            if warn:
                pills.append(ui.pill(text, dot=dot))
        for p in pills:
            p.draw(canvas, m, y)
            y += p.height + 8

        if self.state == "REVIEW" and self.review is not None:
            self.draw_quality_card(canvas, sign, m, y + 8)

    def draw_ref_pills(self, canvas: np.ndarray, sign: hc.Sign, hands: list[hc.DetectedHand], bar_y: int) -> None:
        """Reference mode, under the sign name: overall progress, file name, done/pending, hands."""
        m, r = ui.MARGIN, self.current_ref
        assert r is not None
        done, total = len(self.ref_done), len(self.refs)
        ui.progress_bar(PROGRESS_W, 8, done / total if total else 0).draw(canvas, m + 2, bar_y)
        y = bar_y + 24
        is_done = self.is_complete(sign)
        status = "pending"
        if is_done:
            status = "done + version 2" if sign.sign_id in self.ref_v2 else "done"
        if self.state in ("COUNTDOWN", "RECORDING", "REVIEW") and self.ref_version == 2:
            status += "  ·  now recording version 2"
        pills = [
            ui.pill(f"Reference · {self.ref_hand} hand · {done} / {total}"),
            ui.pill(f"{r.folder}  ›  {r.file_stem}  ·  {status}",
                    dot=ui.SUCCESS if is_done else ui.MUTED),
            ui.pill(f"Hands: {sign.hands}", dot=ui.SUCCESS if len(hands) >= sign.hands else ui.MUTED),
        ]
        if self.state in ("LIVE", "COUNTDOWN"):
            text, dot, warn = ui.brightness_info(self.brightness)
            if warn:
                pills.append(ui.pill(text, dot=dot))
        for p in pills:
            p.draw(canvas, m, y)
            y += p.height + 8
        if self.state == "REVIEW" and self.review is not None:
            self.draw_quality_card(canvas, sign, m, y + 8)

    def draw_quality_card(self, canvas: np.ndarray, sign: hc.Sign, x: int, y: int) -> None:
        r = self.review
        assert r is not None
        two = sign.hands == 2

        def color(rate: float, threshold: float) -> tuple:
            return ui.SUCCESS if rate >= threshold else ui.WARNING

        dur = self.job.duration_s() if self.job else 0.0
        win = (r.window_end - r.window_start + 1) / r.fps if r.fps else 0.0
        rows = [
            ("Hand detection", f"{r.hand_detect_rate:.0%}",
             color(r.hand_detect_rate, hc.HAND_DETECT_WARN_RATE), False),
            ("Both hands", f"{r.both_hands_rate:.0%}",
             color(r.both_hands_rate, hc.BOTH_HANDS_WARN_RATE) if two else ui.MUTED, two),
            ("Face visible", "not checked" if r.face_visible_rate is None else f"{r.face_visible_rate:.0%}",
             ui.MUTED if r.face_visible_rate is None else color(r.face_visible_rate, hc.FACE_VISIBLE_WARN_RATE),
             False),
            ("Sign window", f"{win:.2f} s", ui.WHITE, False),       # the part kept after trimming
            ("Duration", f"{dur:.2f} s", ui.WHITE, False),
        ]
        c = ui.card(rows, "Take quality", width=CARD_W, tag="2-HAND SIGN" if two else None)
        c.draw(canvas, x, y)
        y += c.height + 10
        notes = [ui.pill(w, 14, "semibold", ui.DARK, ui.WARNING, 0.95, pad=14) for w in r.warnings]
        if not r.warnings:
            notes.append(ui.pill("Quality OK", 14, "semibold", dot=ui.SUCCESS))
        for p in notes:
            p.draw(canvas, x, y)
            y += p.height + 6

    def draw_right_column(self, canvas: np.ndarray, sign: hc.Sign | None) -> None:
        """Guides toggle, category pill (opens the category grid) and the sign chips."""
        m, gd = ui.MARGIN, 44
        gx = ui.UI_W - m - gd
        on = self.show_guides
        ui.round_button("settings", gd, bg=ui.PRIMARY if on else ui.DARK,
                        bg_alpha=0.9 if on else ui.PANEL_ALPHA).draw(canvas, gx, m)
        self.hits.add(gx, m, gd, gd, "key:g")
        r = self.current_ref
        text = (r.category_title if r else "Reference videos") if self.ref_mode \
            else store.category_title(self.category or "")
        cat = ui.pill(text, 16, "semibold", height=gd, pad=20)
        cx = gx - 10 - cat.width
        cat.draw(canvas, cx, m)
        self.hits.add(cx, m, cat.width, cat.height, "ref_signs" if self.ref_mode else "crumb:signer")
        self.draw_chips(canvas, m + gd + 14, ui.UI_H - 160)

    def draw_chips(self, canvas: np.ndarray, top: int, bottom: int) -> None:
        """Clickable, scrollable list of the signs: done (tick), current (blue), upcoming (dim)."""
        right, step, d = ui.UI_W - ui.MARGIN, 36, 30
        # Reference mode lists only the current category; otherwise every sign of the open category.
        items = self.ref_folder_indices(self.ref_folder) if self.ref_mode else list(range(len(self.signs)))
        total = len(items)
        cur = items.index(self.sign_index) if self.sign_index in items else -1
        fit = max(3, (bottom - top) // step)
        scrollable = total > fit
        visible = fit - 2 if scrollable else total       # room for the up/down buttons
        if scrollable and getattr(self, "_chips_follow", False) and cur >= 0:
            first = self.scroll.get("chips", 0)          # keep the current sign in view
            if cur < first:
                self.scroll["chips"] = cur
            elif cur >= first + visible:
                self.scroll["chips"] = cur - visible + 1
            self._chips_follow = False
        first = self._scroll_window("chips", total, visible) if scrollable else 0
        self.scroll_regions.append((right - 260, top, right, bottom, "chips"))
        y = top
        if scrollable:
            ui.round_button("up", d, bg_alpha=0.7).draw(canvas, right - d, y, 1.0 if first > 0 else 0.3)
            if first > 0:
                self.hits.add(right - d, y, d, d, "scroll:chips:-1")
            y += step
        for p in range(first, min(total, first + visible)):
            i = items[p]
            s = self.signs[i]
            state = "current" if self.ref_mode and p == cur else \
                "done" if self.is_complete(s) else "current" if p == cur else "next"
            c = ui.chip(self.refs[i].title if self.ref_mode else s.sign_label, state)
            c.draw(canvas, right - c.width, y)
            self.hits.add(right - c.width, y, c.width, c.height, f"chip:{i}")
            y += step
        if scrollable:
            more = first + visible < total
            ui.round_button("down", d, bg_alpha=0.7).draw(canvas, right - d, y, 1.0 if more else 0.3)
            if more:
                self.hits.add(right - d, y, d, d, "scroll:chips:1")

    def draw_center(self, canvas: np.ndarray) -> None:
        """REC / replay pill, countdown, paused and finished overlays."""
        cx, cy = ui.UI_W // 2, ui.UI_H // 2
        if self.state == "RECORDING":
            self.draw_recording_hud(canvas)
        elif self.state == "REVIEW":
            p = ui.pill("Replay", 15, "semibold", dot=ui.PRIMARY_LIGHT, height=38, pad=18)
            p.draw(canvas, cx - p.width // 2, ui.MARGIN)
        if self.state == "COUNTDOWN":
            n = max(1, COUNTDOWN_SECONDS - int(time.perf_counter() - self.countdown_start))
            b = ui.countdown_badge(n)
            b.draw(canvas, cx - b.width // 2, cy - b.height // 2)
            # Recording rule: every take starts (and ends) with the hands down, out of view.
            h = ui.pill("Hands down", 22, "semibold", ui.DARK, ui.WARNING, 0.95, height=50, pad=26, icon="down")
            h.draw(canvas, cx - h.width // 2, cy + b.height // 2 + 18)
        if self.paused:
            p = ui.pill("Paused", 30, "semibold", height=72, pad=40, bg_alpha=0.65)
            p.draw(canvas, cx - p.width // 2, cy - p.height // 2)
        if self.state == "FINISHED":
            left = sum(1 for s in self.signs if not self.is_complete(s))
            if self.ref_mode:
                text = "All reference videos recorded" if left == 0 else f"End of list · {left} sign(s) pending"
            else:
                text = "Category complete" if left == 0 else f"End of list · {left} sign(s) still need takes"
            p = ui.pill(text, 26, "semibold", dot=ui.SUCCESS if left == 0 else ui.WARNING, height=68, pad=34,
                        bg_alpha=0.7)
            p.draw(canvas, cx - p.width // 2, cy - p.height // 2)

    def draw_recording_hud(self, canvas: np.ndarray) -> None:
        """Pulsing REC pill + progress bar (top centre) and the caption."""
        done = self.job.frames_captured if self.job else 0
        frac = self.job.elapsed_fraction(time.perf_counter()) if self.job else 0.0
        pulse = 0.5 + 0.5 * math.sin(time.perf_counter() * 2 * math.pi * 1.2)
        rec = ui.pill("REC", 17, "bold", dot=ui.DANGER, dot_alpha=round(0.3 + 0.7 * pulse, 1),
                      height=40, pad=18, bg_alpha=0.6)
        rec.draw(canvas, (ui.UI_W - rec.width) // 2, ui.MARGIN)
        bar = ui.progress_bar(REC_BAR_W, 8, round(frac, 2), colors=(ui.DANGER, (255, 150, 90)))
        bar.draw(canvas, (ui.UI_W - REC_BAR_W) // 2, ui.MARGIN + rec.height + 12)
        self.draw_caption(canvas, f"Recording…  {frac * self.clip_seconds:.1f} / {self.clip_seconds:g} s  ·  {done} frames")

    def caption_text(self) -> str:
        sign = self.sign
        if self.ref_mode:
            if self.state == "FINISHED":
                return "B previous sign  ·  click a sign on the right  ·  Esc sign list"
            if self.state == "REVIEW":
                return "K keep  ·  R redo  ·  N next sign  ·  B previous sign"
            if self.state == "COUNTDOWN":
                return "Hands down — start from rest, end at rest"
            if self.paused:
                return "Paused — press P to resume"
            if sign is not None and self.is_complete(sign):
                return "SPACE record again  ·  V record version 2  ·  N next sign  ·  B previous sign"
            return "SPACE record  ·  N next sign  ·  B previous sign"
        if self.state == "FINISHED":
            return "Click a sign on the right  ·  Esc back"
        if self.state == "REVIEW":
            return "Keep or retake?"
        if self.state == "COUNTDOWN":
            return "Hands down — start from rest, end at rest"
        if self.paused:
            return "Paused — press P to resume"
        if sign is not None and self.kept_for(sign) >= sign.reps:
            return "This sign is complete — pick another sign"
        return "Press SPACE to record"

    def draw_bottom(self, canvas: np.ndarray) -> None:
        """Caption, notices, Record / Skip / Pause, camera status, Keep / Retake."""
        m = ui.MARGIN
        cap_y = ui.UI_H - m - 46
        if self.state != "RECORDING":                     # recording draws its own caption
            cap_y = self.draw_caption(canvas, self.caption_text())

        # Keep / Retake: active during review only, dimmed otherwise.
        active = self.state == "REVIEW"
        opacity = 1.0 if active else 0.3
        kx, ky = ui.UI_W - m - KEEP_W, ui.UI_H - m - KEEP_H
        ui.keep_button(KEEP_W, KEEP_H).draw(canvas, kx, ky, opacity)
        rx = kx - 12 - KEEP_H
        ui.round_button("retake", KEEP_H, bg_alpha=0.6).draw(canvas, rx, ky, opacity)
        ui.key_badge("R").draw(canvas, rx + KEEP_H - 18, ky - 4, opacity)
        if active:
            self.hits.add(kx, ky, KEEP_W, KEEP_H, "key:k")
            self.hits.add(rx, ky, KEEP_H, KEEP_H, "key:r")

        # Record / Skip / Pause buttons (same as SPACE / S / P), bottom-left.
        sign = self.sign
        can_record = (self.state == "LIVE" and not self.paused and sign is not None
                      and (self.ref_mode or self.kept_for(sign) < sign.reps))
        record = (ui.pill("Record", 16, "semibold", ui.WHITE, ui.DANGER, 0.92, height=44, pad=18, icon="record"),
                  "Space", "key: ", can_record)
        pause = (ui.pill("Resume" if self.paused else "Pause", 16, "medium", height=44, pad=18, bg_alpha=0.65,
                         icon="play" if self.paused else "pause"),
                 "P", "key:p", self.state in ("LIVE", "COUNTDOWN"))
        if self.ref_mode:                                 # N / B instead of Skip
            step_ok = self.state in ("LIVE", "REVIEW", "FINISHED")
            buttons = [
                record,
                (ui.pill("Previous", 16, "medium", height=44, pad=18, bg_alpha=0.65), "B", "key:b", step_ok),
                (ui.pill("Next", 16, "medium", height=44, pad=18, bg_alpha=0.65, icon="skip"), "N", "key:n",
                 step_ok and sign is not None),
                pause,
            ]
        else:
            buttons = [
                record,
                (ui.pill("Skip", 16, "medium", height=44, pad=18, bg_alpha=0.65, icon="skip"),
                 "S", "key:s", self.state in ("LIVE", "REVIEW") and sign is not None),
                pause,
            ]
        x, y = m, ui.UI_H - m - 28 - 12 - 44
        for sprite, badge, action, ok in buttons:
            op = 1.0 if ok else 0.3
            sprite.draw(canvas, x, y, op)
            b = ui.key_badge(badge)
            b.draw(canvas, x + sprite.width - b.width + 6, y - 8, op)
            if ok:
                self.hits.add(x, y, sprite.width, sprite.height, action)
            x += sprite.width + 12

        # Real camera resolution / fps (always visible), bottom-left.
        fps = self.worker.fps()
        ok = not self.standard_warnings and fps >= hc.MIN_ACCEPTABLE_FPS
        cam = ui.pill(f"{self.width}×{self.height} · {fps:.0f} fps · kept {self.session_kept}",
                      13, height=28, pad=12, dot=ui.SUCCESS if ok else ui.DANGER)
        cam.draw(canvas, m, ui.UI_H - m - cam.height)
        self.draw_notices(canvas, cap_y, warnings=True)

    # ----- cleanup ----------------------------------------------------------
    def shutdown(self, reason: str) -> None:
        """Stop everything; remove any take that was not kept."""
        if self.job is not None and not self.job.done.is_set():
            self.job.abort("Shutdown")
            self.job.done.wait(3.0)
        if self.job is not None:
            self.log_write("Unsaved take discarded on exit.")
        self.discard_pending_take()
        self.close_player()
        self.worker.stop()
        self.cap.release()
        self.landmarker.close()
        cv2.destroyAllWindows()
        self.exit_reference()
        if self.log is not None:
            self.log.close(reason)


# =============================================================================
# STARTUP (arguments are optional shortcuts; settings are remembered)
# =============================================================================
def parse_args(categories: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HUDYAT FSL recording tool. With no arguments it opens the dashboard.")
    parser.add_argument("--signer", type=str.upper, choices=store.SIGNER_CODES,
                        help="shortcut: open this signer directly")
    parser.add_argument("--hand", type=str.lower, choices=store.HANDS,
                        help="set/overwrite the signer's signing hand in signers.csv")
    parser.add_argument("--category", choices=categories + ["all"],
                        help="shortcut (with --signer): jump straight to recording this category")
    parser.add_argument("--camera", type=int, default=None, help="camera index (remembered)")
    parser.add_argument("--camera-name", default=None, help='e.g. "Logitech C920" (remembered)')
    parser.add_argument("--backend", choices=list(hc.CAMERA_BACKENDS), default=None,
                        help="camera backend (remembered); compare them with --list-cameras")
    parser.add_argument("--list-cameras", action="store_true",
                        help="list cameras (index, name, resolution, measured fps) and exit")
    parser.add_argument("--reference", action="store_true",
                        help="Reference mode: one reference video per sign for the expert validation form")
    parser.add_argument("--ref-seconds", type=float, default=ref.REF_SECONDS,
                        help=f"length of a reference take in seconds (default {ref.REF_SECONDS:g})")
    args = parser.parse_args()
    if not 1.0 <= args.ref_seconds <= 10.0:
        parser.error("--ref-seconds must be between 1 and 10")
    return args


def windows_camera_names() -> list[str]:
    """Names of the cameras Windows knows about (best effort; [] if unavailable).

    OpenCV cannot report names, and Windows does not promise the same order as
    the camera indexes, so these are printed as a separate list.
    """
    if sys.platform != "win32":
        return []
    import subprocess
    command = ("Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -in 'Camera','Image' "
               "-and $_.Status -eq 'OK' } | ForEach-Object { $_.Name }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def list_cameras_report(backend: str) -> int:
    """--list-cameras: index, resolution and measured fps of every camera."""
    names = windows_camera_names()
    if names:
        print("Cameras known to Windows (order may differ from the indexes below):")
        for name in names:
            print(f"  - {name}")
    remembered = store.load_settings()
    print(f"\nProbing camera indexes 0-{hc.MAX_CAMERA_INDEX} with backend {backend} "
          f"at {hc.TARGET_WIDTH}x{hc.TARGET_HEIGHT} (about {LIST_FPS_SECONDS:.0f} s each)...")
    found = hc.list_cameras(backend=backend)
    if not found:
        print("No camera found. Close Zoom/Teams/the Camera app and check Windows camera privacy settings.")
        return 1
    print(f"{'Index':>5} | {'Name':<24} | {'Resolution':>10} | {'Fps':>5} | Note")
    print("-" * 70)
    for index in found:
        cap = hc.open_camera(index, backend=backend)
        size = hc.read_actual_size(cap) if cap is not None else None
        if cap is None or size is None:
            print(f"{index:>5} | {'':<24} | {'-':>10} | {'-':>5} | could not read frames")
            if cap is not None:
                cap.release()
            continue
        fps, _ = hc.measure_fps(cap, LIST_FPS_SECONDS)
        ok, frame = cap.read()                           # after warm-up: exposure has settled
        cap.release()
        name = remembered["camera_name"] if index == remembered["camera"] else ""
        notes = []
        if size != (hc.TARGET_WIDTH, hc.TARGET_HEIGHT):
            notes.append(f"not {hc.TARGET_WIDTH}x{hc.TARGET_HEIGHT}")
        if fps < hc.MIN_ACCEPTABLE_FPS:
            notes.append("low fps (recording still works, 3 s per take)")
        if ok and hc.mean_brightness(frame) < 5:
            notes.append("almost black: probably an infrared face-login camera")
        print(f"{index:>5} | {name[:24]:<24} | {f'{size[0]}x{size[1]}':>10} | {fps:5.1f} | "
              f"{'; '.join(notes) or 'OK'}")
    print("\nRecord with:  python record_tool.py --camera <index> --camera-name \"<model>\" "
          f"--backend {backend}")
    return 0


def select_signs(all_signs: list[hc.Sign], category: str) -> list[hc.Sign]:
    if category == "all":
        return list(all_signs)
    return [s for s in all_signs if s.category == category]


def main() -> int:
    all_signs = hc.load_signs()
    args = parse_args(hc.list_categories(all_signs))
    if args.list_cameras:
        return list_cameras_report(args.backend or store.load_settings()["backend"])
    if args.category and not args.signer:
        print("NOTE: --category only works together with --signer; opening the dashboard.")
        args.category = None
    if args.reference:                                   # check the sign list before opening the camera
        try:
            ref.reference_signs(all_signs)
        except ValueError as exc:
            print(f"Reference mode stopped: {exc}")
            return 1

    # Camera settings: command line > settings.json > defaults. Remember what was used.
    settings = store.load_settings()
    for key, value in (("camera", args.camera), ("camera_name", args.camera_name), ("backend", args.backend)):
        if value is not None:
            settings[key] = value
    store.save_settings(settings)

    if not args.reference:                               # reference mode never touches dataset/
        hc.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Opening camera {settings['camera']} ({settings['backend']}) at "
          f"{hc.TARGET_WIDTH}x{hc.TARGET_HEIGHT} @ {hc.TARGET_FPS} fps...")
    cap = hc.open_camera(settings["camera"], backend=settings["backend"])
    size = hc.read_actual_size(cap) if cap is not None else None
    if cap is None or size is None:
        print(f"Could not open camera {settings['camera']}. Run  python record_tool.py --list-cameras  "
              "to find the right index,")
        print("then start with  --camera <index>  (it is remembered).")
        if cap is not None:
            cap.release()
        return 1

    app = App(all_signs, cap, size, settings)
    app.ref_seconds = args.ref_seconds
    if args.signer and args.hand:                        # --hand updates signers.csv
        app.signers[args.signer].hand = args.hand
        store.save_signers(app.signers)
    reason = "error"
    try:
        app.start_camera()
        if args.reference:
            app.enter_reference()
        elif args.signer:                                # command-line shortcuts
            app.select_signer(args.signer)
            if args.category:
                if app.modal:                            # hand unknown: continue after the answer
                    app.pending_category = args.category
                else:
                    app.open_category(args.category)
                    app.open_record(app.first_incomplete_index())
        reason = app.run()
    except KeyboardInterrupt:
        reason = "Ctrl+C"
        print("\nCtrl+C received - stopping safely.")
    finally:
        app.shutdown(reason)
    t = app.totals
    print(f"\nSession ended ({reason}). Kept {t['kept']}, retaken {t['retaken']}, skipped {t['skipped']}.")
    if args.reference or app.refs:
        print(f"Reference videos: {ref.validation_dir().relative_to(hc.PROJECT_DIR)}  "
              f"(log: {(ref.REF_DIR / 'reference_log.txt').relative_to(hc.PROJECT_DIR)})")
    if not args.reference:
        print(f"Session logs: {hc.SESSION_LOG_DIR.relative_to(hc.PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:          # Ctrl+C while answering a console question
        print("\nCancelled.")
        sys.exit(1)
