# HUDYAT Hand Tracker: FSL Data-Collection Tools

Tools for recording Filipino Sign Language (FSL) clips for the HUDYAT project
and importing the FSL-105 public dataset. Preprocessing (landmark extraction)
and model training are **not** part of this folder; they live in a separate
project that reads this folder's `dataset/`.

**The process, in order** (details in [section 10](#10-the-data-collection-process)):

| Step | Script | What it does |
|---|---|---|
| 1 | `record_tool.py` | Record S1–S7 (camera list: `--list-cameras`). |
| 2 | `import_public.py` | Import the FSL-105 public dataset as signer `PUB`. |
| 3 | *(manual)* | FSL experts validate the clips (`expert_status` column). |

| File | What it does |
|---|---|
| `record_tool.py` | The recording application: dashboard (signers, categories, signs, saved takes, progress overview) and recording screen in one window. `--list-cameras` lists the cameras with their real resolution and fps. |
| `import_public.py` | Converts the FSL-105 `.mov` files and adds them to `dataset/` as signer `PUB`, using `public_mapping.csv`. |
| `hudyat_common.py` | Shared settings and code (recording standard, orientation/handedness rule, sign list loader). |
| `hudyat_ui.py` | The on-screen interface of the recorder (display only; never touches saved video). |
| `hudyat_store.py` | Reads/writes `signers.csv` and `settings.json`, computes progress from `metadata.csv`, flags takes, and defines the `metadata.csv` columns. |
| `signers.csv` | One row per signer: `code, hand, display_name_optional, notes`. Nothing else about a signer is stored. |
| `settings.json` | Remembered camera index, camera name and backend (created on first run). |
| `public_mapping.csv` | Which FSL-105 label becomes which of our signs: `fsl105_id, fsl105_label, category, sign_id`. |
| `assets/fonts/` | Poppins font (SIL Open Font License, see `OFL.txt`) used for the on-screen text. |
| `signs.csv` | The sign list (110 rows). |
| `references/` | Expert reference videos, named `<sign_id>.mp4` (for the team and the experts; not shown in the recorder). |
| `tests/` | Unit tests: `python -m pytest`. |
| `requirements.txt` | Python packages. |

Everything runs **offline** once setup is done. No script uploads anything.

---

## 1. Setup (once per laptop)

You need **Python 3.12 (64-bit)** from python.org. Check with `py -0p`; the
list should contain `-V:3.12`.

Open **Command Prompt** (`cmd`) and run these commands one at a time:

```bat
cd "%USERPROFILE%\Documents\HUDYAT - HAND TRACKER"

:: 1. Create the virtual environment (a private Python just for this project)
py -3.12 -m venv venv

:: 2. Activate it (do this every time you open a new terminal)
venv\Scripts\activate.bat

:: 3. Install the pinned libraries
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. Download the MediaPipe hand model (about 7.5 MB) into models\
if not exist models mkdir models
curl.exe -L -o models\hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Using **PowerShell** instead? Steps 1, 3 and 4 are the same (use
`mkdir models -Force` for the folder). Activate with:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell says "running scripts is disabled", run this once (it affects
only your Windows user):
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Make every laptop (and the browser app) use the same model file.** The URL
above always gives the *latest* model. The copy used during development has
this SHA-256:

```
fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
```

Check a copy with `certutil -hashfile models\hand_landmarker.task SHA256`. The
safest approach is to copy this one file to every laptop and into the web app,
rather than downloading it again.

Pinned versions (`requirements.txt`): `mediapipe==0.10.21`,
`opencv-contrib-python==4.11.0.86`, `numpy==1.26.4`, `pillow==12.3.0` (for
the smooth on-screen text), `imageio-ffmpeg` (ffmpeg for `import_public.py`)
and `pytest`. Do **not** also install `opencv-python`: two OpenCV packages in
one venv break `cv2`.

---

## 2. Before every session

### 2.1 Check the camera

```powershell
.\venv\Scripts\Activate.ps1
python record_tool.py --list-cameras
```

It prints the cameras Windows knows about, then for every working camera
index (0–5): the name (if it is the one remembered in `settings.json`), the
resolution it **actually** delivers at the 1280×720 request, and the fps
measured over 3 seconds, using the `msmf` backend. Then it exits. Example:

```
Index | Name                     | Resolution |   Fps | Note
    0 | Built-in                 |   1280x720 |  29.7 | OK
    1 |                          |   1280x720 |  30.1 | almost black: probably an infrared face-login camera
```

The built-in camera is usually `0` and a USB webcam usually `1`. A camera
reported as "almost black" is an infrared (face login) sensor, or a phone
webcam app that is not connected; don't use it.

**You want `1280x720` at about 30 fps.** Lower fps does not stop recording
(every take is still exactly 3 seconds, see section 5), but it gives fewer
frames per take, so fix it if you can: see [Troubleshooting](#8-troubleshooting).

The live setup check (framing guides, hand landmarks tagged Left/Right, light
level) is the recorder's own recording screen: open a signer and a sign and
look before pressing SPACE.

> **Camera backend.** Windows has two camera APIs. On the development laptop,
> `msmf` (Media Foundation, the default) gave 1280×720 at **29.7 fps**, while
> `dshow` (DirectShow) gave only **8 fps** on the same camera. Some USB webcams
> are the other way round. Compare with `python record_tool.py --list-cameras --backend dshow`
> and pass the faster `--backend` to `record_tool.py` (it is remembered).

### 2.2 Setup checklist

- [ ] **Green cloth** pulled tight, no folds or shadows, filling the whole background.
- [ ] Signer seated about **1 m in front of the cloth** (this avoids green light reflecting onto the hands and stops shadows falling on the cloth).
- [ ] Camera about **1 m from the signer**, at chest/shoulder height, pointing straight at them.
- [ ] **Front lighting**: lights behind or beside the camera, facing the signer. No window behind the signer.
- [ ] Brightness indicator says **OK** (not TOO DARK / TOO BRIGHT).
- [ ] Framing guides: **top of head** near the top corner brackets, **waist** near the bottom brackets, body centered between the small center ticks. Hands must stay inside the frame while signing.
- [ ] Raise your right hand: the preview labels it **Right**; then the left hand: **Left**.
- [ ] **Hands-down rule explained to the signer:** every take starts and ends with **both hands down, resting below the camera view** (on the lap), like the FSL-105 signer. Raise the hands only after "REC" appears, sign once, then lower them out of view before the 3 seconds end. Check it: with the hands resting, the `check_camera.py` preview shows **Hands detected: 0**. Extraction cuts each clip to the part where hands are visible (section 6.4), so idle frames at both ends are needed.
- [ ] Signer wears plain, long sleeves that are **not green**, with no rings, bracelets or watch.
- [ ] Other apps that use the camera (Zoom, Teams, Camera app) are closed.

---

## 3. Running a recording session

### 3.1 Start

```bat
venv\Scripts\activate.bat
python record_tool.py
```

The window opens on the **dashboard**. Everything else is done by clicking.

**First time on a laptop**, tell it which camera to use. The settings are saved
in `settings.json`, so you only type them once:

```bat
python record_tool.py --camera 0 --camera-name "Logitech C920"
```

| Optional argument | Meaning |
|---|---|
| `--camera` | camera index (remembered) |
| `--camera-name` | free text saved in the metadata, e.g. the webcam model (remembered) |
| `--backend` | `msmf` or `dshow`; see section 2.1 (remembered) |
| `--list-cameras` | list cameras (index, name, resolution, measured fps) and exit; see section 2.1 |
| `--signer S1` | shortcut: open S1's categories directly |
| `--signer S1 --category shapes` | shortcut: jump straight to recording that category (the first sign that still needs takes) |
| `--hand left` / `--hand right` | together with `--signer`: set that signer's hand in `signers.csv` |

`--category all` (every category) also works as a shortcut.

At startup the tool opens the camera, measures the real fps for 2 seconds,
and keeps the camera open while you move between screens. If the camera is not
delivering 1280×720, or fps is below 25, a **yellow warning** stays on the
recording screen and is written to the session log. A low fps is only a
warning: recording still works and every take is still exactly 3 seconds.

Signers are **S1–S7**. Each signer records with their natural dominant hand
(asked once, or `--hand left|right`; stored in `signers.csv` and in every
`metadata.csv` row). Never half left / half right. `PUB` (the public dataset)
is never recorded with this tool; it comes from `import_public.py`.

Targets per signer (shown as progress everywhere): **15 takes** for each of
each of the 110 signs: the 84 word signs and the 26 letters A–Z (category
`alphabet`, sign ids `letter_a` … `letter_z`). Every signer records every
category: 1,650 clips per signer.

### 3.2 The dashboard screens

All screens share the same look: the live camera blurred in the background,
**breadcrumbs** at the top (for example `Signers › S3 › Shapes › Circle`, and
each one is clickable to go back), a **back** button (Esc), **Progress
overview**, and **Quit** (Q).

1. **Signers**: seven cards, S1–S7. Each card shows the code, the signing hand,
   and overall progress
   (clips kept / clips required, with a bar). If a signer's hand is not set
   yet, clicking the card asks **Left hand / Right hand** once and saves it
   in `signers.csv`.
2. **Categories** for the chosen signer: one tile per category with kept /
   required (e.g. `45 / 150`). A tile turns **green with a tick** when it is
   complete.
3. **Signs** in the category: each sign shows `kept / 15` (✓ green when done),
   a "2 hands" tag for two-handed signs and a yellow "N flagged" tag if takes
   were flagged. **Click a sign** to open the recording screen at that sign.
   Click its **list icon** to see its saved takes.
4. **Saved takes** of one sign: one row per take (take number, time, hand %,
   both-hands %, duration, status). Click a take to **replay** it (mirror view)
   with its quality info. In the replay you can **Flag: needs re-record** (or
   undo it); see 3.6. "Record this sign" jumps to the recording screen.
5. **Progress overview** (blue button at the top): a table of signers (rows) ×
   categories (columns) with kept/required in each cell. Green = complete,
   blue = in progress (darker blue = less done), dark = not started. Click a cell to open that signer's
   category, or a signer's name to open that signer. It is read fresh from
   `metadata.csv` every time.

### 3.3 The recording screen

The live camera fills the window as a mirror (selfie) view, with translucent
controls on top:

| Where | What |
|---|---|
| Top-left | **Back** button (Esc) → sign list, and the breadcrumbs. Under them: the **sign name** in big letters, a blue progress bar of kept takes for this sign, and pills for **Signer** (click to switch signer), **Take X / N** and **Hands: 1 or 2**. The Hands dot turns **green** when the required number of hands is detected. A yellow pill appears if the scene is too dark or too bright. |
| Top-right | **Category** pill (click to open the category grid) and the **guides** button (G). Under them: the clickable list of the category's signs (✓ green = complete, blue = current, dim = still to do). Click any sign to jump to it. Long lists scroll with the mouse wheel or the small ⌃/⌄ buttons. |
| Middle | Thin **corner brackets** mark the head-to-waist area (G hides them). Only **hand** landmarks are drawn (red, on both hands). No face model runs at all. |
| Bottom-left | **Record** (Space), **Skip** (S) and **Pause** (P) buttons, and below them the real camera resolution / fps plus clips kept this session. |
| Bottom | A **caption** says what to do next ("Press SPACE to record", "Recording…", "Keep or retake?"). Messages and camera warnings appear just above it. |
| Bottom-right | The blue **Keep (K)** and round **Retake (R)** buttons, active only during review. |

Everything on the screen is drawn on a display copy only: **the saved video is
always the plain, unmirrored camera image**, with no text, landmarks or buttons.

**You cannot leave the recording screen, switch sign or switch signer while a
take is recording or waiting for Keep / Retake.** Finish it (K) or discard it
(R) first. Leaving during the 3-2-1 countdown cancels the countdown.

### 3.4 Per sign

1. Check the **sign name**, **Take X / N** and **Hands** pills.
2. Signer rests both hands **down, out of the camera view**. The operator presses **SPACE** (or clicks **Record**). A **3-2-1** countdown appears in a circle with a yellow **Hands down** reminder. The signer raises the hands only after "REC" appears, signs once, and lowers them out of view again before the take ends. Then the tool records exactly **3 seconds** (90 frames at 30 fps; fewer on a slower camera, e.g. 45 at 15 fps). During the take the screen is kept minimal (a pulsing **REC** pill, a red progress bar and the caption), so the computer can concentrate on capturing every frame.
3. **Take review**: the clip replays in the main view with the hand landmarks, and a **Take quality** card shows:
   - **Hand detection** (`hand_detect_rate`): % of frames with at least one hand, measured **inside the sign window** (the part extraction keeps, section 6.4), so the hands-down frames at both ends don't lower it;
   - **Both hands** (`both_hands_rate`): % of sign-window frames with two hands (highlighted for 2-hand signs);
   - **Sign window**: how long the kept part is, in seconds;
   - **Duration**: the real duration of the whole take in seconds.
4. Yellow warning pills suggest a retake when:
   - `hand_detect_rate < 80%`, or, for 2-hand signs, `both_hands_rate < 80%`;
   - **"Hand visible at the start / end. Start and end with hands down."** A hand was detected in the first or last **0.3 s** of the take. Retake it, because the trim needs idle frames at both ends.

   Nothing is ever deleted automatically; the operator decides.
5. Press **K** to keep or **R** to retake (or click the buttons).
6. When a sign has all its kept takes, the tool moves to the next sign by itself. A sign that is already complete cannot be recorded again; pick another sign.

### 3.5 Keyboard controls

Click the window first so it receives the key presses. Every key also has an
on-screen button.

| Key | Recording screen (live) | Take review | Other screens |
|---|---|---|---|
| **SPACE** | countdown + record one take | – | – |
| **K** | – | **keep** the take (saves the file and the metadata row) | – |
| **R** | – | **retake**: discard this take and record it again | – |
| **S** | skip this sign | discard this take and skip the sign | – |
| **P** | pause / resume (SPACE is disabled while paused) | – | – |
| **G** | show / hide the framing guides | show / hide the framing guides | – |
| **Esc** | back to the sign list | – (keep or retake first) | back one screen |
| **Q** | save progress and quit | quit; **this take is NOT saved** (press K first if you want it) | quit |

Closing the window, pressing **Ctrl+C** in the terminal, or unplugging the
camera are all handled safely. The unfinished take is deleted, and every
kept clip and metadata row is already on disk. Start the tool again to resume:
progress is always read from `metadata.csv`, per signer and per sign.

### 3.6 Flagging a saved take ("Needs re-record")

When you (or an FSL expert) find a bad kept take, open it in **Saved takes**,
replay it, and click **Flag: needs re-record**.

- Only the `status` column of that row in `metadata.csv` changes, from `kept`
  to `needs_rerecord`. **No file is deleted, renamed or moved.**
- A flagged take **does not count** toward the 15, so the sign shows one take
  missing and gets re-recorded. The new take gets a new number; the flagged
  file's number is never reused.
- **Unflag** puts the status back to `kept`.
- Flagged videos stay on disk; preprocessing must use only rows with
  `status == kept`.

### 3.7 What gets saved

- Kept clip: `dataset/raw/<category>/<sign_id>/<signer>/<sign_id>_<signer>_<take>.mp4`, e.g. `dataset/raw/shapes/circle/S1/circle_S1_07.mp4`.
- Take numbers start at `01`, continue across sessions, and **existing files are never overwritten**. "Take X / N" on screen counts kept takes; the number in the file name is shown in the "Saved …" message after you press K. They can differ if an orphan file is found on disk.
- A take is first written as `…_07.tmp.mp4` and only gets its final name when you press **K**. The row is added to `dataset/metadata.csv` **after** the file is complete and closed.
- Codec: `mp4v` in `.mp4`. If that is not available, the fallback is `XVID` in `.avi`. The codec used is recorded in `metadata.csv`.
- Session log: `dataset/session_logs/<signer>_<YYYY-MM-DD_HHMM>.txt`, with the start/end time, camera details, every keep/retake/skip/flag, and the totals. Switching to another signer closes that signer's log and starts one for the new signer.
- `status` is `kept` for every new take, or `needs_rerecord` after flagging (section 3.6). Only `kept` takes count toward a sign's 15.

`metadata.csv` columns:
`timestamp, signer, signing_hand, category, sign_id, sign_label, hands_required, take, file_path, camera_index, camera_name, width, height, fps_actual, n_frames, duration_s, codec, orientation, hand_detect_rate, both_hands_rate, mean_brightness, status`

- `fps_actual` and `duration_s` are measured from the real capture time of each frame; `n_frames` is the number of frames in the clip (90 at 30 fps, fewer on a slower camera; `duration_s` is ≈ 3.0 either way).
- Rows with `signer = PUB` come from `import_public.py`: `camera_name` holds the source file (`FSL-105:<path>`), and the quality columns are empty (the hand detection rate is measured later, during landmark extraction).
- Optional column for step 3 (expert validation): `expert_status` = `approved` / `rejected`. Add it in Excel; `rejected` clips must be left out of training.
- `hand_detect_rate` and `both_hands_rate` are fractions from 0 to 1 (`0.950` = 95%).
- `orientation` is always `camera_raw_unmirrored` (see section 6).

### 3.8 After recording

Landmark extraction, splitting and training are done by the separate
preprocessing/training project, which reads `dataset/raw/` and
`dataset/metadata.csv` from this folder. Section 6 describes the landmark
format that project (and the browser app) must follow.

---

## 4. Folder structure

```
HUDYAT - HAND TRACKER/
├── record_tool.py             1  record
├── import_public.py           2  import FSL-105 as PUB
├── hudyat_common.py           shared settings + code
├── hudyat_ui.py               on-screen interface (display only)
├── hudyat_store.py            signers / settings / progress / flagging / metadata columns
├── signers.csv                code, hand, display_name_optional, notes
├── settings.json              remembered camera settings (created on first run)
├── public_mapping.csv         fsl105_id, fsl105_label, category, sign_id
├── assets/
│   └── fonts/                 Poppins-*.ttf + OFL.txt (on-screen font)
├── signs.csv                  sign list (edit 'hands' and 'mirror_ok' after expert review)
├── requirements.txt
├── README.md
├── tests/                     unit tests (python -m pytest)
├── models/
│   └── hand_landmarker.task   MediaPipe model (downloaded at setup)
├── references/
│   ├── README.txt
│   └── <sign_id>.mp4          expert reference clips (optional, not shown in the recorder)
├── venv/                      Python environment (created at setup)
└── dataset/
    ├── metadata.csv           one row per KEPT clip (recorded and imported)
    ├── landmarks_index.csv    one row per .npy sample (from an earlier extraction)
    ├── session_logs/
    │   └── S1_2026-10-12_0930.txt
    ├── raw/
    │   └── <category>/<sign_id>/<signer>/<sign_id>_<signer>_<NN>.mp4     (signer S1..S7 or PUB)
    └── landmarks/
        └── <category>/<sign_id>/<signer>/<sign_id>_<signer>_<NN>.npy   (earlier extraction; kept as data)
```

`signs.csv` columns are `category,sign_id,sign_label,reps,hands,mirror_ok`.
`reps` is 15 for every sign. `hands`
defaults to 1; **set it to 2 for two-handed signs once the FSL experts confirm
them**. This turns on the `both_hands_rate` check. `mirror_ok` defaults to 0;
set it to **1 only for signs the FSL experts approve for left-handed
(mirrored) performance**; training may then use mirrored copies of them. Don't rename a `sign_id` after recording has started, because it is
part of every file name.

The data split is done later, in the preprocessing/training project. Signers
are **not** assigned to training, validation or testing while recording: all
of S1–S7 record every sign. The model is evaluated in two ways:

1. **Random split, 70 / 15 / 15.** All kept clips are shuffled into 70 %
   training, 15 % validation and 15 % testing, stratified by `sign_id` so every
   sign has the same proportions in each part. Use a fixed random seed so the
   split can be repeated.
2. **Leave-one-signer-out (LOSO).** Seven rounds, one per signer: one of
   S1–S7 is the test set and the other six signers are used for training (and
   validation). The result is reported per held-out signer and as the average
   of the seven rounds. This measures how well the model works on a person it
   has never seen.

In both tests, `PUB` clips are used for training only. Mirrored copies of a
clip always go into the same part as the original clip, and clips with
`expert_status = rejected` are left out.

---

## 5. Recording standard

| Setting | Value |
|---|---|
| Resolution | **1280 × 720** (16:9) |
| Camera backend | **MSMF** (Media Foundation) by default; `--backend dshow` to change |
| Frame rate | **30 fps** requested; the real fps is measured and saved per clip |
| Clip length | exactly **3.0 seconds**, counted in time, not frames (90 frames at 30 fps, 45 at 15 fps) |
| Preview | mirrored (selfie view); the saved video is raw, unmirrored, without any overlay |
| Container / codec | MP4 / `mp4v` (fallback AVI / `XVID`) |
| Saved orientation | **unmirrored** (as the camera sees it) |
| Low-fps warning | below 25 fps (a warning only; it never blocks recording) |
| Retake warning | hand detection < 80% (2-hand signs: both hands < 80%) |

All of these are constants at the top of `hudyat_common.py` (shared) or of
each script (script-specific), so the team can change them in one place.

The tool never silently records at a different setting. If the camera
delivers something else, a warning stays on screen and the real width,
height, fps and frame count are stored for every clip.

Why time-based: some laptop cameras only reach 15–19 fps in dim light. A
fixed 90-frame take would then last 5–6 seconds and the signer would have to
wait. With a fixed 3-second take, every clip covers the same stretch of time,
and preprocessing resamples every clip to 32 frames anyway.

How frames are never dropped: a dedicated capture thread reads every camera
frame and passes it to a separate writer thread. The MediaPipe preview runs in
the main thread on a 640-pixel-wide copy (every other frame while recording),
so a slow preview can never cost a recorded frame. The saved video is always
full resolution with every frame.

---

## 6. Landmark format (the browser app must match this exactly)

This is the reference for the separate preprocessing/training project and
the browser app. (The extraction script that implemented it was removed
from this folder.)

### 6.1 Orientation and handedness convention

1. **Saved videos are NOT mirrored.** They are stored exactly as the camera
   sees them, so the signer's real right hand appears on the **left** side of
   the image.
2. **MediaPipe always runs on these unmirrored frames**: live preview, take
   review and extraction alike. Only the on-screen preview is mirrored
   (selfie view), and that is purely for display.
3. **Handedness labels are used exactly as MediaPipe returns them, without
   swapping.** The Tasks-API HandLandmarker labels the hand as it *appears*
   in the image. We verified this on MediaPipe's own sample photo: an
   anatomical left hand is labelled `"Left"`. On an unmirrored frame that is
   the signer's real hand, so the signer's **real right hand = `"Right"`**.

   This differs from the old MediaPipe "Hands" documentation, which said input
   was assumed to be mirrored and labels should be swapped. That advice does
   not apply to this setup. The handedness check (section 9, step 5) confirms
   it on your camera. If it ever shows the opposite, set
   `SWAP_HANDEDNESS_LABELS = True` in `hudyat_common.py` and re-extract the
   landmarks.
4. **Left-handed signers are never flipped.** Their data is stored as recorded,
   and `signing_hand = left` goes in the metadata.

**Browser app:** run `@mediapipe/tasks-vision` HandLandmarker
(`runningMode: "VIDEO"`, `numHands: 2`, same `.task` file) on the **raw**
`<video>` frames. Mirror only with CSS (`transform: scaleX(-1)`) for display.
**Never** draw a flipped frame into a canvas and run detection on that. Use
`handedness[i][0].categoryName` as-is.

### 6.2 Per-frame vector: 128 float values

| Index | Content |
|---|---|
| 0 – 62 | **Left** hand: 21 landmarks × (x, y, z), landmark-major: `x0, y0, z0, x1, y1, z1, …, x20, y20, z20` |
| 63 | Left present flag (`1.0` / `0.0`) |
| 64 – 126 | **Right** hand: same 63-value layout |
| 127 | Right present flag |

- Hands go into slots by **handedness label**, never by detection order.
- A missing hand is 63 zeros with its flag set to 0.
- If both detections in a frame get the same label, the one with the higher
  handedness score takes that slot and the other is dropped (the frame is
  logged).
- Value for landmark *k*, axis *a* (0 = x, 1 = y, 2 = z) of a hand:
  `slot_offset + 3*k + a`, where `slot_offset` is 0 (Left) or 64 (Right).

### 6.3 Per-hand normalization (applied to each hand separately)

Starting from MediaPipe's normalized landmarks (`x` and `y` in 0..1 of the
image width and height, `z` relative depth):

1. **Aspect correction**: `x *= W/H` and `z *= W/H`, where W×H is the frame
   size (×1.778 for 1280×720). MediaPipe's `x` is a fraction of the width and
   `y` a fraction of the height, so without this step the same hand shape
   would give different numbers on a 16:9 and a 4:3 camera. MediaPipe's `z`
   uses roughly the same scale as `x`, so it is scaled too. (This was the
   `ASPECT_CORRECT` setting of the old extraction script.)
2. **Translate**: subtract the wrist (landmark 0) from every landmark, which
   puts the wrist at the origin.
3. **Scale**: divide all 63 values by the 3D distance from the wrist
   (landmark 0) to the middle-finger MCP (landmark 9). If that distance is
   below `1e-6`, skip the division (guard against division by zero).

```js
// Browser reference implementation for one hand (lm = 21 {x,y,z}; W,H = video size)
function normalizeHand(lm, W, H) {
  const a = W / H;
  const p = lm.map(q => [q.x * a, q.y, q.z * a]);
  const [wx, wy, wz] = p[0];
  for (const q of p) { q[0] -= wx; q[1] -= wy; q[2] -= wz; }
  const s = Math.hypot(p[9][0], p[9][1], p[9][2]);
  const out = p.flat();
  return s > 1e-6 ? out.map(v => v / s) : out;
}
```

### 6.4 Sign window (trimming idle frames)

Before resampling, every clip is cut to its **sign window**, the part where
the hands are visible. FSL-105 clips are 4.07 s long, but only about 1.4 s is
signing: the hands are out of view for about 1.6 s before and 1.0 s after.
Without trimming, about 21 of the 32 frames would be empty. Our own takes
follow the same hands-down rule (section 2.2).

1. For every frame, `present = at least one hand detected`. **Hand presence
   only.** "No movement" is never used, because static signs (numbers,
   letters) hold the hand still on purpose.
2. `first` / `last` = the first and last frame with a hand. Gaps in between
   (short detection dropouts) are kept.
3. Margin: `m = round(0.1 × fps)` frames (3 at 30 fps, 6 at 60 fps).
   `start = max(0, first − m)`, `end = min(T − 1, last + m)` (inclusive).
4. Minimum length **8 frames**: if `end − start + 1 < 8`, widen the window to 8
   frames around its centre, shifted to stay inside the clip. A clip shorter
   than 8 frames is kept whole.
5. **No hand in any frame**: keep the **whole clip**.
6. Resample frames `start … end` to 32 (6.5).

Constants: `TRIM_MARGIN_S = 0.1`, `TRIM_MIN_FRAMES = 8` in `hudyat_common.py`
(function `sign_window`). The recorder's take review uses it, and preprocessing
should use the same function so the numbers match.

```js
// Browser reference implementation. present = array of booleans, one per frame.
function signWindow(present, fps) {
  const T = present.length;
  const first = present.indexOf(true), last = present.lastIndexOf(true);
  if (first < 0) return [0, T - 1];                       // no hand: whole window
  const m = Math.round(0.1 * fps);
  let start = Math.max(0, first - m), end = Math.min(T - 1, last + m);
  const need = Math.min(8, T);
  if (end - start + 1 < need) {                           // widen to 8 frames around the centre
    const centre = (start + end) / 2;
    start = Math.max(0, Math.min(Math.round(centre - (need - 1) / 2), T - need));
    end = start + need - 1;
  }
  return [start, end];                                    // inclusive
}
```

Rounding is "half up" in both versions (`Math.round` in JS, `floor(x + 0.5)`
in Python), so 25 fps gives a margin of 3 frames in both.

### 6.5 Resampling to 32 frames

The sign window (from 6.4; any length) becomes exactly **32 frames**,
sampled uniformly over time:

- Output frame `j` (0..31) is at time `t = j * (T - 1) / 31` in the original
  T-frame clip, between frames `i0 = floor(t)` and `i1 = i0 + 1`, with weight
  `w = t - i0`. The nearest frame is `i0` if `w < 0.5`, otherwise `i1`.
- For **each hand slot** separately:
  - **presence flag** = the flag of the **nearest** original frame;
  - if the hand is present in **both** `i0` and `i1`: linear interpolation
    `(1 - w) * v[i0] + w * v[i1]` of the 63 values;
  - otherwise, copy the **nearest** frame's 63 values (and its flag). A
    present hand is never blended with the zeros of an absent hand.

Saved sample: `.npy`, shape **(32, 128)**, dtype **float32**.

### 6.6 Live app rule (browser)

The learner does not press a button, so the app finds the sign window itself:

1. **Start:** the window starts when a hand appears (the first frame with
   `present`). Keep a short ring buffer of the last **0.1 s** of frames, so the
   frames just before the hand appeared are included, matching the 0.1 s
   margin of 6.4.
2. **End:** the window ends when the hands **disappear** (no hand in the frame)
   **or after a maximum of 3 seconds**, whichever comes first. The 3-second
   limit means recognition still works when a learner keeps their hands in
   view. Then a new window can start straight away.
3. Build the 128-value vector for every frame in the window (6.2, 6.3), apply
   `signWindow` (6.4), resample to 32 (6.5) and run the model.

Recommended (not required) for robustness: treat "disappear" as "no hand for
about 0.2 s", so a single frame where detection drops out doesn't cut a
sign in half. The trailing absent frames are removed again by the 6.4 trim
(apart from its 0.1 s margin).

Training clips match this: FSL-105 signs are 1.1–2.0 s long (90 % of
clips), so they fit inside the 3-second limit, and our 3-second takes can
never be longer than it.

---

## 7. Daily workflow summary

1. `.\venv\Scripts\Activate.ps1`
2. `python record_tool.py --list-cameras`: confirm 1280×720 at ~30 fps.
3. `python record_tool.py`, then click the signer → category → sign, and set up the scene with the live view (checklist 2.2). (Camera settings are remembered; add `--camera <n> --camera-name "<model>"` only when the camera changes.)
4. Record. Switch signers from the **Signer** pill without restarting. Press **Q** when done.
5. Check **Progress overview** to see what is left for every signer.
6. After the session, back up the `dataset/` folder and `signers.csv` (external drive).
7. Preprocessing and training: the separate project.

---

## 8. Troubleshooting

| Problem | What to do |
|---|---|
| **Camera not found** / "Could not open camera" | Close Zoom, Teams, the Camera app and browsers using the camera. Unplug and replug the USB webcam (use a port directly on the laptop, not a hub). Check *Settings → Privacy & security → Camera*: "Let desktop apps access your camera" must be **On**. |
| **Wrong camera index** (wrong camera opens, or a black image) | Run `python record_tool.py --list-cameras` and try each listed index (`--camera 1`, …). Built-in is usually 0, USB usually 1, but this can change when you plug or unplug devices, so check every session. A black image is usually the infrared face-login camera. |
| **Low fps** (below 25, red warning) | Recording still works (each take is still 3 seconds, just with fewer frames), but fix it if you can. 1) Try the other backend: `python record_tool.py --list-cameras --backend dshow` (or `msmf`) and use the faster one with `record_tool.py --backend …`. 2) Add more **front light**: in dim light, webcams lengthen exposure and drop to 15 fps or lower. 3) Plug the laptop into power and close heavy programs. 4) Use a USB 3 port for USB webcams. |
| **Camera gives the wrong resolution** (e.g. 640×480) | The camera or backend doesn't support 1280×720. Try the other backend, then another camera. Never record a full session with the red resolution warning showing. |
| **Hands not detected** / low `hand_detect_rate` | More front light (the brightness indicator should say OK). Keep hands inside the frame and not too close to the edges. Use plain long sleeves in a colour different from skin and the cloth. Make sure the signer isn't too far away: they should fill the framing guide. Remove rings and bracelets. |
| **Green tint on hands** | Green light bouncing off the cloth. Move the **signer farther from the cloth** (at least 1 m), light the signer from the front, and stretch the cloth so it's matte and flat. |
| **Video won't play** in the Windows player | `mp4v` files don't play in every default player. Install **VLC** (free) and open the files with it. The files themselves are fine. |
| `.avi` files instead of `.mp4` | The `mp4v` codec wasn't available, so the tool fell back to XVID (logged in the `codec` column). VLC plays both, and OpenCV reads both. |
| "MediaPipe model not found" | Do Setup step 4 (download `models\hand_landmarker.task`). |
| `'python' is not recognized` / wrong Python | Activate the venv first (`venv\Scripts\activate.bat`). The prompt should start with `(venv)`. |
| Keys do nothing | Click on the video window so it has keyboard focus. |
| A button/sign "does nothing" on the recording screen | A take is recording or waiting for Keep / Retake; the message above the caption says so. Press K or R first. |
| Wrong hand saved for a signer | Edit the `hand` column in `signers.csv` (`left` / `right`) while the tool is closed, or start once with `--signer S3 --hand left`. Takes already kept keep the hand they were recorded with in `metadata.csv`. |
| Mouse wheel doesn't scroll a list | Move the mouse over the list first, or use the small ⌃/⌄ buttons. |
| On-screen text looks plain / "font files not found" warning | The `assets/fonts/` folder is missing or incomplete. Copy it from another laptop (it must contain the four `Poppins-*.ttf` files). The tool still works with the fallback font. |
| Preview feels slow | Hand detection takes most of the time on slower laptops. Recording is not affected: the camera thread saves every frame on its own. Close other programs and plug in the charger. |
| Left/Right labels look swapped | Remember the preview is a **mirror**: your right hand appears on the right of the screen and should be labelled "Right". If it is labelled "Left", see section 6.1, point 3. |
| VS Code shows "Module 'cv2' has no … member" | A linter false positive (OpenCV is a compiled module). The scripts run fine. |

---

## 9. Pre-session test checklist (do this once before real data collection)

1. `python record_tool.py --list-cameras`: confirm **1280x720 at ~30 fps** (compare with `--backend dshow` and pick the faster backend).
2. Record 2 test clips with `python record_tool.py`, using signer **S1** and keeping 1 take each:
   - a 1-hand sign (Shapes → Circle);
   - a 2-hand sign: temporarily set `hands` to 2 for one sign in `signs.csv` (e.g. `parents`, before starting the tool) and check the review highlights `both_hands_rate`;

   Click the signs in the list on the right to jump between them.
3. Check the folders and file names (e.g. `dataset/raw/shapes/circle/S1/circle_S1_01.mp4`) and that each video plays in VLC.
4. Open `dataset/metadata.csv`: 2 rows, `width 1280`, `height 720`, `fps_actual ≈ 30`, `n_frames ≈ 90` (= 3 s × fps_actual), `duration_s ≈ 3.0`, `codec mp4v`, `orientation camera_raw_unmirrored`.
5. **Handedness check.** Record a take (e.g. Shapes → Circle as S1) raising **only your real right hand** and confirm the preview labels it "Right" (the preview is a mirror: your right hand appears on the right). Repeat with only your left hand. Delete these test clips afterwards (see below).
6. Keep 1–2 takes of a sign, press **Q**, start the tool again, open the same signer and sign, and confirm **Take X / 15** continues where it stopped (and the "Saved …" file number goes up).
7. Open that sign's **saved takes** (list icon), replay one, click **Flag: needs re-record**, and confirm the sign's count drops by one, the file is still in `dataset/raw/…`, and its row in `metadata.csv` now says `needs_rerecord`. Click **Unflag** to undo.

**After testing**, delete the test data so it doesn't mix with real data:
delete the test files in `dataset/raw/…` and their rows in
`dataset/metadata.csv` (or the whole
`dataset/` folder if nothing real has been recorded yet). Undo any test edits
to `signs.csv`, and clear test hands in `signers.csv` if needed.

---

## 10. The data-collection process

Run everything from the project folder in **PowerShell** with the venv active
(`.\venv\Scripts\Activate.ps1`).

### Step 1: Record (`record_tool.py`)

```powershell
python record_tool.py --list-cameras
python record_tool.py --camera 0 --camera-name "Built-in" --backend msmf
```

Sections 2 and 3. S1–S7, 3-second takes, progress target 15 takes per sign.

### Step 2: Import the public dataset (`import_public.py`)

1. Download FSL-105 and unzip it, e.g. to `D:\FSL-105`. The videos must be in
   one sub-folder per label (named by id or by label) under the folder you pass
   as `--source`.
2. Fill in `public_mapping.csv`, one row per FSL-105 label you want to use.
   Only labels that are the same sign as one of ours belong here. `category`
   and `sign_id` must exist in `signs.csv`.
   ```
   fsl105_id,fsl105_label,category,sign_id
   0,GOOD MORNING,greetings,good_morning
   ```
   If the FSL-105 folder contains its `labels.csv` (`id,label` columns),
   filling in **either** the id or the label is enough.
3. Check the plan, then import:
   ```powershell
   python import_public.py --source "D:\FSL-105\clips" --dry-run
   python import_public.py --source "D:\FSL-105\clips"
   ```

Each video is converted to H.264 `.mp4` (HEVC and the phone's rotation flag are
handled) and saved as `dataset/raw/<category>/<sign_id>/PUB/<sign_id>_PUB_<NN>.mp4`,
with a `metadata.csv` row (`signer = PUB`). Unmapped labels are skipped and
listed. **Running it again never imports a file twice** (the source path is
stored in the row's `camera_name`). Add `--flip` only if the public videos
turn out to be mirrored (check the Left/Right labels when replaying them).
ffmpeg comes from the system if installed, otherwise from the `imageio-ffmpeg`
package.

### Step 3: Expert validation (manual)

The FSL experts review the clips (Saved takes → replay in the recorder, or
VLC). Bad takes: flag them **Needs re-record** in the recorder (section 3.6)
and record new ones. Clips the experts reject outright (e.g. PUB clips): add a
column `expert_status` to `dataset/metadata.csv` (Excel; close the recorder
first) and write `rejected` (or `approved`) in it. Also set `hands = 2` and
`mirror_ok = 1` in `signs.csv` where the experts say so.

### Tests

```powershell
python -m pytest
```

Unit tests for the time-based recording, the sign-window rule (section 6.4)
and the public-dataset import plan.
