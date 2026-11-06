"""Reference videos for the expert validation (hudyat_reference.py, record_tool.py --reference)."""

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

import hudyat_common as hc
import hudyat_reference as ref
import record_tool as rt

FOLDERS = ["01 Alphabet", "02 Numbers", "03 Greetings", "04 Colors", "05 Shapes", "06 Family",
           "07 Calendar", "08 WH-Questions", "09 Word Concepts"]
COUNTS = [28, 10, 10, 6, 8, 10, 22, 6, 10]
ALPHABET = [chr(c) for c in range(65, 91)] + ["Ñ", "NG"]   # A-Z keep 01-26; Ñ, NG are 27, 28


@pytest.fixture(scope="module")
def refs():
    return ref.reference_signs(hc.load_signs())


def sign(category, sign_id, label):
    return hc.Sign(category, sign_id, label, 15, 1)


# --- sign order and numbering ---------------------------------------------------
def test_110_signs_in_table_order_numbered_per_folder(refs):
    assert len(refs) == 110
    assert [r.index for r in refs] == list(range(110))
    folders = list(dict.fromkeys(r.folder for r in refs))
    assert folders == FOLDERS
    assert [sum(r.folder == f for r in refs) for f in FOLDERS] == COUNTS
    for f in FOLDERS:
        assert [r.no for r in refs if r.folder == f] == list(range(1, COUNTS[FOLDERS.index(f)] + 1))
    greetings = [r.file_stem for r in refs if r.folder == "03 Greetings"]
    assert greetings[0] == "01 Good Morning" and greetings[5] == "06 I'm Fine"


def test_alphabet_is_a_to_z_then_enye_and_ng(refs):
    alphabet = [r for r in refs if r.folder == "01 Alphabet"]
    assert [r.sign.sign_label for r in alphabet] == ALPHABET
    assert alphabet[25].file_stem == "26 Z"
    assert alphabet[26].file_stem == "27 Ñ" and alphabet[27].file_stem == "28 NG"


@pytest.mark.parametrize("static", ["alphabet_static", "alphabet_static_test"])
def test_static_and_dynamic_letters_merge_and_none_is_skipped(static):
    others = [s for s in hc.load_signs() if s.category != "alphabet"]
    letters = [sign(static, f"letter_{c.lower()}", c) for c in "ABCDEFGHIKLMN"]
    letters += [sign(static, "letter_enye", "Ñ"), sign(static, "letter_ng", "NG")]
    letters += [sign(static, f"letter_{c.lower()}", c) for c in "OPQRSTUVWXY"]
    dynamic = [sign("alphabet_dynamic", "letter_j", "J"), sign("alphabet_dynamic", "letter_z", "Z")]
    out = ref.reference_signs(dynamic + others + letters + [sign("none", "none", "None")])
    alphabet = [r for r in out if r.folder == "01 Alphabet"]
    assert [r.sign.sign_label for r in alphabet] == ALPHABET
    assert alphabet[9].file_stem == "10 J" and alphabet[25].file_stem == "26 Z"
    assert len(out) == 110


def test_wrong_total_stops():
    with pytest.raises(ValueError, match="expected 110"):
        ref.reference_signs(hc.load_signs()[:-1])


def test_unknown_category_stops():
    with pytest.raises(ValueError, match="unknown categories"):
        ref.reference_signs(hc.load_signs() + [sign("animals", "dog", "Dog")])


# --- file names --------------------------------------------------------------------
def test_clean_filename_keeps_capitals_and_apostrophes():
    assert ref.clean_filename("I'm Fine") == "I'm Fine"
    assert ref.clean_filename("You're Welcome") == "You're Welcome"
    assert ref.clean_filename("Nice to Meet You") == "Nice to Meet You"
    assert ref.clean_filename('A/B:C*D?"E<F>G|H\\I') == "A_B_C_D__E_F_G_H_I"
    assert ref.clean_filename("Etc.") == "Etc"                        # no trailing dot on Windows


# --- re-recording: old files are moved, never deleted --------------------------------
def test_archive_moves_old_files_with_timestamp_and_never_overwrites(refs, tmp_path):
    r = refs[0]
    now = datetime(2026, 10, 26, 9, 30, 15)
    for round_no in (1, 2):
        ref.mp4_path(r, tmp_path).parent.mkdir(parents=True, exist_ok=True)
        ref.mp4_path(r, tmp_path).write_bytes(b"mp4 %d" % round_no)
        ref.gif_path(r, tmp_path).write_bytes(b"gif %d" % round_no)
        moved = ref.archive_old(r, tmp_path, now)
        assert len(moved) == 2
        assert not ref.mp4_path(r, tmp_path).exists() and not ref.gif_path(r, tmp_path).exists()
    old = sorted(p.name for p in (tmp_path / "_old" / "01 Alphabet").iterdir())
    assert old == ["01 A 2026-10-26_093015 (2).gif", "01 A 2026-10-26_093015 (2).mp4",
                   "01 A 2026-10-26_093015.gif", "01 A 2026-10-26_093015.mp4"]
    assert (tmp_path / "_old" / "01 Alphabet" / "01 A 2026-10-26_093015.mp4").read_bytes() == b"mp4 1"
    assert ref.archive_old(r, tmp_path) == []                              # nothing to move


# --- manifest ----------------------------------------------------------------------
def fake_result(r, base, seconds=2.6, gif_bytes=512_000):
    return ref.ExportResult(ref.mp4_path(r, base), ref.gif_path(r, base), seconds, gif_bytes, 480, 12, True, [])


def test_manifest_rows_are_written_in_form_order(refs, tmp_path):
    greet = next(r for r in refs if r.file_stem == "06 I'm Fine")
    rows = {}
    for r in (greet, refs[0]):
        rows[(r.sign.sign_id, 1)] = ref.manifest_row(r, fake_result(r, tmp_path), 0.9, "right",
                                                datetime(2026, 10, 26, 10, 0, 0))
    ref.write_manifest(rows, refs, tmp_path)
    with open(ref.manifest_path(tmp_path), newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames[:13] == ["no", "category", "folder", "sign_id", "sign_label", "file_stem",
                                          "mp4", "gif", "seconds", "gif_kb", "hand_detect_rate",
                                          "hand", "recorded_at"]
        out = list(reader)
    assert reader.fieldnames[-1] == "version"
    assert [r["sign_id"] for r in out] == ["letter_a", "im_fine"]
    assert out[1] == {"no": "06", "category": "greetings", "folder": "03 Greetings", "sign_id": "im_fine",
                      "sign_label": "I'm Fine", "file_stem": "06 I'm Fine",
                      "mp4": "03 Greetings/06 I'm Fine.mp4", "gif": "03 Greetings/06 I'm Fine.gif",
                      "seconds": "2.60", "gif_kb": "500", "hand_detect_rate": "0.900",
                      "hand": "right", "recorded_at": "2026-10-26T10:00:00", "gif_settings": "480px 12fps",
                      "version": "1"}
    assert ref.read_manifest(tmp_path)[("im_fine", 1)]["sign_label"] == "I'm Fine"


def test_manifest_without_version_column_reads_as_version_1(refs, tmp_path):
    ref.manifest_path(tmp_path).parent.mkdir(parents=True)
    ref.manifest_path(tmp_path).write_text("no,sign_id,sign_label\n06,im_fine,I'm Fine\n", encoding="utf-8-sig")
    assert list(ref.read_manifest(tmp_path)) == [("im_fine", 1)]


def test_version_2_files_and_manifest_rows(refs, tmp_path):
    r = next(x for x in refs if x.file_stem == "06 I'm Fine")
    assert ref.mp4_path(r, tmp_path).name == "06 I'm Fine.mp4"
    assert ref.mp4_path(r, tmp_path, 2).name == "06 I'm Fine (2).mp4"
    assert ref.gif_path(r, tmp_path, 2).name == "06 I'm Fine (2).gif"
    rows = {}
    for v in (2, 1):
        res = ref.ExportResult(ref.mp4_path(r, tmp_path, v), ref.gif_path(r, tmp_path, v), 2.0, 1024,
                               480, 12, True, [])
        rows[(r.sign.sign_id, v)] = ref.manifest_row(r, res, 0.9, "right", version=v)
    ref.write_manifest(rows, refs, tmp_path)
    with open(ref.manifest_path(tmp_path), newline="", encoding="utf-8-sig") as f:
        out = list(csv.DictReader(f))
    assert [(x["version"], x["mp4"]) for x in out] == [("1", "03 Greetings/06 I'm Fine.mp4"),
                                                      ("2", "03 Greetings/06 I'm Fine (2).mp4")]
    assert out[1]["file_stem"] == "06 I'm Fine (2)"


def test_trim_range_is_sign_window_plus_0_3_s_inside_the_take():
    assert ref.trim_range(30, 89, 30.0, 120) == pytest.approx((0.7, 3.3))
    assert ref.trim_range(2, 119, 30.0, 120) == pytest.approx((0.0, 4.0))


# --- synthetic video (no webcam) ------------------------------------------------------
def write_synthetic_take(path_stem: Path, seconds: float = 4.0, fps: float = 30.0, noise: bool = False) -> Path:
    """'Record' a take through the real RecordingJob + video writer, without a camera."""
    writer, path, codec = hc.open_video_writer(path_stem, fps, (1280, 720))
    job = rt.RecordingJob(writer, path, codec, seconds)
    rng = np.random.default_rng(0)
    for i in range(int(seconds * fps) + 5):
        frame = np.full((720, 1280, 3), 60, np.uint8)
        if noise:
            frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        x = 200 + i * 6                                   # something that moves, left to right
        cv2.rectangle(frame, (x, 300), (x + 120, 420), (40, 200, 240), -1)
        if job.push(frame, 1000.0 + i / fps):
            break
    assert job.done.wait(5.0) and not job.failed
    return path


def video_codec_info(path: Path) -> str:
    out = subprocess.run([ref.ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return out.stderr


def test_gif_size_limit_steps_down_until_it_fits(tmp_path):
    raw = write_synthetic_take(tmp_path / "noisy.tmp", seconds=1.0, noise=True)
    gif = tmp_path / "x.gif"
    width, fps, first_size, ok = ref.make_gif(raw, gif)
    assert (width, fps) == (480, 12)                     # the standard setting when it fits
    width, fps, size, ok = ref.make_gif(raw, gif, max_bytes=first_size)   # 480/12 is now "too big"
    assert ok and size < first_size and (width, fps) != (480, 12)
    assert (width, fps) in ref.GIF_SETTINGS
    width, fps, size, ok = ref.make_gif(raw, gif, max_bytes=1000)         # impossible: keep the smallest
    assert not ok and (width, fps) == ref.GIF_SETTINGS[-1] and gif.exists()


def test_record_keep_files_manifest_and_form_files_without_webcam(refs, tmp_path):
    r = next(x for x in refs if x.file_stem == "06 I'm Fine")
    raw = write_synthetic_take(ref.tmp_dir(tmp_path) / ("im_fine" + hc.TMP_TAG))
    result = ref.keep_take(raw, r, refs, window_start=30, window_end=89, fps=30.0, n_frames=120,
                           hand_detect_rate=0.75, hand="left", base=tmp_path)

    mp4 = tmp_path / "validation" / "03 Greetings" / "06 I'm Fine.mp4"
    gif = tmp_path / "validation" / "03 Greetings" / "06 I'm Fine.gif"
    assert result.mp4 == mp4 and mp4.exists() and gif.exists() and not raw.exists()
    info = video_codec_info(mp4)
    assert "h264" in info and "yuv420p" in info and "1280x720" in info and "Audio" not in info
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert abs(n - 78) <= 2                               # 0.7 .. 3.3 s at 30 fps
    assert result.seconds == pytest.approx(2.6)
    assert gif.stat().st_size < ref.GIF_MAX_BYTES and result.gif_settings == "480px 12fps"
    assert "GIF89a" == gif.read_bytes()[:6].decode() and b"NETSCAPE2.0" in gif.read_bytes()   # loops

    # orientation: the video is NOT mirrored (the square starts on the left, as recorded)
    cap = cv2.VideoCapture(str(mp4))
    ok, first = cap.read()
    cap.release()
    cols = np.where((first[:, :, 2] > 150).any(axis=0))[0]
    assert ok and cols.mean() < 640

    rows = ref.read_manifest(tmp_path)
    assert list(rows) == [("im_fine", 1)]
    row = rows[("im_fine", 1)]
    assert row["seconds"] == "2.60"
    assert row["hand_detect_rate"] == "0.750" and row["hand"] == "left"
    assert ref.done_ids(refs, tmp_path) == {"im_fine"}

    folder = tmp_path / "validation"
    assert {"manifest.csv", "form_list.md", "build_validation_form.gs", "FORM_SETUP.md"} <= \
        {p.name for p in folder.iterdir()}
    assert "Reference videos recorded: 1 / 110" in (folder / "form_list.md").read_text(encoding="utf-8")
    gs = gs_sections(folder / "build_validation_form.gs")
    assert [s["folder"] for s in gs] == ["03 Greetings"]              # only signs with a video
    assert gs[0]["signs"] == [{"no": "06", "label": "I'm Fine", "sign_id": "im_fine", "mp4": "06 I'm Fine.mp4",
                               "gif": "06 I'm Fine.gif"}]

    # Re-record: the first version is moved to _old, never deleted.
    raw2 = write_synthetic_take(ref.tmp_dir(tmp_path) / ("im_fine" + hc.TMP_TAG))
    first_bytes = mp4.read_bytes()
    result2 = ref.keep_take(raw2, r, refs, window_start=0, window_end=119, fps=30.0, n_frames=120,
                            hand_detect_rate=1.0, hand="left", base=tmp_path)
    assert len(result2.archived) == 2 and all(p.exists() for p in result2.archived)
    assert any(p.read_bytes() == first_bytes for p in result2.archived)
    assert mp4.exists() and ref.read_manifest(tmp_path)[("im_fine", 1)]["seconds"] == "4.00"
    assert "ARCHIVED _old/03 Greetings/06 I'm Fine" in (tmp_path / "reference_log.txt").read_text(encoding="utf-8")

    # Version 2: a second video next to the first; version 1 is not touched.
    v1_bytes = mp4.read_bytes()
    raw3 = write_synthetic_take(ref.tmp_dir(tmp_path) / ("im_fine" + hc.TMP_TAG), seconds=1.0)
    result3 = ref.keep_take(raw3, r, refs, window_start=0, window_end=29, fps=30.0, n_frames=30,
                            hand_detect_rate=0.5, hand="left", base=tmp_path, version=2)
    mp4_2 = tmp_path / "validation" / "03 Greetings" / "06 I'm Fine (2).mp4"
    assert result3.mp4 == mp4_2 and mp4_2.exists() and mp4_2.with_suffix(".gif").exists()
    assert result3.archived == [] and mp4.read_bytes() == v1_bytes
    assert set(ref.read_manifest(tmp_path)) == {("im_fine", 1), ("im_fine", 2)}
    assert ref.done_ids(refs, tmp_path) == {"im_fine"} and ref.version2_ids(refs, tmp_path) == {"im_fine"}
    s = gs_sections(folder / "build_validation_form.gs")[0]["signs"][0]
    assert s["mp4_2"] == "06 I'm Fine (2).mp4" and s["gif_2"] == "06 I'm Fine (2).gif"
    assert "06 I'm Fine (2 versions)" in (folder / "form_list.md").read_text(encoding="utf-8")

    # Recording version 2 again moves only the old version 2 to _old.
    raw4 = write_synthetic_take(ref.tmp_dir(tmp_path) / ("im_fine" + hc.TMP_TAG), seconds=1.0)
    result4 = ref.keep_take(raw4, r, refs, window_start=0, window_end=29, fps=30.0, n_frames=30,
                            hand_detect_rate=0.5, hand="left", base=tmp_path, version=2)
    assert sorted(p.name.split(" 20")[0] for p in result4.archived) == ["06 I'm Fine (2)", "06 I'm Fine (2)"]
    assert mp4.read_bytes() == v1_bytes


# --- form material -------------------------------------------------------------------
def gs_sections(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^var SECTIONS = (\[.*?^\]);", text, re.S | re.M)
    assert m, "SECTIONS not found in the .gs"
    return json.loads(m.group(1))


def fake_recorded(refs, base, skip_folder):
    """Manifest rows + (empty) .mp4 files for every sign except one folder."""
    rows = {}
    for r in refs:
        if r.folder == skip_folder:
            continue
        mp4 = ref.mp4_path(r, base)
        mp4.parent.mkdir(parents=True, exist_ok=True)
        mp4.write_bytes(b"")
        rows[(r.sign.sign_id, 1)] = ref.manifest_row(r, fake_result(r, base), 0.9, "right")
    ref.write_manifest(rows, refs, base)


def gs_request_signs(path: Path) -> list[str]:
    m = re.search(r"^var REQUEST_SIGNS = (\[.*?\]);", path.read_text(encoding="utf-8"), re.S | re.M)
    assert m, "REQUEST_SIGNS not found in the .gs"
    return json.loads(m.group(1))


def test_generated_gs_has_recorded_signs_and_asks_for_help_with_the_rest(refs, tmp_path):
    fake_recorded(refs, tmp_path, "09 Word Concepts")
    ref.write_form_files(refs, tmp_path)
    path = tmp_path / "validation" / "build_validation_form.gs"
    text = path.read_text(encoding="utf-8")
    assert "var FOLDER_ID = 'PASTE_DRIVE_FOLDER_ID_HERE';" in text
    assert "var UPLOAD_LINK = 'PASTE_UPLOAD_FOLDER_LINK_HERE';" in text
    assert "FormApp.create(FORM_TITLE)" in text and "HUDYAT – FSL Gesture Validation Form" in text
    assert "['Correct', 'Other version', 'Needs fix', 'Wrong']" in text
    assert "If not Correct, what should change?" in text and "Play video" not in text
    assert "Can it also be signed with the other hand (left or right)?" in text
    assert "['Yes, either hand is fine', 'No, only with this hand', 'Not sure']" in text
    assert "var title = s.label;" in text                            # no "01 " before the name
    assert "Tacloban Association of the Deaf (TAD)" in text and "Leyte Association of the Deaf (LAD)" in text
    assert "setSharing" not in text and "[Contact: name, Messenger/phone]" in text
    assert "getEditUrl()" in text and "getPublishedUrl()" in text
    assert "Which version is correct?" in text and "['Version 1', 'Version 2', 'Neither is correct']" in text
    assert "addCheckboxItem()" in text
    assert "There are 100 signs in 8 groups" in text
    # Signs with a video are validated, in form order.
    sections = gs_sections(path)
    assert [s["folder"] for s in sections] == FOLDERS[:-1]
    assert [s["title"] for s in sections] == [f.split(" ", 1)[1] for f in FOLDERS[:-1]]
    assert [len(s["signs"]) for s in sections] == COUNTS[:-1]
    flat = [(s["folder"], x["no"], x["label"]) for s in sections for x in s["signs"]]
    assert flat == [(r.folder, f"{r.no:02d}", r.sign.sign_label) for r in refs if r.folder != "09 Word Concepts"]
    # Signs without a video: the validators are asked to send a video.
    word = [r.sign.sign_label for r in refs if r.folder == "09 Word Concepts"]
    assert gs_request_signs(path) == word and len(word) == 10
    assert "addGridItem()" in text and "['I know it', 'Not sure', \"I don't know\"]" in text
    assert "'Yes, on Messenger'" in text and "'Yes, I will upload it to the Google Drive folder'" in text

    md = (tmp_path / "validation" / "form_list.md").read_text(encoding="utf-8")
    assert "## 03 Greetings\n\n01 Good Morning\n02 Good Afternoon" in md
    assert "06 I'm Fine" in md and md.count("\n## ") == 10           # 9 categories + "Not in the form yet"
    assert "## Not in the form yet: validators are asked to send a video\n\nYes\nNo\nPlease" in md
    setup = (tmp_path / "validation" / "FORM_SETUP.md").read_text(encoding="utf-8")
    for step in ("Upload the folder", "folder ID", "script.google.com", "FOLDER_ID", "UPLOAD_LINK",
                 "buildForm", "link in the log"):
        assert step in setup


def test_no_help_section_when_every_sign_has_a_video(refs, tmp_path):
    fake_recorded(refs, tmp_path, None)
    ref.write_form_files(refs, tmp_path)
    path = tmp_path / "validation" / "build_validation_form.gs"
    assert gs_request_signs(path) == [] and len(gs_sections(path)) == 9
    assert "Not in the form yet" not in (tmp_path / "validation" / "form_list.md").read_text(encoding="utf-8")


# --- deleting a saved video --------------------------------------------------------
def saved_video(r, refs, base, version=1):
    """A saved video on disk (.mp4 + .gif) with its manifest row."""
    mp4, gif = ref.mp4_path(r, base, version), ref.gif_path(r, base, version)
    mp4.parent.mkdir(parents=True, exist_ok=True)
    mp4.write_bytes(b"mp4")
    gif.write_bytes(b"gif")
    rows = ref.read_manifest(base)
    res = ref.ExportResult(mp4, gif, 2.0, 1024, 480, 12, True, [])
    rows[(r.sign.sign_id, version)] = ref.manifest_row(r, res, 0.9, "right", version=version)
    ref.write_manifest(rows, refs, base)


def test_delete_video_erases_it_and_sends_the_sign_back_to_pending(refs, tmp_path):
    r = next(x for x in refs if x.file_stem == "06 I'm Fine")
    saved_video(r, refs, tmp_path)
    ref.write_form_files(refs, tmp_path)
    assert "I'm Fine" not in gs_request_signs(tmp_path / "validation" / "build_validation_form.gs")

    ref.delete_video(r, refs, tmp_path)

    folder = tmp_path / "validation" / "03 Greetings"
    assert not (folder / "06 I'm Fine.mp4").exists() and not (folder / "06 I'm Fine.gif").exists()
    assert ("im_fine", 1) not in ref.read_manifest(tmp_path)
    assert "im_fine" not in ref.done_ids(refs, tmp_path)
    assert not (tmp_path / "_old").exists()                        # permanent: nothing is archived
    assert "I'm Fine" in gs_request_signs(tmp_path / "validation" / "build_validation_form.gs")
    assert "DELETED 03 Greetings/06 I'm Fine" in (tmp_path / "reference_log.txt").read_text(encoding="utf-8")


def test_delete_version_2_leaves_version_1_alone(refs, tmp_path):
    r = next(x for x in refs if x.file_stem == "06 I'm Fine")
    saved_video(r, refs, tmp_path, 1)
    saved_video(r, refs, tmp_path, 2)

    ref.delete_video(r, refs, tmp_path, version=2)

    folder = tmp_path / "validation" / "03 Greetings"
    assert (folder / "06 I'm Fine.mp4").exists() and (folder / "06 I'm Fine.gif").exists()
    assert not (folder / "06 I'm Fine (2).mp4").exists() and not (folder / "06 I'm Fine (2).gif").exists()
    assert list(ref.read_manifest(tmp_path)) == [("im_fine", 1)]


def test_delete_that_windows_refuses_keeps_the_manifest_row(refs, tmp_path, monkeypatch):
    r = next(x for x in refs if x.file_stem == "06 I'm Fine")
    saved_video(r, refs, tmp_path)

    def locked(self, missing_ok=False):
        raise PermissionError("file is open in another program")
    monkeypatch.setattr(Path, "unlink", locked)

    with pytest.raises(OSError):
        ref.delete_video(r, refs, tmp_path)
    assert ("im_fine", 1) in ref.read_manifest(tmp_path)
    assert "im_fine" in ref.done_ids(refs, tmp_path)
