"""Reference mode inside the recorder App, driven without a webcam or a window.

The camera thread is not started: frames are pushed into the RecordingJob
directly, and render() is called to build every screen.
"""

import os

import numpy as np
import pytest

import hudyat_common as hc
import hudyat_reference as ref
import record_tool as rt

pytestmark = pytest.mark.skipif(not hc.MODEL_PATH.exists(), reason="MediaPipe model not downloaded")


class FakeCap:
    def read(self):
        return True, np.zeros((720, 1280, 3), np.uint8)

    def release(self):
        pass


def dataset_snapshot():
    """Files directly in dataset/ and dataset/session_logs (with sizes)."""
    out = {}
    for folder in (hc.DATASET_DIR, hc.SESSION_LOG_DIR):
        if folder.exists():
            for p in folder.iterdir():
                if p.is_file():
                    out[p] = p.stat().st_size
    return out


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(ref, "REF_DIR", tmp_path / "references")
    a = rt.App(hc.load_signs(), FakeCap(), (1280, 720), {"camera": 0, "camera_name": "test", "backend": "msmf"})
    monkeypatch.setattr(a, "render_busy", lambda text: None)       # no window in tests
    yield a
    a.landmarker.close()


def record_take(app, key=" "):
    """SPACE (or V: version 2) -> countdown -> record (synthetic frames) -> review."""
    app.do("key:" + key)
    assert app.state == "COUNTDOWN"
    app.start_recording()
    job = app.job
    assert job.path.parent == ref.tmp_dir() and job.duration == app.ref_seconds
    frame = np.full((720, 1280, 3), 80, np.uint8)
    for i in range(200):
        if job.push(frame, 1000.0 + i / 30):
            break
    assert job.done.wait(5.0)
    app.finish_recording()
    assert app.state == "REVIEW"
    assert any("hand" in w.lower() for w in app.review.warnings)   # no hands -> warning, keeping still allowed


def test_reference_mode_record_keep_next_previous(app):
    before = dataset_snapshot()
    app.render(None)
    assert any(a[4] == "reference" for a in app.hits._areas)      # dashboard button

    app.do("reference")
    assert app.modal == "ref_setup"
    app.render(None)
    assert {a[4] for a in app.hits._areas} == {"refhand:left", "refhand:right", "refcancel"}   # no name to type
    app.do("refhand:right")                                        # one click starts
    assert (app.ref_mode, app.ref_hand, app.screen) == (True, "right", "REF_LIST")
    assert ref.load_state()["hand"] == "right"
    assert (ref.validation_dir() / "build_validation_form.gs").exists()
    app.render(None)                                               # category tiles, like the dashboard
    assert sorted(a[4] for a in app.hits._areas if a[4].startswith("refcat:")) == [f"refcat:{i}" for i in range(9)]

    app.do("refcat:2")                                             # 03 Greetings
    assert (app.screen, app.ref_folder) == ("REF_SIGNS", "03 Greetings")
    app.render(None)
    greet_ids = [r.index for r in app.refs if r.folder == "03 Greetings"]
    assert sorted(a[4] for a in app.hits._areas if a[4].startswith("ref:")) == sorted(f"ref:{i}" for i in greet_ids)

    greet = next(r for r in app.refs if r.file_stem == "01 Good Morning")
    app.do(f"ref:{greet.index}")
    assert app.screen == "RECORD" and app.sign.sign_id == "good_morning"
    frame = np.zeros((720, 1280, 3), np.uint8)
    app.render(frame)
    chips = [a[4] for a in app.hits._areas if a[4].startswith("chip:")]
    assert sorted(chips) == sorted(f"chip:{i}" for i in greet_ids)   # only this category on the right
    record_take(app)
    app.render(frame)

    app.do("key:k")
    mp4 = ref.validation_dir() / "03 Greetings" / "01 Good Morning.mp4"
    assert mp4.exists() and mp4.with_suffix(".gif").exists()
    row = ref.read_manifest()[("good_morning", 1)]
    assert (row["hand"], row["seconds"]) == ("right", "4.00") and "demonstrator" not in row
    assert app.sign.sign_id == "good_afternoon" and app.state == "LIVE"   # moved on to the next sign
    assert "good_morning" in app.ref_done
    app.render(frame)

    # B goes back; recording a done sign again moves the old files to _old.
    app.do("key:b")
    assert app.sign.sign_id == "good_morning"
    record_take(app)
    app.do("key:k")
    old = list((ref.old_dir() / "03 Greetings").iterdir())
    assert len(old) == 2 and mp4.exists()

    # V on a done sign records version 2 next to it; version 1 is not touched.
    v1 = mp4.read_bytes()
    app.do("key:b")
    assert app.sign.sign_id == "good_morning"
    record_take(app, "v")
    app.render(frame)
    app.do("key:k")
    mp4_2 = ref.validation_dir() / "03 Greetings" / "01 Good Morning (2).mp4"
    assert mp4_2.exists() and mp4_2.with_suffix(".gif").exists() and mp4.read_bytes() == v1
    assert ("good_morning", 2) in ref.read_manifest() and "good_morning" in app.ref_v2
    assert app.sign.sign_id == "good_afternoon"
    # V on a sign without a video does nothing (record version 1 with SPACE first).
    app.do("key:v")
    assert app.state == "LIVE"
    app.render(frame)

    # N during review discards the take and moves on; R redoes.
    app.do("key:b")
    record_take(app)
    app.do("key:r")
    assert app.state == "LIVE" and app.job is None
    record_take(app)
    app.do("key:n")
    assert app.state == "LIVE" and app.sign.sign_id == "good_afternoon"
    assert not any(ref.tmp_dir().glob("*.mp4"))                    # discarded raw takes are gone

    # N past the last sign of a category moves into the next category.
    app.do(f"ref:{greet_ids[-1]}")
    app.do("key:n")
    assert app.ref_folder == "04 Colors" and app.sign.sign_id == "blue"
    app.do("key:b")
    assert app.ref_folder == "03 Greetings"

    # Esc -> category's signs -> categories -> dashboard, reference mode off.
    app.do("back")
    assert (app.screen, app.ref_folder) == ("REF_SIGNS", "03 Greetings")
    app.render(None)
    assert ref.load_state()["last_sign_id"] == "see_you_tomorrow"
    app.do("back")
    assert app.screen == "REF_LIST"
    app.render(None)
    app.do("back")
    assert app.screen == "HOME" and not app.ref_mode
    assert dataset_snapshot() == before                           # dataset/ untouched
