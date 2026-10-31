"""Sign-window trimming (README section 6.4) and the hands-down recording rule.

The browser app must apply the same window rule, so these numbers are pinned.
"""

import numpy as np
import pytest

import hudyat_common as hc
import record_tool as rt


def present(total, first, last):
    p = np.zeros(total, bool)
    p[first:last + 1] = True
    return p


# --- hudyat_common.sign_window ------------------------------------------------
def test_window_is_first_to_last_hand_plus_0_1_s():
    # 60 fps -> margin 6 frames
    assert hc.sign_window(present(244, 100, 180), 60) == (94, 186, True)
    # 30 fps -> margin 3 frames
    assert hc.sign_window(present(90, 20, 60), 30) == (17, 63, True)


def test_margin_rounds_half_up_like_javascript():
    assert hc.sign_window(present(100, 50, 60), 25) == (47, 63, True)   # 0.1 * 25 = 2.5 -> 3


def test_gaps_inside_the_sign_are_kept():
    p = present(90, 20, 60)
    p[35:45] = False                                    # detection dropout mid-sign
    assert hc.sign_window(p, 30) == (17, 63, True)


def test_no_hand_keeps_whole_clip():
    assert hc.sign_window(np.zeros(90, bool), 30) == (0, 89, False)


def test_margin_is_clipped_to_the_clip():
    assert hc.sign_window(present(90, 1, 88), 30) == (0, 89, False)
    assert hc.sign_window(present(90, 0, 50), 30) == (0, 53, True)


def test_minimum_eight_frames_around_the_centre():
    start, end, trimmed = hc.sign_window(present(244, 120, 120), 10)   # margin 1 -> 3 frames
    assert end - start + 1 == hc.TRIM_MIN_FRAMES == 8
    assert start <= 119 and end >= 121 and trimmed
    # near the edge the window shifts inside the clip
    assert hc.sign_window(present(90, 0, 0), 10) == (0, 7, True)
    # a clip shorter than 8 frames is kept whole
    assert hc.sign_window(present(5, 2, 2), 30) == (0, 4, False)


def test_movement_is_not_used_only_presence():
    # a static sign: hand held still the whole time it is visible -> same window
    assert hc.sign_window(present(90, 30, 70), 30) == (27, 73, True)


def test_hands_at_edges_uses_0_3_s():
    assert hc.hands_at_edges(present(90, 9, 80), 30) == (False, False)    # first 9 / last 9 frames empty
    assert hc.hands_at_edges(present(90, 8, 80), 30) == (True, False)
    assert hc.hands_at_edges(present(90, 20, 81), 30) == (False, True)


# --- recorder: quality inside the window + hands-down warning ----------------
SHAPES = hc.Sign("shapes", "circle", "Circle", 15, 1)


def counts(total, first, last, n=1):
    c = np.zeros(total, int)
    c[first:last + 1] = n
    return list(c)


def test_take_quality_measures_inside_window_and_is_quiet_for_a_good_take():
    q = rt.take_quality(counts(90, 20, 60), 30.0, SHAPES, 90, 30.0)
    assert (q.window_start, q.window_end) == (17, 63)
    assert q.hand_detect_rate == pytest.approx(41 / 47)
    assert q.warnings == []


def test_hand_visible_at_start_or_end_warns_hands_down():
    q = rt.take_quality(counts(90, 2, 60), 30.0, SHAPES, 90, 30.0)
    assert any("Start and end with hands down" in w and "start" in w for w in q.warnings)
    q = rt.take_quality(counts(90, 20, 89), 30.0, SHAPES, 90, 30.0)
    assert any("Start and end with hands down" in w and "end" in w for w in q.warnings)



# --- face fully visible (checked on the hands-down frames only) ---------------
def face_flags(hand_counts, visible):
    """Face result per frame: None where a hand is in the picture (not checked)."""
    return [None if c else visible for c in hand_counts]


def test_face_visible_on_hands_down_frames_is_quiet():
    c = counts(90, 20, 60)
    q = rt.take_quality(c, 30.0, SHAPES, 90, 30.0, face_visible=face_flags(c, True))
    assert q.face_visible_rate == 1.0 and q.warnings == []


def test_face_cut_off_warns():
    c = counts(90, 20, 60)
    q = rt.take_quality(c, 30.0, SHAPES, 90, 30.0, face_visible=face_flags(c, False))
    assert q.face_visible_rate == 0.0
    assert any("Face not fully visible" in w for w in q.warnings)


def test_face_rate_ignores_frames_where_a_hand_covers_the_face():
    c = counts(90, 20, 60)
    flags = face_flags(c, True)
    for i in range(0, 10):                                 # 10 of the 49 hands-down frames cut off
        flags[i] = False
    q = rt.take_quality(c, 30.0, SHAPES, 90, 30.0, face_visible=flags)
    assert q.face_visible_rate == pytest.approx(39 / 49)
    assert any("Face not fully visible (80%)" in w for w in q.warnings)


def test_no_face_check_when_model_is_missing_or_no_hands_down_frames():
    c = counts(90, 20, 60)
    assert rt.take_quality(c, 30.0, SHAPES, 90, 30.0).face_visible_rate is None
    q = rt.take_quality(c, 30.0, SHAPES, 90, 30.0, face_visible=[None] * 90)
    assert q.face_visible_rate is None and not any("Face" in w for w in q.warnings)


def test_face_inside_frame():
    inside = np.array([[0.4, 0.2, 0.0], [0.6, 0.5, 0.0]], np.float32)
    assert hc.face_inside_frame(inside)
    assert not hc.face_inside_frame(inside - [0.0, 0.25, 0.0])   # forehead above the top edge
    assert not hc.face_inside_frame(inside + [0.5, 0.0, 0.0])    # off the right edge
    assert not hc.face_inside_frame(np.zeros((0, 3), np.float32))  # no face found


@pytest.mark.skipif(not hc.FACE_MODEL_PATH.exists(), reason="MediaPipe face model not downloaded")
def test_face_landmarker_finds_no_face_in_a_blank_frame():
    lm = hc.create_face_landmarker()
    try:
        assert hc.face_fully_visible(lm, np.full((720, 1280, 3), 90, np.uint8), 0) is False
    finally:
        lm.close()
