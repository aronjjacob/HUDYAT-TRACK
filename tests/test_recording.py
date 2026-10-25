"""Recording is time-based: a take is exactly CLIP_SECONDS long whatever the camera fps."""

from pathlib import Path

import numpy as np

import record_tool as rt


class FakeWriter:
    def __init__(self) -> None:
        self.frames = 0
        self.released = False

    def write(self, frame) -> None:
        self.frames += 1

    def release(self) -> None:
        self.released = True


def record(fps: float, seconds: float = 10.0) -> tuple[rt.RecordingJob, FakeWriter]:
    """Feed a job frames at a steady fps until it says it is done."""
    writer = FakeWriter()
    job = rt.RecordingJob(writer, Path("x.tmp.mp4"), "mp4v", duration_s=3.0)
    frame = np.zeros((2, 2, 3), np.uint8)
    t0 = 1000.0
    for i in range(int(fps * seconds)):
        if job.push(frame, t0 + i / fps):
            break
    assert job.done.wait(2.0)
    return job, writer


def test_slow_camera_still_records_three_seconds():
    job, writer = record(fps=15.0)
    assert writer.frames == 45                 # 0.000 .. 2.933 s at 15 fps
    assert abs(job.duration_s() - 3.0) < 0.1


def test_thirty_fps_camera_gives_ninety_frames():
    job, writer = record(fps=30.0)
    assert writer.frames == 90


def test_measured_fps_matches_camera():
    job, _ = record(fps=17.0)
    assert abs(job.measured_fps() - 17.0) < 0.01
