"""Behaviour + liveness + environment monitoring.

* Face Landmarker (MediaPipe): iris-based gaze, head pose (from the facial transformation matrix),
  blink events (blendshapes) -> second-screen / reading score over a rolling 10 s buffer.
* rPPG (POS/CHROM) on forehead+cheek skin -> liveness score.
* YOLO26 (Ultralytics, COCO) -> second person / phone / extra monitor / book in frame.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar

import cv2
import numpy as np

from trusthire.behavior.rppg import chrom, pos, pulse_snr
from trusthire.common.registry import ensure, models_dir
from trusthire.common.types import Modality, ModuleOutput, Window, missing

# MediaPipe 478-landmark indices
L_EYE = (
    33,
    133,
    159,
    145,
)  # outer corner, inner corner, top lid, bottom lid (subject's right eye in image-left)
R_EYE = (362, 263, 386, 374)
L_IRIS, R_IRIS = 468, 473
FOREHEAD = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]
L_CHEEK = [117, 118, 101, 36, 205, 187, 123]
R_CHEEK = [346, 347, 330, 266, 425, 411, 352]


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class FrameFeat:
    t: float
    gaze_x: float  # -1 (looking to image-left) .. +1
    gaze_y: float
    yaw: float
    pitch: float
    roll: float
    blink: float  # mean eyeBlink blendshape


def euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    r = m[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def _eye_ratio(lm: np.ndarray, eye: tuple[int, int, int, int], iris: int) -> tuple[float, float]:
    outer, inner, top, bottom = (lm[i] for i in eye)
    c = lm[iris]
    axis = inner - outer
    width = float(np.linalg.norm(axis)) + 1e-6
    gx = float(np.dot(c - outer, axis) / width**2) * 2 - 1  # -1..1 along the eye
    height = float(np.linalg.norm(bottom - top)) + 1e-6
    gy = float((c[1] - (top[1] + bottom[1]) / 2) / height) * 2
    return gx, gy


def reading_sweeps(xs: np.ndarray, ts: np.ndarray) -> float:
    """Return-sweeps per second: runs of >=3 small same-direction gaze steps followed by a fast reverse jump
    (the left-to-right then carriage-return pattern of reading lines of text)."""
    if len(xs) < 8:
        return 0.0
    d = np.diff(xs)
    sweeps, run, direction = 0, 0, 0
    for v in d:
        s = int(np.sign(v))
        if 0.005 < abs(v) < 0.12 and (direction == 0 or s == direction):
            run += 1
            direction = s
        elif abs(v) >= 0.15 and direction != 0 and s == -direction and run >= 3:
            sweeps += 1
            run, direction = 0, 0
        else:
            run, direction = (0, 0) if abs(v) >= 0.12 else (run, direction)
    dur = float(ts[-1] - ts[0]) or 1.0
    return sweeps / dur


class ObjectDetector:
    """YOLO26 (COCO). Uses an ONNX export via Ultralytics when available (faster on CPU)."""

    CLASSES: ClassVar[dict[int, str]] = {0: "person", 67: "cell phone", 62: "tv", 63: "laptop", 73: "book"}

    def __init__(self, weights: str = "yolo26n.pt", imgsz: int = 480):
        from ultralytics import YOLO

        pt = ensure("behavior.yolo26n") if weights == "yolo26n.pt" else models_dir() / "behavior" / weights
        onnx = pt.with_name(f"{pt.stem}_{imgsz}.onnx")
        if not onnx.exists():
            try:
                exported = YOLO(str(pt)).export(
                    format="onnx", imgsz=imgsz, dynamic=False, simplify=False, verbose=False
                )
                from pathlib import Path

                Path(exported).replace(onnx)
            except Exception:
                onnx = pt
        self.model = YOLO(str(onnx), task="detect")
        self.imgsz = imgsz
        self._lock = threading.Lock()

    def detect(self, img: np.ndarray, conf: float = 0.35) -> list[dict]:
        with self._lock:
            results: Any = self.model.predict(
                img, imgsz=self.imgsz, conf=conf, classes=list(self.CLASSES), verbose=False
            )
            r = results[0]
        out = []
        for b in r.boxes:
            c = int(b.cls)
            out.append(
                {
                    "cls": self.CLASSES.get(c, str(c)),
                    "conf": round(float(b.conf), 3),
                    "box": [round(float(v)) for v in b.xyxy[0].tolist()],
                }
            )
        return out


class BehaviorMonitor:
    """Stateful per-session monitor (keeps rolling buffers for gaze patterns, blinks and rPPG)."""

    _objdet: ObjectDetector | None = None
    _objdet_lock = threading.Lock()

    def __init__(
        self, landmark_stride: int = 2, buffer_s: float = 10.0, fps: float = 25.0, use_objects: bool = True
    ):
        import mediapipe as mp

        task = ensure("behavior.face_landmarker")
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.mp = mp
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self.stride = landmark_stride
        self.feats: deque[FrameFeat] = deque()
        self.rgb: deque[tuple[float, np.ndarray]] = deque()
        self.buffer_s = buffer_s
        self.fps = fps
        self.use_objects = use_objects
        self._lock = threading.Lock()
        self.blink_events: deque[float] = deque()
        self._eye_closed = False
        self._last_t = -1.0

    @classmethod
    def objects(cls) -> ObjectDetector:
        with cls._objdet_lock:
            if cls._objdet is None:
                from trusthire.common.settings import get_settings

                cls._objdet = ObjectDetector(get_settings().yolo_model)
            return cls._objdet

    # ------------------------------------------------------------------ per-frame
    def _landmarks(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray | None, dict[str, float], np.ndarray | None, int]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None, {}, None, 0
        h, w = frame.shape[:2]
        lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        mat = np.array(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else None
        return lm, bs, mat, len(res.face_landmarks)

    @staticmethod
    def _skin_rgb(frame: np.ndarray, lm: np.ndarray) -> np.ndarray | None:
        mask = np.zeros(frame.shape[:2], np.uint8)
        for poly in (FOREHEAD, L_CHEEK, R_CHEEK):
            pts = lm[poly].astype(np.int32)
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
        if mask.sum() < 255 * 200:
            return None
        b, g, r, _ = cv2.mean(frame, mask=mask)
        return np.array([r, g, b], np.float64)

    def process(
        self, w: Window, run_objects: bool = True
    ) -> tuple[ModuleOutput, ModuleOutput, ModuleOutput | None]:
        t0 = time.perf_counter()
        n_faces_max = 0
        last_lm = None
        with self._lock:
            for i, (t, fr) in enumerate(zip(w.frame_times, w.frames, strict=True)):
                # windows overlap by 50 %: each frame must enter the rolling buffers only once
                if t <= self._last_t:
                    continue
                self._last_t = t
                if i % self.stride == 0:
                    lm, bs, mat, nf = self._landmarks(fr)
                    n_faces_max = max(n_faces_max, nf)
                    last_lm = lm
                    if lm is not None:
                        gl, gr = _eye_ratio(lm, L_EYE, L_IRIS), _eye_ratio(lm, R_EYE, R_IRIS)
                        yaw, pitch, roll = euler_from_matrix(mat) if mat is not None else (0.0, 0.0, 0.0)
                        blink = 0.5 * (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0))
                        self.feats.append(
                            FrameFeat(t, (gl[0] + gr[0]) / 2, (gl[1] + gr[1]) / 2, yaw, pitch, roll, blink)
                        )
                        closed = blink > 0.45
                        if closed and not self._eye_closed:
                            self.blink_events.append(t)
                        self._eye_closed = closed
                if last_lm is not None:  # rPPG at full frame rate using the latest landmarks
                    c = self._skin_rgb(fr, last_lm)
                    if c is not None:
                        self.rgb.append((t, c))
            horizon = w.t_end - self.buffer_s
            while self.feats and self.feats[0].t < horizon:
                self.feats.popleft()
            while self.rgb and self.rgb[0][0] < horizon:
                self.rgb.popleft()
            while self.blink_events and self.blink_events[0] < horizon:
                self.blink_events.popleft()
            feats = list(self.feats)
            rgb = list(self.rgb)
            blinks = list(self.blink_events)
        lat_lm = (time.perf_counter() - t0) * 1e3
        beh = self._behavior_output(w, feats, blinks, n_faces_max, lat_lm)
        live = self._liveness_output(w, rgb)
        if not self.use_objects:
            obj: ModuleOutput | None = missing(Modality.OBJECTS, w.t_start, w.t_end, "disabled")
        else:
            obj = self._objects_output(w) if run_objects else None  # None -> analyzer carries the last result
        return beh, live, obj

    # ------------------------------------------------------------------ aggregates
    def _behavior_output(
        self, w: Window, feats: list[FrameFeat], blinks: list[float], n_faces: int, lat: float
    ) -> ModuleOutput:
        if len(feats) < 6:
            return missing(Modality.BEHAVIOR, w.t_start, w.t_end, "face not tracked")
        gx = np.array([f.gaze_x for f in feats])
        gy = np.array([f.gaze_y for f in feats])
        yaw = np.array([f.yaw for f in feats])
        pitch = np.array([f.pitch for f in feats])
        ts = np.array([f.t for f in feats])
        span = float(ts[-1] - ts[0]) or 1.0
        # gaze relative to the session's own neutral (median) — cameras sit at different positions
        # vertical deviation catches glancing down at notes / a phone below the camera
        off_axis = (
            (np.abs(gx - np.median(gx)) > 0.35)
            | (np.abs(gy - np.median(gy)) > 0.6)
            | (np.abs(yaw - np.median(yaw)) > 18)
            | (np.abs(pitch - np.median(pitch)) > 15)
        )
        off_frac = float(off_axis.mean())
        # consistent side-glances: off-axis frames concentrated on one side
        side = np.sign((gx - np.median(gx))[off_axis]) if off_axis.any() else np.array([0.0])
        side_consistency = float(abs(side.mean())) if off_axis.any() else 0.0
        sweeps = reading_sweeps(gx, ts)
        head_still = float(np.clip(1 - (yaw.std() + pitch.std()) / 8.0, 0, 1))
        blink_rate = len(blinks) / span * 60.0
        # second-screen / reading score (hand-set prior weights; re-learned by the fusion stacker)
        z = (
            -2.2
            + 3.0 * off_frac * (0.5 + side_consistency)
            + 3.0 * min(sweeps, 1.0)
            + 0.8 * head_still * min(sweeps * 2, 1)
            + (0.8 if blink_rate < 4 and span > 6 else 0.0)
        )
        score = _sig(z)
        conf = float(np.clip(span / self.buffer_s, 0.2, 1.0))
        return ModuleOutput(
            module=Modality.BEHAVIOR,
            score=score,
            confidence=conf,
            t_start=w.t_start,
            t_end=w.t_end,
            latency_ms=lat,
            details={
                "off_axis_frac": round(off_frac, 3),
                "side_consistency": round(side_consistency, 3),
                "reading_sweeps_per_s": round(sweeps, 3),
                "blink_rate_per_min": round(blink_rate, 1),
                "head_stillness": round(head_still, 3),
                "yaw_deg": round(float(np.median(yaw)), 1),
                "pitch_deg": round(float(np.median(pitch)), 1),
                "faces_in_frame": n_faces,
                "second_screen_score": round(score, 3),
            },
        )

    def _liveness_output(self, w: Window, rgb: list[tuple[float, np.ndarray]]) -> ModuleOutput:
        if len(rgb) < self.fps * 6:
            return missing(Modality.LIVENESS, w.t_start, w.t_end, "rPPG buffer filling")
        ts = np.array([t for t, _ in rgb])
        fs = (len(ts) - 1) / max(1e-6, ts[-1] - ts[0])
        c = np.stack([v for _, v in rgb])
        hr_p, snr_p = pulse_snr(pos(c, fs), fs)
        hr_c, snr_c = pulse_snr(chrom(c, fs), fs)
        snr = max(snr_p, snr_c)
        hr = hr_p if snr_p >= snr_c else hr_c
        # weak, flat pulse spectrum -> suspicious; webcams under compression are noisy, so keep it soft
        score = _sig((-1.0 - snr) / 1.5)
        conf = float(np.clip((ts[-1] - ts[0]) / self.buffer_s, 0, 1)) * 0.6
        return ModuleOutput(
            module=Modality.LIVENESS,
            score=score,
            confidence=conf,
            t_start=w.t_start,
            t_end=w.t_end,
            details={
                "hr_bpm": None if math.isnan(hr) else round(hr, 1),
                "snr_db": round(snr, 2),
                "snr_pos_db": round(snr_p, 2),
                "snr_chrom_db": round(snr_c, 2),
                "fs": round(fs, 1),
            },
        )

    def _objects_output(self, w: Window) -> ModuleOutput:
        t0 = time.perf_counter()
        if not w.frames:
            return missing(Modality.OBJECTS, w.t_start, w.t_end, "no video")
        dets = self.objects().detect(w.frames[len(w.frames) // 2])
        persons = [d for d in dets if d["cls"] == "person"]
        items = {d["cls"] for d in dets}
        risk = 0.05
        reasons = []
        if len(persons) >= 2:
            risk = max(risk, 0.9)
            reasons.append(f"{len(persons)} people visible")
        if "cell phone" in items:
            risk = max(risk, 0.7)
            reasons.append("phone visible")
        if items & {"tv", "laptop"}:
            risk = max(risk, 0.5)
            reasons.append("extra screen visible")
        if "book" in items:
            risk = max(risk, 0.3)
            reasons.append("book/notes visible")
        return ModuleOutput(
            module=Modality.OBJECTS,
            score=risk,
            confidence=0.8,
            t_start=w.t_start,
            t_end=w.t_end,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            details={"detections": dets, "reasons": reasons},
        )

    def close(self) -> None:
        self.landmarker.close()
