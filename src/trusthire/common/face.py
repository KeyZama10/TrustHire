"""Face detection (SCRFD) + ArcFace alignment/embedding on raw ONNX Runtime.

Re-implements the small amount of InsightFace glue we need so that the
`insightface` package (which needs a C++ toolchain on Windows) is not required.
Models: InsightFace buffalo_l — det_10g.onnx (SCRFD-10GF) and w600k_r50.onnx (ArcFace R50).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from trusthire.common.ort import make_session

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


@dataclass
class Face:
    bbox: np.ndarray  # x1, y1, x2, y2
    kps: np.ndarray  # 5x2 (left eye, right eye, nose, left mouth, right mouth)
    score: float

    @property
    def size(self) -> float:
        return float(max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))

    @property
    def center(self) -> tuple[float, float]:
        return float((self.bbox[0] + self.bbox[2]) / 2), float((self.bbox[1] + self.bbox[3]) / 2)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class SCRFD:
    """SCRFD face detector (InsightFace det_10g.onnx)."""

    def __init__(
        self, model_path: str | Path, det_size: int = 320, det_thresh: float = 0.5, nms_thresh: float = 0.4
    ):
        self.sess = make_session(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        n_out = len(self.sess.get_outputs())
        self.fmc = 3
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.use_kps = n_out == 9
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}
        self._lock = threading.Lock()

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._centers:
            ys, xs = np.mgrid[:h, :w]
            c = np.stack([xs, ys], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            c = np.repeat(c[:, None, :], self.num_anchors, axis=1).reshape(-1, 2)
            self._centers[key] = c
        return self._centers[key]

    def detect(self, img_bgr: np.ndarray, max_num: int = 0) -> list[Face]:
        size = self.det_size
        h0, w0 = img_bgr.shape[:2]
        im_ratio = h0 / w0
        if im_ratio > 1:
            new_h, new_w = size, int(size / im_ratio)
        else:
            new_w, new_h = size, int(size * im_ratio)
        scale = new_h / h0
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(img_bgr, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(None, {self.input_name: blob})  # ORT Run is thread-safe
        if outs[0].ndim == 3:  # batched export
            outs = [o[0] for o in outs]
        scores_l, boxes_l, kps_l = [], [], []
        for i, stride in enumerate(self.strides):
            scores = outs[i].reshape(-1)
            bbox = outs[i + self.fmc] * stride
            centers = self._anchor_centers(size // stride, size // stride, stride)
            pos = np.where(scores >= self.det_thresh)[0]
            if pos.size == 0:
                continue
            c = centers[pos]
            b = bbox[pos]
            boxes = np.stack(
                [c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], axis=-1
            )
            scores_l.append(scores[pos])
            boxes_l.append(boxes)
            if self.use_kps:
                k = outs[i + 2 * self.fmc][pos] * stride
                kp = np.stack([c[:, 0:1] + k[:, 0::2], c[:, 1:2] + k[:, 1::2]], axis=-1)
                kps_l.append(kp)
        if not scores_l:
            return []
        scores_a = np.concatenate(scores_l)
        boxes_a = np.concatenate(boxes_l) / scale
        kps_a = np.concatenate(kps_l) / scale if kps_l else np.zeros((len(scores_a), 5, 2), np.float32)
        dets = np.hstack([boxes_a, scores_a[:, None]]).astype(np.float32)
        keep = _nms(dets, self.nms_thresh)
        faces = [Face(bbox=dets[i, :4], kps=kps_a[i], score=float(dets[i, 4])) for i in keep]
        faces.sort(key=lambda f: f.size, reverse=True)
        return faces[:max_num] if max_num else faces


def align_face(img_bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    dst = ARCFACE_DST * (size / 112.0)
    m, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
    if m is None:
        m = cv2.getAffineTransform(kps[:3].astype(np.float32), dst[:3])
    return cv2.warpAffine(img_bgr, m, (size, size), borderValue=(0, 0, 0))


class ArcFace:
    """ArcFace embedder (InsightFace w600k_r50.onnx / glintr100.onnx)."""

    def __init__(self, model_path: str | Path):
        self.sess = make_session(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self._lock = threading.Lock()

    def embed_aligned(self, aligned_bgr: list[np.ndarray]) -> np.ndarray:
        blob = cv2.dnn.blobFromImages(
            aligned_bgr, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        emb = self.sess.run(None, {self.input_name: blob})[0]
        return emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)

    def embed(self, img_bgr: np.ndarray, face: Face) -> np.ndarray:
        return self.embed_aligned([align_face(img_bgr, face.kps)])[0]


def square_crop(img: np.ndarray, face: Face, scale: float = 1.3, out: int = 256) -> np.ndarray:
    """Square crop around the face box with `scale` margin, zero-padded at borders."""
    cx, cy = face.center
    half = face.size * scale / 2
    return crop_box(img, cx - half, cy - half, cx + half, cy + half, (out, out))


def syncnet_crop(img: np.ndarray, face: Face) -> np.ndarray:
    """Face crop matching syncnet_python's crop_video (crop_scale=0.4), 224x224."""
    cx, cy = face.center
    bs = face.size / 2
    cs = 0.4
    return crop_box(img, cx - bs * (1 + cs), cy - bs, cx + bs * (1 + cs), cy + bs * (1 + 2 * cs), (224, 224))


def crop_box(
    img: np.ndarray, x1: float, y1: float, x2: float, y2: float, size: tuple[int, int]
) -> np.ndarray:
    h, w = img.shape[:2]
    pad = int(max(0, -x1, -y1, x2 - w, y2 - h)) + 1
    if pad > 1:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    crop = img[int(y1) : int(y2), int(x1) : int(x2)]
    if crop.size == 0:
        return np.zeros((size[1], size[0], 3), np.uint8)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


class FaceStack:
    """Lazily-built shared detector + embedder (one instance per process)."""

    _inst: FaceStack | None = None
    _lock = threading.Lock()

    def __init__(self, pack_dir: Path, det_size: int = 320):
        det = pack_dir / "det_10g.onnx"
        if not det.exists():  # antelopev2 naming
            det = pack_dir / "scrfd_10g_bnkps.onnx"
        rec = next(
            (p for p in [pack_dir / "w600k_r50.onnx", pack_dir / "glintr100.onnx"] if p.exists()), None
        )
        if rec is None:
            raise FileNotFoundError(f"no ArcFace model in {pack_dir}")
        self.detector = SCRFD(det, det_size=det_size)
        self.arcface = ArcFace(rec)

    @classmethod
    def get(cls) -> FaceStack:
        from trusthire.common.registry import ensure
        from trusthire.common.settings import get_settings

        with cls._lock:
            if cls._inst is None:
                s = get_settings()
                pack = ensure("identity.antelopev2" if s.profile == "gpu" else "identity.buffalo_l")
                cls._inst = FaceStack(pack, det_size=640 if s.profile == "gpu" else s.face_det_size)
            return cls._inst


def anchor_indices(n_frames: int, k: int = 5) -> list[int]:
    """The frames of a window on which the face detector actually runs (shared by all branches)."""
    if n_frames <= 0:
        return []
    return sorted({int(i) for i in np.linspace(0, n_frames - 1, num=min(k, n_frames)).round()})


def interpolated_faces(window_cache: dict, n_frames: int) -> list[Face | None]:
    """Largest-face track for every frame, linearly interpolated between the frames that were actually
    detected (keys 'faces:<i>' in the window cache). Avoids running the detector on all 75 frames."""
    known = sorted(
        (int(k.split(":")[1]), v[0]) for k, v in window_cache.items() if k.startswith("faces:") and v
    )
    if not known:
        return [None] * n_frames
    out: list[Face | None] = []
    j = 0
    for i in range(n_frames):
        while j + 1 < len(known) and known[j + 1][0] <= i:
            j += 1
        i0, f0 = known[j]
        if i <= i0 or j + 1 >= len(known):
            out.append(f0)
            continue
        i1, f1 = known[j + 1]
        a = (i - i0) / (i1 - i0)
        out.append(
            Face(
                bbox=(1 - a) * f0.bbox + a * f1.bbox,
                kps=(1 - a) * f0.kps + a * f1.kps,
                score=min(f0.score, f1.score),
            )
        )
    return out


def detect_cached(window_cache: dict, idx: int, frame: np.ndarray) -> list[Face]:
    """Face detection memoised per window so the video/identity/sync branches share it."""
    key = f"faces:{idx}"
    if key not in window_cache:
        window_cache[key] = FaceStack.get().detector.detect(frame, max_num=4)
    faces: list[Face] = window_cache[key]
    return faces
