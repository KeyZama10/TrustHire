"""WebRTC ingest (aiortc): browser camera+mic -> 25 fps frames + 16 kHz audio -> chunker -> session worker."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import av
import cv2
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

from trusthire.common.settings import get_settings
from trusthire.pipeline.chunker import SlidingChunker
from trusthire.pipeline.session import Session

log = logging.getLogger(__name__)
PEERS: dict[str, RTCPeerConnection] = {}


class LiveIngest:
    """Both tracks are stamped on the receive clock (time since the first media packet), which keeps
    audio and video aligned to within network jitter; audio time advances by sample count."""

    def __init__(self, session: Session, max_side: int = 640):
        s = get_settings()
        self.session = session
        self.chunker = SlidingChunker(s.window_s, s.hop_s)
        self.fps = s.analysis_fps
        self.max_side = max_side
        self.t0: float | None = None
        self.next_vt = 0.0
        self.audio_samples = 0
        self.audio_started_at: float | None = None
        self.video_anchor: tuple[float, float] | None = (
            None  # (receive-clock time, media time) of first frame
        )
        self.resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        self._lock = asyncio.Lock()

    def _now(self) -> float:
        if self.t0 is None:
            self.t0 = time.monotonic()
            self.session.t0 = 0.0
        return time.monotonic() - self.t0

    async def video(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                frame: Any = await track.recv()  # av.VideoFrame
                # Sender media clock (RTP timestamps) for frame spacing, anchored once to the first arrival.
                # Arrival-time stamping jitters under load (bursty delivery) and scrambles lip motion for
                # SyncNet (measured: live offset drifted +1 -> -3 frames, confidence 1.7 vs 6-10 offline).
                if frame.pts is not None and frame.time_base is not None:
                    if self.video_anchor is None:
                        self.video_anchor = (self._now(), float(frame.pts * frame.time_base))
                    t = self.video_anchor[0] + float(frame.pts * frame.time_base) - self.video_anchor[1]
                else:
                    t = self._now()
                if t + 1e-3 < self.next_vt:
                    continue
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                if max(h, w) > self.max_side:
                    sc = self.max_side / max(h, w)
                    img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
                async with self._lock:
                    while (
                        self.next_vt <= t + 1e-3
                    ):  # constant 25 fps grid (duplicate if the camera is slower)
                        self.chunker.push_frame(self.next_vt, img)
                        self.next_vt += 1.0 / self.fps
                    self._emit()
        except MediaStreamError:
            pass

    async def audio(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                frame: Any = await track.recv()  # av.AudioFrame
                t = self._now()
                if self.audio_started_at is None:
                    self.audio_started_at = t
                    # pad silence so audio sample 0 sits at the receive time of the first audio packet
                    self.chunker.push_audio(np.zeros(int(t * 16000), np.float32))
                    self.audio_samples = int(t * 16000)
                chunks = [f.to_ndarray().reshape(-1) for f in self.resampler.resample(frame)]
                if not chunks:
                    continue
                x = np.concatenate(chunks).astype(np.float32)
                async with self._lock:
                    self.chunker.push_audio(x)
                    self.audio_samples += len(x)
                    self._emit()
        except MediaStreamError:
            pass

    def _emit(self) -> None:
        for w in self.chunker.pop_windows():
            self.session.submit(w)


async def handle_offer(session: Session, sdp: str, typ: str) -> dict[str, str]:
    pc = RTCPeerConnection()
    PEERS[session.id] = pc
    ingest = LiveIngest(session)
    loop = asyncio.get_running_loop()
    tasks: list[asyncio.Task[None]] = []

    @pc.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        log.info("session %s: %s track", session.id, track.kind)
        if track.kind == "video":
            tasks.append(loop.create_task(ingest.video(track)))
        elif track.kind == "audio":
            tasks.append(loop.create_task(ingest.audio(track)))

    @pc.on("connectionstatechange")
    async def on_state() -> None:
        log.info("session %s: peer %s", session.id, pc.connectionState)
        session.broadcast({"type": "peer", "state": pc.connectionState})
        if pc.connectionState in ("failed", "closed"):
            await close_peer(session.id)

    if session.analyzer is None:
        await loop.run_in_executor(None, session.start_analysis, loop)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=typ))
    await pc.setLocalDescription(await pc.createAnswer())
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


async def close_peer(sid: str) -> None:
    pc = PEERS.pop(sid, None)
    if pc is not None:
        await pc.close()
