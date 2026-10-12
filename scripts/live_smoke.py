"""End-to-end smoke test of the LIVE path without a webcam.

Streams a video file to a running TrustHire server over real WebRTC (aiortc MediaPlayer acts as the browser),
listens on the session WebSocket and reports the windows/alerts that come back, then stops the session.

    python scripts/live_smoke.py [video] [--server http://127.0.0.1:8000] [--seconds 25]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

from trusthire.common.registry import ensure


async def main(video: str, server: str, seconds: float) -> None:
    async with httpx.AsyncClient(base_url=server, timeout=120) as http:
        sess = (await http.post("/api/sessions", json={"candidate": "Live smoke test (WebRTC)"})).json()
        sid = sess["id"]
        print("session", sid)
        ws_url = server.replace("http", "ws") + f"/ws/sessions/{sid}"
        stats = {"points": 0, "alerts": 0, "answers": 0, "lat": []}

        async def listen() -> None:
            async with websockets.connect(ws_url, max_size=2**24) as ws:
                async for raw in ws:
                    m = json.loads(raw)
                    if m["type"] == "point":
                        stats["points"] += 1
                        p = m["point"]
                        stats["lat"].append(p["latency_ms"])
                        mods = {k: round(v["score"], 2) for k, v in p["modules"].items() if not v["missing"]}
                        print(
                            f"  window {p['t_start']:5.1f}-{p['t_end']:5.1f}s risk={p['risk']:.3f} "
                            f"latency={p['latency_ms']:.0f}ms trust={m['trust']['trust_score']} {mods}"
                        )
                    elif m["type"] == "alert":
                        stats["alerts"] += 1
                        print("  ALERT", m["alert"]["message"])
                    elif m["type"] == "answer":
                        stats["answers"] += 1
                        print("  ANSWER", m["answer"]["text"][:90])

        listener = asyncio.create_task(listen())
        player = MediaPlayer(video)
        pc = RTCPeerConnection()
        if player.video:
            pc.addTrack(player.video)
        if player.audio:
            pc.addTrack(player.audio)
        await pc.setLocalDescription(await pc.createOffer())
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)
        ans = (
            await http.post(
                f"/api/sessions/{sid}/webrtc/offer",
                json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
            )
        ).json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
        t0 = time.time()
        await asyncio.sleep(seconds)
        await pc.close()
        print("stopping…")
        final = (await http.post(f"/api/sessions/{sid}/stop")).json()
        await asyncio.sleep(1)
        listener.cancel()
        lat = sorted(x for x in stats["lat"] if x)
        print(
            f"\nstreamed {time.time() - t0:.0f}s -> {stats['points']} windows, {stats['alerts']} alerts, "
            f"{stats['answers']} answers; latency p50 {lat[len(lat) // 2] if lat else float('nan'):.0f} ms; "
            f"final trust {final.get('trust')}"
        )
        if stats["points"] == 0:
            raise SystemExit("FAIL: no windows were analysed over the live path")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--seconds", type=float, default=25)
    a = ap.parse_args()
    asyncio.run(main(a.video or str(ensure("avsync.example")), a.server, a.seconds))
