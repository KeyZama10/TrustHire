import { useEffect, useRef, useState } from "react";
import { api, wsUrl, type Alert, type Answer, type Point, type Trust } from "./api";

/** Open camera+mic with processing off (noise suppression/AGC would mask voice-spoof artefacts). */
export async function openMedia(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 25, max: 30 } },
    audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
  });
}

/** Send a stream to the backend over WebRTC (aiortc answers; no trickle ICE, so wait for gathering). */
export async function startWebRTC(sessionId: string, stream: MediaStream): Promise<RTCPeerConnection> {
  const pc = new RTCPeerConnection({ iceServers: [] });
  stream.getTracks().forEach((t) => pc.addTrack(t, stream));
  await pc.setLocalDescription(await pc.createOffer());
  await new Promise<void>((resolve) => {
    if (pc.iceGatheringState === "complete") return resolve();
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(resolve, 2500);
  });
  const answer = await api.offer(sessionId, pc.localDescription!.toJSON());
  await pc.setRemoteDescription(answer);
  return pc;
}

export interface LiveState {
  points: Point[];
  alerts: Alert[];
  answers: Answer[];
  trust: Trust | null;
  status: string;
  peer: string;
  connected: boolean;
}

/** Subscribe to a session's WebSocket feed. */
export function useSessionFeed(sessionId: string | null): LiveState {
  const [s, setS] = useState<LiveState>({
    points: [], alerts: [], answers: [], trust: null, status: "idle", peer: "new", connected: false,
  });
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    let closed = false;
    let retry: number | undefined;
    const connect = () => {
      const ws = new WebSocket(wsUrl(sessionId));
      wsRef.current = ws;
      ws.onopen = () => setS((p) => ({ ...p, connected: true }));
      ws.onclose = () => {
        setS((p) => ({ ...p, connected: false }));
        if (!closed) retry = window.setTimeout(connect, 1500);
      };
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        setS((p) => {
          switch (m.type) {
            case "hello":
              return { ...p, points: m.points, alerts: m.alerts, status: m.session.status, trust: m.session.trust };
            case "point":
              return { ...p, points: [...p.points, m.point], trust: m.trust };
            case "update": {
              const pts = [...p.points];
              const i = pts.findIndex((x) => Math.abs(x.t_start - m.point.t_start) < 1e-3);
              if (i >= 0) pts[i] = m.point;
              return { ...p, points: pts };
            }
            case "alert":
              return { ...p, alerts: [...p.alerts, m.alert] };
            case "answer":
              return { ...p, answers: [...p.answers, m.answer] };
            case "status":
              return { ...p, status: m.status, trust: m.trust ?? p.trust };
            case "peer":
              return { ...p, peer: m.state };
            default:
              return p;
          }
        });
      };
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry);
      wsRef.current?.close();
    };
  }, [sessionId]);
  return s;
}

/** Record `seconds` of microphone audio as a webm blob (voice enrollment). */
export async function recordAudio(seconds: number, onTick?: (left: number) => void): Promise<Blob> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
  });
  const rec = new MediaRecorder(stream);
  const chunks: BlobPart[] = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  const done = new Promise<Blob>((resolve) => {
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      resolve(new Blob(chunks, { type: rec.mimeType }));
    };
  });
  rec.start();
  for (let left = seconds; left > 0; left--) {
    onTick?.(left);
    await new Promise((r) => setTimeout(r, 1000));
  }
  rec.stop();
  return done;
}

/** Grab a JPEG snapshot from a <video> element (webcam enrollment when no ID photo is at hand). */
export function snapshot(video: HTMLVideoElement): Promise<Blob> {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d")!.drawImage(video, 0, 0);
  return new Promise((resolve) => c.toBlob((b) => resolve(b!), "image/jpeg", 0.92));
}
