import { Camera, CheckCircle2, Circle, FileText, Flag, Loader2, Mic, Play, Square, Upload } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { AlertsFeed, ModuleGrid, RiskTimeline, TranscriptFeed } from "../components/panels";
import { Badge, Button, Card, TrustGauge } from "../components/ui";
import { api, fmtTime, type ModuleKey, type SessionSummary } from "../lib/api";
import { openMedia, recordAudio, snapshot, startWebRTC, useSessionFeed } from "../lib/live";

const READING = "Please read aloud: “I am joining this interview from my own device. The quick brown fox jumps over the lazy dog, and I will answer every question honestly in my own words.”";

function Step({ done, label, children }: { done: boolean; label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 border-t border-slate-800 py-3 first:border-0">
      {done ? <CheckCircle2 className="mt-0.5 size-5 text-emerald-400" /> : <Circle className="mt-0.5 size-5 text-slate-600" />}
      <div className="flex-1">
        <div className="mb-2 text-sm font-medium text-slate-200">{label}</div>
        {children}
      </div>
    </div>
  );
}

export default function Live() {
  const [candidate, setCandidate] = useState("");
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [phase, setPhase] = useState<"setup" | "connecting" | "live" | "stopping" | "done">("setup");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [enrolled, setEnrolled] = useState({ face: false, voice: false, resume: false });
  const [countdown, setCountdown] = useState(0);
  const [focus, setFocus] = useState<ModuleKey | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [summary, setSummary] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const startRef = useRef<number>(0);
  const feed = useSessionFeed(session && phase !== "setup" ? session.id : null);

  useEffect(() => {
    if (videoRef.current && stream) videoRef.current.srcObject = stream;
  }, [stream, phase]);

  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream]);

  useEffect(() => {
    if (phase !== "live") return;
    const id = window.setInterval(() => setElapsed((Date.now() - startRef.current) / 1000), 500);
    return () => window.clearInterval(id);
  }, [phase]);

  const run = async (label: string, fn: () => Promise<void>) => {
    setBusy(label);
    setErr(null);
    try { await fn(); } catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setBusy(null); }
  };

  const create = () => run("create", async () => {
    const s = await api.createSession(candidate || "Candidate");
    setSession(s);
    setStream(await openMedia());
  });

  const markQuestion = useCallback(() => {
    if (session && phase === "live") void api.question(session.id);
  }, [session, phase]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "q" && !(e.target instanceof HTMLInputElement)) markQuestion();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [markQuestion]);

  const start = () => run("start", async () => {
    if (!session || !stream) return;
    setPhase("connecting");
    pcRef.current = await startWebRTC(session.id, stream);
    startRef.current = Date.now();
    setPhase("live");
  });

  const stop = () => run("stop", async () => {
    if (!session) return;
    setPhase("stopping");
    pcRef.current?.close();
    await api.stop(session.id);
    stream?.getTracks().forEach((t) => t.stop());
    setPhase("done");
  });

  const last = feed.points[feed.points.length - 1];
  const lat = feed.points.slice(-10).map((p) => p.latency_ms ?? 0);
  const avgLat = lat.length ? lat.reduce((a, b) => a + b, 0) / lat.length : null;

  if (phase === "setup") {
    return (
      <div className="grid gap-4 lg:grid-cols-[1fr_420px]">
        <Card title="Camera preview">
          {stream ? (
            <video ref={videoRef} autoPlay muted playsInline className="aspect-video w-full rounded-lg bg-black object-cover" />
          ) : (
            <div className="flex aspect-video items-center justify-center rounded-lg bg-slate-950 text-sm text-slate-500">
              Create a session to open the camera and microphone
            </div>
          )}
          <p className="mt-3 text-xs text-slate-500">
            Video and audio are streamed to the TrustHire server over WebRTC and analysed in 3-second windows with 50% overlap.
            Browser noise-suppression is disabled so voice-spoofing artefacts are preserved.
          </p>
        </Card>
        <Card title="New interview">
          <Step done={!!session} label="1 · Candidate">
            <div className="flex gap-2">
              <input value={candidate} onChange={(e) => setCandidate(e.target.value)} disabled={!!session}
                placeholder="Candidate name" className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-teal-500" />
              <Button onClick={create} disabled={!!session || busy !== null}>
                {busy === "create" ? <Loader2 className="size-4 animate-spin" /> : null} Create
              </Button>
            </div>
          </Step>
          <Step done={enrolled.face} label="2 · Identity photo (ID card / reference photo)">
            <div className="flex flex-wrap gap-2">
              <label className={`inline-flex cursor-pointer items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm hover:bg-slate-800 ${!session && "pointer-events-none opacity-40"}`}>
                <Upload className="size-4" /> Upload photo
                <input type="file" accept="image/*" hidden onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f && session) void run("face", async () => { await api.enrollFace(session.id, f); setEnrolled((s) => ({ ...s, face: true })); });
                }} />
              </label>
              <Button variant="ghost" disabled={!session || !stream || busy !== null} onClick={() => run("face", async () => {
                const b = await snapshot(videoRef.current!);
                await api.enrollFace(session!.id, b);
                setEnrolled((s) => ({ ...s, face: true }));
              })}><Camera className="size-4" /> Use webcam snapshot</Button>
              {busy === "face" && <Loader2 className="size-4 animate-spin self-center" />}
            </div>
          </Step>
          <Step done={enrolled.voice} label="3 · Voice sample (8 s)">
            <p className="mb-2 text-xs text-slate-400">{READING}</p>
            <Button variant="ghost" disabled={!session || busy !== null} onClick={() => run("voice", async () => {
              const blob = await recordAudio(8, setCountdown);
              setCountdown(0);
              await api.enrollVoice(session!.id, blob);
              setEnrolled((s) => ({ ...s, voice: true }));
            })}>
              <Mic className="size-4" /> {countdown ? `Recording… ${countdown}s` : busy === "voice" ? "Processing…" : "Record voiceprint"}
            </Button>
          </Step>
          <Step done={enrolled.resume} label="4 · Resume (optional, enables consistency check)">
            <label className={`inline-flex cursor-pointer items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm hover:bg-slate-800 ${!session && "pointer-events-none opacity-40"}`}>
              <FileText className="size-4" /> Upload PDF / TXT
              <input type="file" accept=".pdf,.txt,.md" hidden onChange={(e) => {
                const f = e.target.files?.[0];
                if (f && session) void run("resume", async () => { await api.resume(session.id, f); setEnrolled((s) => ({ ...s, resume: true })); });
              }} />
            </label>
          </Step>
          <div className="mt-3 flex items-center justify-between gap-2">
            <span className="text-xs text-slate-500">Skipped enrollment? Identity is auto-enrolled from the first seconds.</span>
            <Button onClick={start} disabled={!session || !stream || busy !== null}><Play className="size-4" /> Start interview</Button>
          </div>
          {err && <p className="mt-3 rounded-md bg-red-950/60 p-2 text-sm text-red-300">{err}</p>}
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1.4fr)]">
        <Card title={<span className="flex items-center gap-2">{session?.candidate}
          {phase === "live" && <Badge tone="red">● LIVE {fmtTime(elapsed)}</Badge>}
          {phase === "done" && <Badge tone="teal">finished</Badge>}
          {phase === "connecting" && <Badge tone="amber">connecting…</Badge>}</span>}
          right={<span className="text-xs text-slate-500">{feed.connected ? "feed connected" : "feed offline"} · peer {feed.peer}</span>}>
          {phase !== "done" ? (
            <video ref={videoRef} autoPlay muted playsInline className="aspect-video w-full rounded-lg bg-black object-cover" />
          ) : (
            <div className="space-y-3">
              <p className="text-sm text-slate-300">Interview finished. Generate the explainability report (Grad-CAM, audio saliency, fusion attribution and an LLM summary).</p>
              <div className="flex flex-wrap gap-2">
                <Button disabled={busy !== null} onClick={() => run("report", async () => {
                  const r = await api.report(session!.id);
                  setSummary(r.summary);
                })}>{busy === "report" ? <Loader2 className="size-4 animate-spin" /> : <FileText className="size-4" />} Generate report</Button>
                {summary !== null && <a href={`/api/sessions/${session!.id}/report.pdf`} target="_blank" rel="noreferrer"><Button variant="ghost">Download PDF</Button></a>}
                <a href={`#/sessions/${session!.id}`}><Button variant="ghost">Open session</Button></a>
              </div>
              {summary && <pre className="whitespace-pre-wrap rounded-lg bg-slate-950 p-3 text-sm text-slate-300">{summary}</pre>}
            </div>
          )}
          {phase === "live" && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Button variant="ghost" onClick={markQuestion}><Flag className="size-4" /> Mark question asked <kbd className="rounded bg-slate-800 px-1 text-[10px]">Q</kbd></Button>
              <Button variant="danger" onClick={stop} disabled={busy !== null}><Square className="size-4" /> Stop & finish</Button>
              <span className="ml-auto text-xs text-slate-500">
                {feed.points.length} windows · {avgLat ? `analysis ${Math.round(avgLat)} ms/window` : "warming up…"}
              </span>
            </div>
          )}
          {phase === "stopping" && <p className="mt-3 flex items-center gap-2 text-sm text-slate-400"><Loader2 className="size-4 animate-spin" /> Finishing analysis and transcript…</p>}
          {err && <p className="mt-3 rounded-md bg-red-950/60 p-2 text-sm text-red-300">{err}</p>}
        </Card>
        <div className="grid gap-4">
          <Card title="Trust & risk timeline" right={focus && <Badge tone="amber">overlay: {focus}</Badge>}>
            <div className="flex flex-col gap-4 md:flex-row md:items-center">
              <div className="flex flex-col items-center gap-1">
                <TrustGauge value={feed.trust?.trust_score} />
                <span className="text-xs text-slate-400">current risk {last ? (last.risk * 100).toFixed(0) : "—"}%</span>
              </div>
              <div className="min-w-0 flex-1"><RiskTimeline points={feed.points} highlight={focus} /></div>
            </div>
          </Card>
          <Card title="Alerts" right={<Badge tone={feed.alerts.length ? "red" : "slate"}>{feed.alerts.length}</Badge>}>
            <AlertsFeed alerts={feed.alerts} />
          </Card>
        </div>
      </div>
      <Card title="Modules (latest window)" right={<span className="text-xs text-slate-500">click a module to overlay it on the timeline</span>}>
        <ModuleGrid point={last} active={focus} onSelect={setFocus} />
      </Card>
      <Card title="Transcript & answer analysis"><TranscriptFeed answers={feed.answers} /></Card>
    </div>
  );
}
