import { FileVideo, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Bar, Button, Card } from "../components/ui";
import { api } from "../lib/api";

export default function UploadPage() {
  const [video, setVideo] = useState<File | null>(null);
  const [photo, setPhoto] = useState<File | null>(null);
  const [resume, setResume] = useState<File | null>(null);
  const [candidate, setCandidate] = useState("");
  const [sid, setSid] = useState<string | null>(null);
  const [backend, setBackend] = useState("");
  const [job, setJob] = useState<{ state: string; progress?: number; error?: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!sid) return;
    const id = window.setInterval(async () => {
      const d = await api.session(sid);
      setJob(d.job);
      if (d.job.state === "done" || d.job.state === "failed") window.clearInterval(id);
    }, 1500);
    return () => window.clearInterval(id);
  }, [sid]);

  const submit = async () => {
    if (!video) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await api.upload(video, candidate || video.name, photo, resume);
      setSid(r.session_id);
      setBackend(r.backend);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const fileInput = (label: string, accept: string, f: File | null, set: (f: File | null) => void) => (
    <label className="flex cursor-pointer items-center justify-between gap-3 rounded-lg border border-dashed border-slate-700 px-3 py-3 text-sm hover:border-teal-600">
      <span className="text-slate-300">{label}</span>
      <span className="truncate text-xs text-slate-500">{f ? f.name : "choose file"}</span>
      <input type="file" hidden accept={accept} onChange={(e) => set(e.target.files?.[0] ?? null)} />
    </label>
  );

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <Card title="Analyse a recorded interview">
        <div className="space-y-3">
          <input value={candidate} onChange={(e) => setCandidate(e.target.value)} placeholder="Candidate name"
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-teal-500" />
          {fileInput("Interview video (mp4, webm, mov, avi)", "video/*", video, setVideo)}
          {fileInput("ID / reference photo (optional)", "image/*", photo, setPhoto)}
          {fileInput("Resume PDF/TXT (optional)", ".pdf,.txt,.md", resume, setResume)}
          <div className="flex justify-end">
            <Button onClick={submit} disabled={!video || busy || (!!sid && job?.state !== "done" && job?.state !== "failed")}>
              {busy ? <Loader2 className="size-4 animate-spin" /> : <FileVideo className="size-4" />} Upload & analyse
            </Button>
          </div>
          {err && <p className="rounded-md bg-red-950/60 p-2 text-sm text-red-300">{err}</p>}
        </div>
      </Card>
      {sid && (
        <Card title="Job" right={<span className="text-xs text-slate-500">runner: {backend}</span>}>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between"><span className="text-slate-300">{job?.state ?? "queued"}</span>
              <span className="tabular-nums text-slate-400">{Math.round((job?.progress ?? 0) * 100)}%</span></div>
            <Bar value={job?.progress ?? 0} color="#14b8a6" />
            {job?.state === "done" && <a className="inline-block text-teal-400 hover:underline" href={`#/sessions/${sid}`}>Open results →</a>}
            {job?.state === "failed" && <pre className="whitespace-pre-wrap text-xs text-red-300">{job.error}</pre>}
          </div>
        </Card>
      )}
    </div>
  );
}
