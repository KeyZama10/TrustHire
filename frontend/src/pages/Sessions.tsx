import { Download, FileText, Loader2, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { AlertsFeed, ModuleGrid, ModuleStatsTable, RiskTimeline, TranscriptFeed } from "../components/panels";
import { Badge, Button, Card, Empty, TrustGauge, trustColor } from "../components/ui";
import { api, fmtTime, type ModuleKey, type SessionDetail, type SessionSummary } from "../lib/api";

export function SessionList() {
  const [rows, setRows] = useState<SessionSummary[] | null>(null);
  const load = () => api.sessions().then(setRows).catch(() => setRows([]));
  useEffect(() => { void load(); }, []);
  if (!rows) return <Loader2 className="mx-auto size-6 animate-spin text-slate-500" />;
  return (
    <Card title="Session history" right={<span className="text-xs text-slate-500">{rows.length} sessions</span>}>
      {!rows.length ? <Empty>No sessions yet. Start a live interview or upload a recording.</Empty> : (
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-slate-400">
            <tr><th className="py-2">Candidate</th><th>When</th><th>Source</th><th>Duration</th><th>Alerts</th><th>Trust</th><th>Status</th><th /></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="cursor-pointer border-t border-slate-800 hover:bg-slate-800/40"
                onClick={() => (location.hash = `#/sessions/${r.id}`)}>
                <td className="py-2 text-slate-200">{r.candidate}</td>
                <td className="text-slate-400">{new Date(r.created).toLocaleString()}</td>
                <td><Badge>{r.source}</Badge></td>
                <td className="text-slate-400">{r.duration_s ? fmtTime(r.duration_s) : "—"}</td>
                <td className="text-slate-400">{r.n_alerts ?? 0}</td>
                <td className="font-semibold tabular-nums" style={{ color: r.trust?.trust_score != null ? trustColor(r.trust.trust_score) : undefined }}>
                  {r.trust?.trust_score ?? "—"}</td>
                <td><Badge tone={r.status === "done" ? "teal" : r.status === "failed" ? "red" : "amber"}>{r.status}</Badge></td>
                <td onClick={(e) => e.stopPropagation()}>
                  <button title="delete" className="text-slate-500 hover:text-red-400" onClick={async () => {
                    if (confirm(`Delete session ${r.candidate}?`)) { await api.deleteSession(r.id); void load(); }
                  }}><Trash2 className="size-4" /></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}

export function SessionView({ id }: { id: string }) {
  const [d, setD] = useState<SessionDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [focus, setFocus] = useState<ModuleKey | null>(null);
  const [sel, setSel] = useState<number | null>(null);

  const load = () => api.session(id).then(setD).catch((e) => setErr(String(e)));
  useEffect(() => {
    void load();
    fetch(`/api/sessions/${id}/summary`).then((r) => r.json()).then((j) => setSummary(j.summary)).catch(() => {});
  }, [id]);
  useEffect(() => {
    if (!d || d.status === "done" || d.status === "failed") return;
    const t = window.setInterval(load, 2000);
    return () => window.clearInterval(t);
  }, [d?.status]);

  if (err) return <p className="text-red-400">{err}</p>;
  if (!d) return <Loader2 className="mx-auto size-6 animate-spin text-slate-500" />;
  const point = sel != null ? d.points[sel] : d.points.reduce((a, b) => (b.risk > (a?.risk ?? -1) ? b : a), d.points[0]);

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
        <Card>
          <div className="flex flex-col items-center gap-2">
            <TrustGauge value={d.trust?.trust_score} />
            <div className="text-center">
              <div className="font-semibold text-slate-100">{d.candidate}</div>
              <div className="text-xs text-slate-500">{new Date(d.created).toLocaleString()} · {d.source}</div>
            </div>
            <div className="grid w-full grid-cols-2 gap-2 text-center text-xs">
              <div className="rounded bg-slate-800/60 p-2"><div className="text-slate-400">peak risk</div><div className="text-slate-100">{d.trust?.peak_risk?.toFixed(2) ?? "—"}</div></div>
              <div className="rounded bg-slate-800/60 p-2"><div className="text-slate-400">duration</div><div className="text-slate-100">{d.duration_s ? fmtTime(d.duration_s) : "—"}</div></div>
              <div className="rounded bg-slate-800/60 p-2"><div className="text-slate-400">windows</div><div className="text-slate-100">{d.points.length}</div></div>
              <div className="rounded bg-slate-800/60 p-2"><div className="text-slate-400">alerts</div><div className="text-slate-100">{d.alerts.length}</div></div>
            </div>
            {d.status !== "done" && <Badge tone="amber">{d.job?.state ?? d.status} {d.job?.progress != null && `${Math.round(d.job.progress * 100)}%`}</Badge>}
            <div className="flex w-full flex-col gap-2 pt-2">
              <Button disabled={busy || d.status !== "done"} onClick={async () => {
                setBusy(true);
                try { const r = await api.report(id); setSummary(r.summary); await load(); } catch (e) { setErr(String(e)); } finally { setBusy(false); }
              }}>{busy ? <Loader2 className="size-4 animate-spin" /> : <FileText className="size-4" />} {d.has_report ? "Regenerate report" : "Generate report"}</Button>
              {d.has_report && <a href={`/api/sessions/${id}/report.pdf`} target="_blank" rel="noreferrer"><Button variant="ghost" className="w-full"><Download className="size-4" /> Download PDF</Button></a>}
            </div>
          </div>
        </Card>
        <Card title="Risk timeline" right={<span className="text-xs text-slate-500">click a point to inspect that window</span>}>
          <div onClick={() => setSel(null)}><RiskTimeline points={d.points} height={250} highlight={focus} /></div>
          <div className="mt-2 flex gap-1 overflow-x-auto pb-1">
            {d.points.map((p, i) => (
              <button key={i} title={`${fmtTime(p.t_start)} risk ${p.risk.toFixed(2)}`} onClick={() => setSel(i)}
                className={`h-5 min-w-2 flex-1 rounded-sm ${sel === i ? "ring-2 ring-white" : ""}`}
                style={{ background: `rgba(${p.risk > 0.65 ? "239,68,68" : p.risk > 0.4 ? "245,158,11" : "16,185,129"},${0.25 + p.risk * 0.75})` }} />
            ))}
          </div>
        </Card>
      </div>
      {summary && <Card title="AI summary"><pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-slate-300">{summary}</pre></Card>}
      <Card title={`Window ${point ? `${fmtTime(point.t_start)}–${fmtTime(point.t_end)} (risk ${point.risk.toFixed(2)})` : ""}`}
        right={point && Object.keys(point.contributions).length > 0 && <span className="text-xs text-slate-500">
          top driver: {Object.entries(point.contributions).sort((a, b) => b[1] - a[1])[0][0]}</span>}>
        <ModuleGrid point={point} active={focus} onSelect={setFocus} />
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Per-module evidence"><ModuleStatsTable stats={d.module_stats} /></Card>
        <Card title="Alerts"><AlertsFeed alerts={d.alerts} /></Card>
      </div>
      {d.frames.length > 0 && (
        <Card title="Most suspicious face crops" right={<span className="text-xs text-slate-500">Grad-CAM overlays appear after the report is generated</span>}>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {d.frames.map((f) => (
              <figure key={f.file} className="space-y-1">
                <img className="w-full rounded-md" src={`/api/sessions/${id}/frames/${d.has_report ? `cam_${f.file}` : f.file}`}
                  onError={(e) => ((e.target as HTMLImageElement).src = `/api/sessions/${id}/frames/${f.file}`)} alt="" />
                <figcaption className="text-center text-xs text-slate-400">{fmtTime(f.t)} · P(fake) {f.p_fake.toFixed(2)}</figcaption>
              </figure>
            ))}
          </div>
        </Card>
      )}
      <Card title="Transcript & answer analysis"><TranscriptFeed answers={d.answers} /></Card>
    </div>
  );
}
