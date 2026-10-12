import { AlertTriangle, Info, ShieldAlert } from "lucide-react";
import { Area, AreaChart, CartesianGrid, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtTime, MODULES, type Alert, type Answer, type ModuleKey, type ModuleOut, type Point } from "../lib/api";
import { Badge, Bar, Empty, riskColor } from "./ui";

export function RiskTimeline({ points, height = 220, highlight }: { points: Point[]; height?: number; highlight?: ModuleKey | null }) {
  const data = points.map((p) => ({
    t: Number(p.t.toFixed(1)),
    risk: Number(p.risk.toFixed(3)),
    mod: highlight && p.modules[highlight] && !p.modules[highlight]!.missing ? p.modules[highlight]!.score : null,
  }));
  if (!data.length) return <Empty>The risk timeline appears once the first 3-second window is analysed.</Empty>;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <defs>
          <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#14b8a6" stopOpacity={0.45} />
            <stop offset="100%" stopColor="#14b8a6" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
        <XAxis dataKey="t" stroke="#64748b" fontSize={11} tickFormatter={(v) => fmtTime(v)} />
        <YAxis domain={[0, 1]} stroke="#64748b" fontSize={11} />
        <Tooltip
          contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
          labelFormatter={(v) => `t = ${fmtTime(Number(v))}`}
        />
        <ReferenceLine y={0.65} stroke="#ef4444" strokeDasharray="4 4" />
        <Area type="monotone" dataKey="risk" name="fused risk" stroke="#14b8a6" strokeWidth={2} fill="url(#riskFill)"
          isAnimationActive={false} />
        {highlight && <Line type="monotone" dataKey="mod" name={highlight} stroke="#f59e0b" dot={false} strokeWidth={1.5}
          isAnimationActive={false} connectNulls />}
      </AreaChart>
    </ResponsiveContainer>
  );
}

const DETAIL_KEYS: Partial<Record<ModuleKey, [string, string, (v: unknown) => string][]>> = {
  video_deepfake: [["face_px", "face", (v) => `${v}px`], ["backbone", "model", String]],
  voice_spoof: [["speech_ratio", "speech", (v) => `${Math.round(Number(v) * 100)}%`], ["backend", "model", String]],
  av_sync: [["offset_ms", "offset", (v) => `${v} ms`], ["sync_conf", "conf", (v) => Number(v).toFixed(1)]],
  face_identity: [["sim_enrolled", "sim", (v) => (v == null ? "—" : Number(v).toFixed(2))], ["max_faces", "faces", String]],
  voice_identity: [["sim_enrolled", "sim", (v) => (v == null ? "—" : Number(v).toFixed(2))]],
  behavior: [["off_axis_frac", "off-axis", (v) => `${Math.round(Number(v) * 100)}%`], ["reading_sweeps_per_s", "sweeps/s", (v) => Number(v).toFixed(2)], ["blink_rate_per_min", "blinks/min", (v) => Number(v).toFixed(0)]],
  liveness: [["hr_bpm", "HR", (v) => (v == null ? "—" : `${Number(v).toFixed(0)} bpm`)], ["snr_db", "SNR", (v) => `${Number(v).toFixed(1)} dB`]],
  objects: [["reasons", "seen", (v) => (Array.isArray(v) && v.length ? v.join(", ") : "nothing unusual")]],
  scripted_answer: [["filler_per_100w", "fillers", (v) => Number(v).toFixed(1)], ["ai_text_prob", "AI-text", (v) => (v == null ? "—" : Number(v).toFixed(2))]],
  resume_consistency: [["support", "support", (v) => Number(v).toFixed(2)]],
};

export function ModuleCard({ k, label, hint, out, onClick, active }: {
  k: ModuleKey; label: string; hint: string; out?: ModuleOut; onClick?: () => void; active?: boolean;
}) {
  const missing = !out || out.missing;
  return (
    <button onClick={onClick}
      className={`rounded-lg border p-3 text-left transition ${active ? "border-amber-500/70 bg-slate-800/80" : "border-slate-800 bg-slate-900/60 hover:border-slate-600"}`}>
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-200">{label}</span>
        {missing ? <Badge>no evidence</Badge> : (
          <span className="text-sm font-semibold tabular-nums" style={{ color: riskColor(out!.score) }}>
            {(out!.score * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <div className="mt-2"><Bar value={missing ? 0 : out!.score} /></div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-slate-400">
        {missing ? <span>{String(out?.details?.reason ?? hint)}</span> : <>
          <span>conf {(out!.confidence * 100).toFixed(0)}%</span>
          {(DETAIL_KEYS[k] ?? []).map(([key, name, fmt]) =>
            out!.details[key] !== undefined ? <span key={key}>{name} {fmt(out!.details[key])}</span> : null)}
        </>}
      </div>
    </button>
  );
}

export function ModuleGrid({ point, active, onSelect }: { point?: Point; active?: ModuleKey | null; onSelect?: (k: ModuleKey | null) => void }) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-5">
      {MODULES.map((m) => (
        <ModuleCard key={m.key} k={m.key} label={m.label} hint={m.hint} out={point?.modules[m.key]}
          active={active === m.key} onClick={() => onSelect?.(active === m.key ? null : m.key)} />
      ))}
    </div>
  );
}

const sevIcon = { critical: ShieldAlert, warning: AlertTriangle, info: Info };
const sevTone = { critical: "text-red-400", warning: "text-amber-400", info: "text-sky-400" };

export function AlertsFeed({ alerts, max = 50 }: { alerts: Alert[]; max?: number }) {
  if (!alerts.length) return <Empty>No alerts raised.</Empty>;
  return (
    <ul className="max-h-80 space-y-2 overflow-y-auto pr-1">
      {[...alerts].reverse().slice(0, max).map((a, i) => {
        const Icon = sevIcon[a.severity] ?? Info;
        return (
          <li key={`${a.t}-${a.module}-${i}`} className="flex gap-2 rounded-md bg-slate-800/50 p-2 text-sm">
            <Icon className={`mt-0.5 size-4 shrink-0 ${sevTone[a.severity]}`} />
            <div>
              <div className="text-slate-200">{a.message}</div>
              <div className="text-xs text-slate-500">{fmtTime(a.t)} · {MODULES.find((m) => m.key === a.module)?.label ?? a.module}</div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

export function TranscriptFeed({ answers }: { answers: Answer[] }) {
  if (!answers.length) return <Empty>Answers appear here after the candidate finishes speaking (Whisper runs asynchronously).</Empty>;
  return (
    <ul className="max-h-96 space-y-3 overflow-y-auto pr-1">
      {[...answers].reverse().map((a) => (
        <li key={a.t_start} className="rounded-md bg-slate-800/50 p-3 text-sm">
          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-slate-400">
            <span>{fmtTime(a.t_start)}–{fmtTime(a.t_end)}</span>
            <Badge tone={a.scripted_score >= 0.65 ? "red" : a.scripted_score >= 0.4 ? "amber" : "green"}>
              scripted {(a.scripted_score * 100).toFixed(0)}%
            </Badge>
            {a.ai_text_prob != null && <Badge tone={a.ai_text_prob > 0.7 ? "red" : "slate"}>AI-text {(a.ai_text_prob * 100).toFixed(0)}%</Badge>}
            {a.resume && <Badge tone={a.resume.score > 0.6 ? "red" : "slate"}>resume conflict {(a.resume.score * 100).toFixed(0)}%</Badge>}
            <span>fillers/100w {Number(a.features.filler_per_100w ?? 0).toFixed(1)}</span>
            {a.features.response_latency_s != null && <span>latency {Number(a.features.response_latency_s).toFixed(1)}s</span>}
          </div>
          <p className="leading-relaxed text-slate-200">{a.text}</p>
        </li>
      ))}
    </ul>
  );
}

export function ModuleStatsTable({ stats }: { stats: Record<string, { mean: number; max: number; coverage: number; flagged: number }> }) {
  return (
    <table className="w-full text-sm">
      <thead className="text-left text-xs text-slate-400">
        <tr><th className="py-1">Module</th><th>Mean</th><th>Max</th><th>Coverage</th><th>Flagged</th></tr>
      </thead>
      <tbody>
        {MODULES.filter((m) => stats[m.key]).map((m) => {
          const s = stats[m.key];
          return (
            <tr key={m.key} className="border-t border-slate-800">
              <td className="py-1.5 text-slate-200">{m.label}</td>
              <td style={{ color: riskColor(s.mean) }}>{s.mean.toFixed(2)}</td>
              <td style={{ color: riskColor(s.max) }}>{s.max.toFixed(2)}</td>
              <td className="text-slate-400">{Math.round(s.coverage * 100)}%</td>
              <td className="text-slate-400">{s.flagged}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
