export type ModuleKey =
  | "video_deepfake"
  | "voice_spoof"
  | "av_sync"
  | "face_identity"
  | "voice_identity"
  | "behavior"
  | "liveness"
  | "objects"
  | "scripted_answer"
  | "resume_consistency";

export interface ModuleOut {
  score: number;
  confidence: number;
  missing: boolean;
  latency_ms: number;
  details: Record<string, unknown>;
}

export interface Point {
  t: number;
  t_start: number;
  t_end: number;
  risk: number;
  latency_ms: number | null;
  alerts: string[];
  contributions: Record<string, number>;
  modules: Partial<Record<ModuleKey, ModuleOut>>;
}

export interface Alert {
  t: number;
  module: string;
  severity: "info" | "warning" | "critical";
  message: string;
}

export interface Trust {
  trust_score: number;
  session_risk: number;
  peak_risk: number;
  mean_risk?: number;
  flagged_fraction?: number;
}

export interface Answer {
  t_start: number;
  t_end: number;
  text: string;
  features: Record<string, number | null>;
  ai_text_prob: number | null;
  scripted_score: number;
  resume: { score: number; support: number; checks?: { claim: string; evidence: string; contradiction: number }[] } | null;
}

export interface SessionSummary {
  id: string;
  candidate: string;
  status: string;
  created: string;
  source: string;
  trust: Trust | null;
  n_windows: number;
  n_alerts: number;
  duration_s: number;
  enrolled_face?: boolean;
  enrolled_voice?: boolean;
  has_resume?: boolean;
}

export interface SessionDetail extends SessionSummary {
  points: Point[];
  alerts: Alert[];
  answers: Answer[];
  module_stats: Record<string, { mean: number; max: number; coverage: number; flagged: number }>;
  job: { state: string; progress?: number; error?: string; backend?: string };
  frames: { file: string; t: number; p_fake: number }[];
  has_report: boolean;
}

export const MODULES: { key: ModuleKey; label: string; group: string; hint: string }[] = [
  { key: "video_deepfake", label: "Video deepfake", group: "Video", hint: "Face-forgery CNN on face crops" },
  { key: "face_identity", label: "Face identity", group: "Video", hint: "ArcFace vs enrolled ID photo" },
  { key: "voice_spoof", label: "Synthetic voice", group: "Audio", hint: "Anti-spoofing on raw waveform" },
  { key: "voice_identity", label: "Voice identity", group: "Audio", hint: "ECAPA-TDNN vs voiceprint" },
  { key: "av_sync", label: "Lip-sync", group: "Audio-visual", hint: "SyncNet offset & confidence" },
  { key: "behavior", label: "Gaze / 2nd screen", group: "Behaviour", hint: "Gaze, head pose, reading sweeps" },
  { key: "liveness", label: "Liveness (rPPG)", group: "Behaviour", hint: "Pulse signal from skin colour" },
  { key: "objects", label: "Objects / people", group: "Behaviour", hint: "YOLO: phone, 2nd person, screen" },
  { key: "scripted_answer", label: "Scripted answer", group: "Transcript", hint: "Fluency, fillers, AI-text" },
  { key: "resume_consistency", label: "Resume check", group: "Transcript", hint: "RAG + NLI vs resume" },
];

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = j.detail ?? msg;
    } catch {
      /* not JSON */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const form = (fields: Record<string, Blob | string | undefined | null>): RequestInit => {
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) if (v) fd.append(k, v);
  return { method: "POST", body: fd };
};

export const api = {
  health: () => req<{ ok: boolean; models_ready: boolean; version: string }>("/api/health"),
  system: () => req<Record<string, unknown>>("/api/system"),
  createSession: (candidate: string) => req<SessionSummary>("/api/sessions", json({ candidate })),
  sessions: () => req<SessionSummary[]>("/api/sessions"),
  session: (id: string) => req<SessionDetail>(`/api/sessions/${id}`),
  deleteSession: (id: string) => req<{ deleted: boolean }>(`/api/sessions/${id}`, { method: "DELETE" }),
  enrollFace: (id: string, file: Blob) => req(`/api/sessions/${id}/enroll/face`, form({ file })),
  enrollVoice: (id: string, file: Blob) =>
    req<{ seconds: number }>(`/api/sessions/${id}/enroll/voice`, form({ file: new File([file], "voice.webm") })),
  resume: (id: string, file: File) => req<{ chunks: number }>(`/api/sessions/${id}/resume`, form({ file })),
  offer: (id: string, sdp: RTCSessionDescriptionInit) =>
    req<RTCSessionDescriptionInit>(`/api/sessions/${id}/webrtc/offer`, json(sdp)),
  question: (id: string) => req(`/api/sessions/${id}/question`, json({})),
  stop: (id: string) => req<SessionSummary>(`/api/sessions/${id}/stop`, { method: "POST" }),
  report: (id: string) => req<{ summary: string; seconds: number }>(`/api/sessions/${id}/report`, { method: "POST" }),
  upload: (video: File, candidate: string, idPhoto?: File | null, resume?: File | null) =>
    req<{ session_id: string; backend: string }>(
      "/api/uploads",
      form({ video, candidate, id_photo: idPhoto ?? undefined, resume: resume ?? undefined }),
    ),
};

export const wsUrl = (id: string) =>
  `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/sessions/${id}`;

export const fmtTime = (s: number) => {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};
