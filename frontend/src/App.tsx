import { Activity, History, Server, ShieldCheck, Upload } from "lucide-react";
import { useEffect, useState } from "react";
import { Card } from "./components/ui";
import { api } from "./lib/api";
import Live from "./pages/Live";
import { SessionList, SessionView } from "./pages/Sessions";
import UploadPage from "./pages/Upload";

function useHash() {
  const [h, setH] = useState(location.hash || "#/live");
  useEffect(() => {
    const on = () => setH(location.hash || "#/live");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return h;
}

function SystemPage() {
  const [s, setS] = useState<Record<string, unknown> | null>(null);
  useEffect(() => { api.system().then(setS).catch(() => setS({ error: "API unreachable" })); }, []);
  return (
    <Card title="System">
      <pre className="overflow-x-auto text-xs text-slate-300">{s ? JSON.stringify(s, null, 2) : "loading…"}</pre>
    </Card>
  );
}

const NAV = [
  { href: "#/live", label: "Live interview", icon: Activity },
  { href: "#/upload", label: "Upload recording", icon: Upload },
  { href: "#/sessions", label: "Sessions", icon: History },
  { href: "#/system", label: "System", icon: Server },
];

export default function App() {
  const hash = useHash();
  const [ready, setReady] = useState<boolean | null>(null);
  useEffect(() => {
    const poll = () => api.health().then((h) => setReady(h.models_ready)).catch(() => setReady(null));
    void poll();
    const t = window.setInterval(poll, 5000);
    return () => window.clearInterval(t);
  }, []);
  const sessionMatch = hash.match(/^#\/sessions\/(.+)$/);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-4 px-4 py-3">
          <a href="#/live" className="flex items-center gap-2 font-semibold">
            <ShieldCheck className="size-6 text-teal-400" /> TrustHire
            <span className="hidden text-xs font-normal text-slate-500 sm:inline">multimodal interview integrity</span>
          </a>
          <nav className="flex flex-wrap gap-1">
            {NAV.map(({ href, label, icon: Icon }) => (
              <a key={href} href={href}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm ${hash.startsWith(href) ? "bg-slate-800 text-white" : "text-slate-400 hover:text-slate-200"}`}>
                <Icon className="size-4" /> {label}
              </a>
            ))}
          </nav>
          <span className="ml-auto flex items-center gap-2 text-xs text-slate-400">
            <span className={`size-2 rounded-full ${ready ? "bg-emerald-400" : ready === false ? "animate-pulse bg-amber-400" : "bg-red-500"}`} />
            {ready ? "models ready" : ready === false ? "loading models…" : "API offline"}
          </span>
        </div>
      </header>
      <main className="mx-auto max-w-[1500px] px-4 py-5">
        {sessionMatch ? <SessionView id={sessionMatch[1]} /> :
          hash.startsWith("#/upload") ? <UploadPage /> :
          hash.startsWith("#/sessions") ? <SessionList /> :
          hash.startsWith("#/system") ? <SystemPage /> : <Live />}
      </main>
      <footer className="pb-6 text-center text-xs text-slate-600">
        TrustHire produces evidence for human reviewers, not verdicts. Scores can be wrong — see the limitations in the report.
      </footer>
    </div>
  );
}
