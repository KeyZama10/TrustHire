"""Provider-agnostic LLM summary of a session (Anthropic by default, Ollama locally, or a deterministic template).

Only aggregate, de-identified evidence (scores, timestamps, alert texts, short transcript excerpts) is sent —
never raw video/audio.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

log = logging.getLogger(__name__)

SYSTEM = (
    "You write concise, neutral integrity summaries of remote job interviews for a hiring panel. "
    "You are given machine-generated evidence from a multimodal fraud-detection system. "
    "Describe what the evidence shows, cite timestamps, say how confident the system is, and note benign "
    "explanations (lighting, bandwidth, nervousness, accent) where relevant. Never state that the candidate is "
    "guilty of fraud; recommend human review where risk is elevated. Plain prose, 120-220 words, then up to 4 "
    "short bullet points of recommended follow-ups."
)


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5"):
        import anthropic

        self.client = (
            anthropic.Anthropic()
        )  # ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("LLM declined to summarise this session")
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def complete(self, system: str, prompt: str) -> str:
        import httpx

        r = httpx.post(
            f"{self.host}/api/chat",
            timeout=120,
            json={
                "model": self.model,
                "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        return str(r.json()["message"]["content"]).strip()


class TemplateProvider:
    """Deterministic fallback so reports always have a summary, even offline."""

    name = "template"

    def complete(self, system: str, prompt: str) -> str:
        ev = json.loads(prompt.split("EVIDENCE:", 1)[1])
        t = ev["trust"]
        lines = [
            f"Overall trust score {t.get('trust_score')}/100 (session risk {t.get('session_risk')}, "
            f"peak {t.get('peak_risk')}). {len(ev['alerts'])} alert(s) were raised."
        ]
        top = sorted(ev["module_means"].items(), key=lambda kv: -kv[1])[:3]
        if top:
            lines.append("Highest average module risks: " + ", ".join(f"{k} {v:.2f}" for k, v in top) + ".")
        for a in ev["alerts"][:6]:
            lines.append(f"- {a['t']:.1f}s [{a['severity']}] {a['message']}")
        lines.append("This summary was generated without an LLM; review flagged segments manually.")
        return "\n".join(lines)


def get_provider(name: str, model: str | None = None) -> LLMProvider:
    if name == "anthropic":
        return AnthropicProvider(model or "claude-opus-5-5")
    if name == "ollama":
        return OllamaProvider(model or "llama3.2")
    return TemplateProvider()


def build_evidence(
    trust: dict, alerts: list[dict], module_means: dict[str, float], answers: list[dict], duration_s: float
) -> dict[str, Any]:
    return {
        "duration_s": round(duration_s, 1),
        "trust": trust,
        "alerts": alerts[:25],
        "module_means": {k: round(v, 3) for k, v in module_means.items()},
        "answers": [
            {
                "t": round(a["t_start"], 1),
                "scripted_score": round(a.get("scripted_score", 0), 2),
                "ai_text_prob": a.get("ai_text_prob"),
                "excerpt": a.get("text", "")[:300],
            }
            for a in answers[:8]
        ],
    }


def summarize(
    evidence: dict[str, Any], provider: str = "anthropic", model: str | None = None
) -> tuple[str, str]:
    """Returns (summary, provider_used). Falls back to the template on any provider error."""
    prompt = "Summarise this interview's integrity evidence.\nEVIDENCE:" + json.dumps(evidence)
    for name in dict.fromkeys([provider, "template"]):
        try:
            p = get_provider(name, model if name == provider else None)
            return p.complete(SYSTEM, prompt), p.name
        except Exception as e:
            log.warning("LLM provider %s failed: %s", name, e)
    return "", "none"
