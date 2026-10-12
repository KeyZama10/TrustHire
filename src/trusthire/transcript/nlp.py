"""Text models: AI-generated-text detector and resume RAG consistency (embeddings + FAISS + NLI)."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trusthire.common.registry import models_dir

AI_TEXT_MODEL = "fakespot-ai/roberta-base-ai-text-detection-v1"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


def _hf_cache() -> str:
    return str(models_dir() / "hf")


class AITextDetector:
    _inst: AITextDetector | None = None
    _lock = threading.Lock()

    def __init__(self, model_id: str = AI_TEXT_MODEL):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=_hf_cache())
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, cache_dir=_hf_cache()
        ).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        # resolve which logit means "AI": label names vary between checkpoints
        ai_keys = [
            k for k in labels if any(s in k for s in ("ai", "fake", "machine", "generated", "label_1"))
        ]
        self.ai_idx = labels[ai_keys[0]] if ai_keys else 1
        self._run = threading.Lock()

    @classmethod
    def get(cls) -> AITextDetector:
        with cls._lock:
            if cls._inst is None:
                cls._inst = AITextDetector()
            return cls._inst

    def prob_ai(self, text: str) -> float | None:
        import torch

        if len(text.split()) < 25:  # too short to judge
            return None
        with self._run, torch.inference_mode():
            enc = self.tok(text, truncation=True, max_length=512, return_tensors="pt")
            p = torch.softmax(self.model(**enc).logits, -1)[0, self.ai_idx]
        return float(p)


class _Shared:
    """Lazy singletons for the embedder and NLI cross-encoder."""

    lock = threading.Lock()
    embedder = None
    nli = None

    @classmethod
    def get_embedder(cls):  # type: ignore[no-untyped-def]
        with cls.lock:
            if cls.embedder is None:
                from sentence_transformers import SentenceTransformer

                cls.embedder = SentenceTransformer(EMBED_MODEL, cache_folder=_hf_cache(), device="cpu")
            return cls.embedder

    @classmethod
    def get_nli(cls):  # type: ignore[no-untyped-def]
        with cls.lock:
            if cls.nli is None:
                from sentence_transformers import CrossEncoder

                cls.nli = CrossEncoder(NLI_MODEL, cache_folder=_hf_cache(), device="cpu")
            return cls.nli


def read_resume(path: str | Path) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
    return p.read_text(encoding="utf-8", errors="ignore")


def chunk_resume(text: str, max_words: int = 40) -> list[str]:
    lines = [re.sub(r"\s+", " ", ln).strip(" •-*\t") for ln in re.split(r"[\n\r]+|(?<=[.;])\s+", text)]
    chunks: list[str] = []
    cur: list[str] = []
    for ln in lines:
        if not ln:
            continue
        if len(ln.split()) >= 4:  # bullets/sentences become their own chunk
            if cur:
                chunks.append(" ".join(cur))
                cur = []
            chunks.append(" ".join(ln.split()[:max_words]))
        else:  # short header fragments get merged
            cur.append(ln)
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c) > 3]


@dataclass
class ClaimCheck:
    claim: str
    evidence: str
    similarity: float
    contradiction: float
    entailment: float


@dataclass
class ConsistencyResult:
    score: float  # risk: contradiction with the resume
    support: float
    relevance: float
    checks: list[ClaimCheck] = field(default_factory=list)


class ResumeIndex:
    def __init__(self, text: str):
        import faiss

        self.chunks = chunk_resume(text)
        if not self.chunks:
            raise ValueError("resume has no extractable text")
        emb = _Shared.get_embedder().encode(self.chunks, normalize_embeddings=True, batch_size=32)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.asarray(emb, np.float32))

    @classmethod
    def from_file(cls, path: str | Path) -> ResumeIndex:
        return cls(read_resume(path))

    def check(self, answer: str, k: int = 3, rel_thr: float = 0.55) -> ConsistencyResult | None:
        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.split()) >= 6]
        if not claims:
            return None
        q = _Shared.get_embedder().encode(claims, normalize_embeddings=True)
        sims, idx = self.index.search(np.asarray(q, np.float32), k)
        pairs, meta = [], []
        for ci, claim in enumerate(claims):
            for s, j in zip(sims[ci], idx[ci], strict=True):
                if j >= 0 and s >= rel_thr:
                    pairs.append((self.chunks[j], claim))
                    meta.append((claim, self.chunks[j], float(s)))
        if not pairs:
            return ConsistencyResult(score=0.0, support=0.0, relevance=float(sims.max()), checks=[])
        logits = np.asarray(_Shared.get_nli().predict(pairs, apply_softmax=True))
        # cross-encoder/nli-deberta-v3-* label order: contradiction, entailment, neutral
        checks = [
            ClaimCheck(c, e, s, float(p[0]), float(p[1])) for (c, e, s), p in zip(meta, logits, strict=True)
        ]
        contra = max(c.contradiction * c.similarity for c in checks)
        support = max(c.entailment for c in checks)
        return ConsistencyResult(
            score=float(contra),
            support=float(support),
            relevance=float(sims.max()),
            checks=sorted(checks, key=lambda c: -c.contradiction)[:5],
        )
