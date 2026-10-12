# Phase tracker

Legend: ✅ done & verified here · 🟡 implemented, needs data/GPU you must provide · ⏳ not started

| Phase (from the brief) | Scope | Status | Evidence |
|---|---|---|---|
| 0 Scaffold | uv/Py3.12, settings, registry, ORT factory (FTZ), face stack, media, chunker, metrics, tests, ruff/mypy/pre-commit | ✅ | commit "Phase 0"; `tests/test_core.py` |
| 1–3 Audio + video detectors | Video: EffNet-B4 / Xception (FF++ weights), timm backbones, BiLSTM/transformer heads, ONNX export, train/eval/prepare. Audio: AASIST, AASIST-L, RawNet2, SSL classifier, Spectra-0 (+int8), train/eval. AV sync: SyncNet ONNX + desync study | ✅ inference & audio benchmarks · 🟡 video benchmarks + training (datasets need approval; GPU) | `results/audio/benchmark.md`; `tests/test_models.py` (SyncNet reproduces reference) |
| 4–5 Identity, behaviour, transcript | ArcFace + ECAPA continuity, MediaPipe gaze/pose/blinks, reading sweeps, rPPG POS/CHROM, YOLO26, faster-whisper, fluency/prosody features, AI-text, resume RAG + NLI | ✅ | `results/identity/face_lfw.json`; `tests/test_signals.py` (rPPG recovers HR, reading detector) |
| 6–7 Fusion, trust score, explainability | weighted / LightGBM-XGBoost / cross-modal attention fusers, temperature/isotonic calibration, missing modalities, trust score; Grad-CAM, audio occlusion saliency, SHAP/attention attributions, LLM summary (Anthropic/Ollama/template), PDF | ✅ code & tests · 🟡 learned fusers need the custom set | `tests/test_fusion_api.py`; example report generated end-to-end |
| 8–9 Live demo, dashboard, test set, ablations | aiortc WebRTC ingest, FastAPI + WebSocket, Celery (+ local fallback), React dashboard (enrollment, live, uploads, history), data-collection kit, harness (ablations, per-attack, fairness, calibration, plots, LaTeX) | ✅ app verified in browser (upload → results → report) · 🟡 live camera test on your machine · 🟡 recordings | `results/latency/latency_cpu.md`; harness smoke run on synthetic data (quarantined in `results/SYNTHETIC_DEMO`) |
| 10 Report & polish | paper generator from results, results notebook, model cards, architecture, limitations, Docker, Makefile/make.cmd | ✅ · 🟡 Docker images not built here (no Docker on this machine) | `docs/report/paper.md` |

## What I need from you to finish the 🟡 items

1. **Dataset access** — FF++ and Celeb-DF v2 request forms; a Kaggle account for DFDC; a Hugging Face token
   after accepting LAV-DF (auto) and DeepSpeak (manual) terms. Then run `notebooks/train_video_kaggle.ipynb`.
2. **GPU** — Kaggle/Colab is enough for training; a GPU box (or `--profile gpu` compose) for the < 1.5 s target
   with every module on every window.
3. **Custom recordings** — 30–50 sessions following `docs/DATA_COLLECTION.md` (with participant consent).
4. **Optional** — `ANTHROPIC_API_KEY` for LLM-written report summaries.
