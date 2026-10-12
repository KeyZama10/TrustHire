# Architecture

```mermaid
flowchart LR
  subgraph Browser["Browser (React + TS)"]
    CAM[getUserMedia<br/>camera + mic] --> RTC[RTCPeerConnection]
    UI[Dashboard<br/>timeline · modules · alerts · transcript]
  end
  RTC -- WebRTC (SRTP) --> ING
  subgraph API["FastAPI (uvicorn)"]
    ING[aiortc ingest<br/>25 fps grid · 16 kHz mono] --> CH[Sliding chunker<br/>3 s windows · 1.5 s hop]
    CH --> Q[(per-session queue<br/>drops oldest if behind)]
    Q --> AN[SessionAnalyzer]
    WS[WebSocket /ws/sessions/:id]
    REST[REST /api/*]
  end
  subgraph Branches["Parallel branches (thread pool, ONNX Runtime)"]
    FD[SCRFD face detection<br/>5 shared anchors] --> VD[Video deepfake<br/>EffNet-B4 → temporal]
    FD --> FI[Face identity<br/>ArcFace vs enrollment]
    FD --> AV[AV sync<br/>SyncNet]
    AU[Voice spoof<br/>Spectra-0 int8] 
    VI[Voice identity<br/>ECAPA-TDNN]
    BH[Behaviour · rPPG · YOLO26]
    TR[Transcript async<br/>Whisper → features · AI-text · RAG+NLI]
  end
  AN --> FD & AU & VI & BH
  AN -. audio .-> TR
  VD & FI & AV & AU & VI & BH --> FU[Fusion<br/>weighted / GBM / attention<br/>+ calibration]
  TR -. late update .-> FU
  FU --> PT[FusedPoint<br/>risk · contributions · alerts]
  PT --> WS --> UI
  PT --> ST[(Session store<br/>JSON · frames · audio)]
  ST --> RP[Report builder<br/>Grad-CAM · saliency · SHAP · LLM summary → PDF]
  REST --> RP
  subgraph Async["Offline uploads"]
    UP[POST /api/uploads] --> CEL{Redis up?}
    CEL -- yes --> CW[Celery worker]
    CEL -- no --> LT[in-process thread]
    CW & LT --> OFF[analyze_file<br/>same analyzer]
  end
  OFF --> ST
```

## Key design decisions

| Decision | Why |
|---|---|
| **Every module emits `{score, confidence, t_start, t_end, details}`** | Uniform fusion input; confidence 0 = "no evidence" (no face, no speech), so missing modalities are first-class instead of imputed. |
| **Shared face detections** (5 anchor frames / window) | SCRFD is the most-shared cost; video, identity and SyncNet reuse it (SyncNet interpolates boxes between anchors). |
| **ONNX Runtime everywhere for inference**, providers TensorRT → CUDA → DirectML → CPU | One code path for laptop and GPU box; parity-checked exports; `set_denormal_as_zero` (measured 40× CPU speed-up). |
| **Per-session thread caps** (3 intra-op threads) | Branches run concurrently; uncapped sessions oversubscribed the CPU (9.5 s → 1.7 s per window). |
| **Staggered CPU schedule** | With 50 % overlap, running a module on every other window still tiles the timeline contiguously; heavy modules alternate even/odd windows. GPU profile runs everything every window. |
| **Asynchronous transcript branch** | ASR is slower than the hop on CPU; answers are analysed when an utterance ends (VAD) and *retroactively update* the overlapping timeline points. |
| **Fusion = confidence-weighted mean log-odds + red-flag term** (prior), learned GBM / attention when labelled data exists | Avoids naive-Bayes overconfidence from correlated "looks genuine" modules while letting one strong signal (e.g. face mismatch) dominate. |
| **Calibration** (temperature for logit models, isotonic for GBM) fit on held-out participants | Risk values are meant to be read by people; ECE is reported. |
| **Explanations computed on demand** | Grad-CAM (PyTorch) and occlusion saliency only for flagged evidence when a report is requested, keeping the live path ONNX-only. |

## Code map

```
src/trusthire/
  common/      types, settings, registry (weights), ORT factory, face (SCRFD+ArcFace), media, augment, export
  video/       model (EffNet-B4/Xception/timm + temporal heads), infer, prepare, train, eval
  audio/       model (AASIST/RawNet2/SSL/Spectra-0), infer (+int8), datasets, train, eval
  avsync/      SyncNet towers, infer, eval (desync study)
  identity/    ArcFace + ECAPA monitors, speaker encoder, eval (LFW / LibriSpeech)
  behavior/    MediaPipe gaze/pose/blinks, reading sweeps, rPPG (POS/CHROM), YOLO26
  transcript/  faster-whisper ASR, features, AI-text + RAG/NLI, async monitor, train
  fusion/      features, models (weighted / GBM / attention + calibration), live fuser
  explain/     Grad-CAM, audio saliency, plots, LLM providers, PDF report
  pipeline/    chunker, analyzer, offline runner, session store
  api/         FastAPI app, WebRTC ingest
  worker/      Celery app, job runner (with in-process fallback)
  evaluation/  metrics, dataset schema, harness (ablations/fairness), latency, synthetic (tests only), report
  third_party/ vendored AASIST (MIT), SyncNet (MIT), DeepfakeBench Xception (CC BY-NC)
```
