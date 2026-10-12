# TrustHire: Multimodal Detection of Fraudulent Remote Job Interviews

*Auto-generated from `results/` on {{GENERATED}} by `python -m trusthire.evaluation.report`. Do not edit
`paper.md` by hand — edit `paper_template.md`.*

## Abstract

Remote hiring has created new attack surfaces: real-time face swaps, cloned voices, proxy candidates and
answers read from AI-generated scripts. We present TrustHire, a streaming system that analyses a live
interview in 3-second windows (50 % overlap) with ten evidence modules — face-forgery detection,
voice anti-spoofing, audio-visual synchrony, face and voice identity continuity, gaze/second-screen
behaviour, rPPG liveness, object detection, scripted-answer analysis and resume consistency — and fuses them
into a calibrated per-window risk timeline and a session trust score with visual and textual explanations.
We report component benchmarks on public datasets, an ablation over modality groups on our own
interview-shaped test set, latency on commodity hardware, and a fairness/robustness analysis.

## 1 Introduction

Public deepfake benchmarks (FaceForensics++, Celeb-DF, ASVspoof) contain short, studio-like clips; real
interview fraud is multimodal, interactive and frequently low-tech (proxies, reading). Our contributions:
(i) an open, real-time multimodal pipeline with per-timestamp evidence; (ii) a confidence-aware fusion layer
that tolerates missing modalities and is calibrated; (iii) an interview-shaped test-set protocol and harness;
(iv) a measured accuracy/latency study on CPU-only hardware.

## 2 System

See `docs/ARCHITECTURE.md`. Browser WebRTC → aiortc ingest (25 fps, 16 kHz) → sliding-window chunker →
parallel branches → fusion → WebSocket dashboard + PDF report. Heavy modules on CPU are scheduled on
alternating windows; because windows overlap by 50 %, alternate windows still tile the timeline contiguously.

| Module | Model (deployed) | Baseline from the brief |
|---|---|---|
| Video deepfake | EfficientNet-B4 (DeepfakeBench, FF++ c23) + temporal head | EfficientNet-B4 / Xception |
| Voice spoofing | Spectra-0 (XLS-R 300M + ECAPA), int8 ONNX | AASIST / RawNet2 / wav2vec2 |
| AV sync | SyncNet (offset ± 7 frames, confidence) | SyncNet / AV-HuBERT |
| Face identity | SCRFD-10G + ArcFace R50 (WebFace600K) | InsightFace ArcFace |
| Voice identity | ECAPA-TDNN (VoxCeleb) | ECAPA-TDNN |
| Behaviour | MediaPipe Face Landmarker (gaze, pose, blinks), reading-sweep detector | MediaPipe Face Mesh |
| Liveness | rPPG POS + CHROM spectral SNR | rPPG |
| Objects | YOLO26n (COCO) | YOLOv8 |
| Transcript | faster-whisper (small int8 CPU / large-v3-turbo GPU), fluency + prosody features, RoBERTa AI-text detector | Whisper |
| Resume | bge-small + FAISS retrieval, DeBERTa-v3 NLI contradiction | RAG |
| Fusion | confidence-weighted log-odds pooling / LightGBM stacker / cross-modal attention + calibration | late fusion + GBM |

## 3 Component evaluation

### 3.1 Voice anti-spoofing (public data, real runs)

{{AUDIO}}

### 3.2 Video deepfake detection (cross-dataset)

{{VIDEO}}

### 3.3 Identity verification

{{IDENTITY}}

### 3.4 Audio-visual synchrony

{{AVSYNC}}

## 4 Interview test set: ablation study

{{ABLATION}}

### 4.1 Detection by attack type (threshold at 5 % FPR on genuine windows)

{{PER_ATTACK}}

## 5 Latency and resources

{{LATENCY}}

## 6 Fairness and robustness

Robustness to audio degradations is reported in §3.1. Breakdown of the fused detector by recording
condition and demographic factors on the interview test set:

{{FAIRNESS}}

## 7 Limitations

See `docs/LIMITATIONS.md` (reproduced in the final report): dataset size and diversity, consent-bound
attack generation, threshold transfer across webcams/codecs, rPPG sensitivity to compression, AI-text
detectors on spoken transcripts, accent effects on ASR-derived features, and the risk of automation bias —
TrustHire produces evidence for human reviewers, not verdicts.

## 8 Ethics

All recordings in the custom set require written informed consent from every participant whose face or
voice is captured or synthesised (template in `docs/DATA_COLLECTION.md`). Deepfakes are generated only of
consenting participants and never distributed.

## References

Rössler et al., FaceForensics++, ICCV 2019 · Li et al., Celeb-DF, CVPR 2020 · Yan et al., DeepfakeBench,
NeurIPS 2023 · Jung et al., AASIST, ICASSP 2022 · Müller et al., Does Audio Deepfake Detection Generalize?,
Interspeech 2022 · Chung & Zisserman, Out of time: automated lip sync in the wild, 2016 · Deng et al.,
ArcFace, CVPR 2019 · Desplanques et al., ECAPA-TDNN, Interspeech 2020 · Wang et al., POS rPPG, IEEE TBME
2017 · de Haan & Jeanne, CHROM, IEEE TBME 2013 · Radford et al., Whisper, 2022 · Guo et al., On Calibration
of Modern Neural Networks, ICML 2017 · Selvaraju et al., Grad-CAM, ICCV 2017 · Lundberg & Lee, SHAP, NeurIPS 2017.
