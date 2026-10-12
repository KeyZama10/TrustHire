# TrustHire: Multimodal Detection of Fraudulent Remote Job Interviews

*Auto-generated from `results/` on 2026-10-12 13:56 UTC by `python -m trusthire.evaluation.report`. Do not edit
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

| Model | Test set | n | AUC [95% CI] | EER % [95% CI] | TPR@1%FPR | CPU ms/utt |
|---|---|---|---|---|---|---|
| spectra0 | asv19la | 308 | 1.0000 [1.000, 1.000] | 0.00 [0.0, 0.0] | 1.000 | 3327 |
| spectra0_int8 | asv19la | 308 | 1.0000 [1.000, 1.000] | 0.00 [0.0, 0.0] | 1.000 | 1803 |
| aasist | asv19la | 308 | 0.9983 [0.996, 1.000] | 1.62 [0.6, 4.5] | 0.942 | 501 |
| aasist_l | asv19la | 308 | 0.9992 [0.998, 1.000] | 2.27 [0.0, 3.9] | 0.981 | 395 |
| spectra0 | itw | 400 | 0.9978 [0.993, 1.000] | 1.25 [0.5, 3.0] | 0.980 | 3327 |
| spectra0_int8 | itw | 400 | 0.9977 [0.993, 1.000] | 1.25 [0.5, 3.0] | 0.980 | 1803 |
| aasist | itw | 400 | 0.5795 [0.529, 0.633] | 42.50 [38.0, 47.9] | 0.020 | 501 |
| aasist_l | itw | 400 | 0.6175 [0.567, 0.669] | 43.50 [38.0, 47.5] | 0.050 | 395 |

*CPU ms/utt measured 2026-10-12 19:25 on battery power (CPU capped at 1300 MHz) — about 3x slower than on mains power; re-measure plugged in with `python -m trusthire.audio.eval --measure-latency`.*

**Robustness (In-the-Wild subset, same utterances under each degradation):**

| Condition | aasist EER % | spectra0_int8 EER % | spectra0_int8+wpe EER % |
|---|---|---|---|
| clean | 42.50 | 1.25 | – |
| mp3_32k | 41.75 | 2.25 | – |
| noise_snr10 | 57.50 | 7.50 | – |
| noise_snr20_pink | 48.50 | 1.25 | – |
| opus_16k | 47.00 | 3.00 | – |
| reverb_0.25 | – | 21.00 | – |
| reverb_0.4 | 53.00 | 27.50 | 26.50 |
| reverb_0.7 | – | 34.25 | – |
| telephone | 53.00 | 2.00 | – |

### 3.2 Video deepfake detection (cross-dataset)

> *Pending — FF++ / Celeb-DF / DFDC video benchmark (datasets need access approval, see docs/DATASETS.md). Run `python -m trusthire.video.eval --crops <crops.csv> --test-datasets ffpp celebdf`; this section is filled automatically from the results.*

### 3.3 Identity verification

**Face (LFW test, 1000 pairs, SCRFD→ArcFace R50):** AUC 0.9860, EER 3.00% (threshold cos 0.121), accuracy 96.90%. At the deployed threshold cos=0.30: same-person false flags 3.60%, impostors missed 0.00%.

**Speaker (LibriSpeech test-clean, 2000 trials, 4 s, ECAPA-TDNN):** AUC 0.9983, EER 1.80%. At the deployed threshold cos=0.25: same-speaker false flags 1.10%, impostors missed 3.40%.

### 3.4 Audio-visual synchrony

Videos: 1; windows: 63.

| Audio shift (frames) | Offset accuracy (±1 frame) | Mean SyncNet conf | Mean risk |
|---|---|---|---|
| -10 (-400 ms) | 100.0% | 5.99 | 1.000 |
| -6 (-240 ms) | 100.0% | 9.22 | 0.750 |
| -4 (-160 ms) | 100.0% | 9.01 | 0.250 |
| -2 (-80 ms) | 100.0% | 8.75 | 0.000 |
| 0 (0 ms) | 100.0% | 8.81 | 0.000 |
| 2 (80 ms) | 100.0% | 8.78 | 0.000 |
| 4 (160 ms) | 100.0% | 8.74 | 0.250 |
| 6 (240 ms) | 100.0% | 8.80 | 0.750 |
| 10 (400 ms) | 100.0% | 8.84 | 1.000 |

Detecting |shift| ≥ 4 frames (160 ms) vs in-sync: AUC 1.000, EER 0.0%.

## 4 Interview test set: ablation study

> *Pending — custom interview test set (30–50 recorded sessions; protocol in docs/DATA_COLLECTION.md). Run `make eval-custom`; this section is filled automatically from the results.*

### 4.1 Detection by attack type (threshold at 5 % FPR on genuine windows)

> *Pending — custom interview test set (30–50 recorded sessions; protocol in docs/DATA_COLLECTION.md). Run `make eval-custom`; this section is filled automatically from the results.*

## 5 Latency and resources

**CPU** (Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, 12 threads): window p50 **1689 ms**, p95 2970 ms for a 3 s window (hop 1500 ms; real-time factor 1.13); peak RSS 2084 MB. Per-module p50 (ms, concurrent): voice_identity 2203, video_deepfake 1467, av_sync 1383, voice_spoof 1170, behavior 371, face_identity 311, objects 241.

> *Pending — GPU latency. Run `python -m trusthire.evaluation.latency (TRUSTHIRE_PROFILE=gpu)`; this section is filled automatically from the results.*

## 6 Fairness and robustness

Robustness to audio degradations is reported in §3.1. Breakdown of the fused detector by recording
condition and demographic factors on the interview test set:

> *Pending — custom interview test set (30–50 recorded sessions; protocol in docs/DATA_COLLECTION.md). Run `make eval-custom`; this section is filled automatically from the results.*

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
