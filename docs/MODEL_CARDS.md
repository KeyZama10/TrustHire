# Model cards

All weights are downloaded to `~/.trusthire/models` by `trusthire download-models` (or on first use).
Numbers below marked **measured** come from `results/` in this repository; everything else is the upstream
authors' claim and is labelled as such.

| Module | Model | Source / licence | Trained on | Input | Output → TrustHire score |
|---|---|---|---|---|---|
| Face detection | SCRFD-10GF (`det_10g.onnx`) | InsightFace buffalo_l — **non-commercial research only** | WIDER FACE | BGR frame, 320 px (CPU) / 640 px (GPU) | boxes + 5 landmarks |
| Face identity | ArcFace R50 (`w600k_r50.onnx`); R100 Glint360K (antelopev2) on GPU | InsightFace — non-commercial research | WebFace600K / Glint360K | 112×112 aligned face | cosine to enrollment; risk = σ((0.30 − sim)/0.06), plus sudden-change term |
| Video deepfake | EfficientNet-B4 (default), Xception | DeepfakeBench v1.0.1 — CC BY-NC 4.0 | FF++ c23 (all 4 methods) | 256×256 face crop (1.3× box), mean/std 0.5 | P(fake), top-half mean over 3 frames/window (or trained temporal head) |
| Video (trainable) | ConvNeXt-V2, DINOv2 ViT-S/14, EVA-02 via timm | Apache-2.0 / timm | ImageNet / SSL init → fine-tune with `video.train` | 224 px | same |
| Voice spoof (default) | **Spectra-0** XLS-R-300M + ECAPA head, **int8 dynamic-quantised by us** | lab260/spectra_0 — Apache-2.0 | upstream: multiple anti-spoofing corpora (see card) | 64 600 samples pre-emphasised | softmax spoof prob |
| Voice spoof (baseline) | AASIST, AASIST-L | clovaai/aasist — MIT | ASVspoof 2019 LA train | 64 600 raw samples | softmax spoof prob |
| AV sync | SyncNet v2 | joonson/syncnet_python — MIT | LRS2 / BBC | 5×224² BGR face stacks + 13×20 MFCC | offset (frames) and confidence → risk prior |
| Voice identity | ECAPA-TDNN | speechbrain/spkrec-ecapa-voxceleb — Apache-2.0 | VoxCeleb 1+2 | 16 kHz waveform (4 s) | cosine to voiceprint |
| Landmarks | MediaPipe Face Landmarker (478 pts + 52 blendshapes) | Google — Apache-2.0 | — | RGB frame | gaze, head pose, blinks |
| Objects | YOLO26n | Ultralytics — **AGPL-3.0** | COCO | 480 px frame | person/phone/tv/laptop/book |
| ASR | faster-whisper `small` int8 (CPU), `large-v3-turbo` fp16 (GPU) | MIT | Whisper data | 16 kHz | words + timestamps |
| AI-text | RoBERTa-base AI-text detector | fakespot-ai — Apache-2.0 | written text | transcript | P(AI-written) — weak on speech, low weight |
| Embeddings | bge-small-en-v1.5 | BAAI — MIT | — | sentences | 384-d |
| NLI | nli-deberta-v3-small | cross-encoder — Apache-2.0 | SNLI + MNLI | (resume chunk, claim) | contradiction prob |

## Measured component performance (this repo)

* Voice anti-spoofing, In-the-Wild subset (n=400, stratified): **Spectra-0 EER 1.25 %, int8 1.25 %**, AASIST
  42.5 %, AASIST-L 43.5 % — see `results/audio/benchmark.md`, including degradations.
* AV sync on the SyncNet reference clip: offset 3 / confidence 9.8 vs. reference 3 / 10.0; injected shifts of
  ±k frames are recovered exactly (unit test `test_syncnet_tracks_audio_shift`).
* Face verification on LFW: `results/identity/face_lfw.json`.
* Latency: `results/latency/latency_cpu.json`.

## Intended use & misuse

Decision support for human interviewers. Not suitable as the sole basis for rejecting a candidate. Do not
use to surveil people without consent; comply with local employment and biometric-privacy law (e.g. GDPR
Art. 9, BIPA). Several weights are **non-commercial** (InsightFace, DeepfakeBench) or **AGPL** (Ultralytics):
replace them before any commercial deployment.
