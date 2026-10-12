# Plan (as executed)

## Constraints discovered on the build machine

* Intel i5-1235U, Intel UHD graphics, 16 GB RAM, **no CUDA GPU**, no Docker, Windows 11 with **Smart App
  Control** (blocks some unsigned DLLs: `parselmouth`, mypy's compiled build), system Python 3.14.
* Consequences: everything runs on ONNX Runtime CPU with a GPU profile for later; Docker files are written but
  were not executed here; heavy training goes to Kaggle/Colab notebooks; venv and model caches live outside
  OneDrive (`~/.trusthire`).

## Model choices (baseline from the brief → what we ship, and why)

| Module | Brief | Shipped | Evidence |
|---|---|---|---|
| Voice spoof | wav2vec2 / AASIST / RawNet2 | **Spectra-0 (XLS-R + ECAPA), int8** | Measured In-the-Wild EER 1.25 % vs AASIST 42.5 %; int8 keeps accuracy at 1.7× speed |
| Video | EfficientNet-B4 / Xception | EffNet-B4 (DeepfakeBench FF++) default; Xception, ConvNeXt-V2, DINOv2, EVA-02 trainable | Only FF++-trained public weights available; backbone comparison runs once datasets are approved |
| Temporal | small transformer / BiLSTM | both implemented; top-half mean until trained | needs video datasets |
| AV sync | SyncNet / AV-HuBERT | SyncNet (ONNX) live; AV-HuBERT optional GPU | AV-HuBERT port is 1.9 GB + seq2seq code — infeasible on CPU; SyncNet reproduces reference exactly |
| Identity | ArcFace, ECAPA | ArcFace R50 (CPU) / R100 (GPU), ECAPA | measured on LFW / LibriSpeech |
| Behaviour | MediaPipe Face Mesh, YOLOv8 | MediaPipe Face Landmarker (blendshapes + pose matrix), **YOLO26n** | current SOTA versions |
| Transcript | Whisper | faster-whisper small int8 (CPU) / large-v3-turbo (GPU) | word timestamps + speed |
| Fusion | weighted + GBM / attention | all three + temperature/isotonic calibration | ablation harness |
| LLM summary | — | provider-agnostic: Anthropic (`claude-opus-5-5`) → Ollama → template | works offline |

## Datasets

See `docs/DATASETS.md`. Automatic: In-the-Wild, ASVspoof19 LA eval, LFW, LibriSpeech. Needs the user:
FF++, Celeb-DF, DFDC (forms / Kaggle), LAV-DF / DeepSpeak (Hugging Face gated), ASVspoof19 train (7.6 GB).

## Risks (and mitigations)

| Risk | Mitigation |
|---|---|
| No GPU → live latency | thread caps, FTZ, int8, staggered schedule (measured 9.5 s → 1.7 s/window) |
| No interview-shaped labelled data | data-collection kit + harness; synthetic data only for tests, quarantined |
| Domain shift of FF++ video models to real-time swaps | fine-tune path + DeepSpeak/LAV-DF; documented |
| Fusion overconfidence before training | mean-log-odds prior + red-flag path, unit-tested |
| Licences (InsightFace NC, Ultralytics AGPL) | documented in model cards |
