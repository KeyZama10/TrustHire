# Public datasets — access steps

| Dataset | Used for | Access | Command after download |
|---|---|---|---|
| **In-the-Wild** (Müller et al. 2022) | voice-spoof cross-domain eval | open (HF parquet mirror, Apache-2.0) — **automatic** | `python -m trusthire.audio.eval --datasets itw` |
| **ASVspoof 2019 LA** eval | voice-spoof eval | open (HF parquet mirror, ODC-By) — **automatic** | `--datasets asv19la` |
| ASVspoof 2019 LA train/dev | voice-spoof training | open: <https://datashare.ed.ac.uk/handle/10283/3336> (LA.zip, 7.6 GB) | `python -m trusthire.audio.train data.asvspoof_root=/path/LA` |
| ASVspoof 2021 LA/DF | eval | open (HF mirrors `SpeechAntiSpoofingBenchmarks/ASVspoof2021_*`) | `--datasets asv21la asv21df` |
| **FaceForensics++** (c23) | video training + in-domain test | **request form**: <https://github.com/ondyari/FaceForensics> → you receive `download-FaceForensics.py` | `python download-FaceForensics.py ROOT -d all -c c23 -t videos` then `python -m trusthire.video.prepare manifest --ffpp ROOT -o videos.csv` |
| **Celeb-DF v2** | cross-dataset test | **request form**: <https://github.com/yuezunli/celeb-deepfakeforensics> | `... prepare manifest --celebdf ROOT` |
| **DFDC** (subset) | cross-dataset test | Kaggle account + rules acceptance: `kaggle competitions download -c deepfake-detection-challenge` | `... prepare manifest --dfdc ROOT` |
| **LAV-DF** | AV + content-driven deepfakes | HF `ControlNet/LAV-DF` (gated, auto-approve after login) | `huggingface-cli download ControlNet/LAV-DF --repo-type dataset` → `--lavdf ROOT` |
| **DeepSpeak v1/v2** | webcam-style lip-sync & face-swap fakes — closest to interviews | HF `faridlab/deepspeak_v1` (gated, manual approval) | download, then `prepare manifest` (generic CSV `path,label,dataset,split`) |
| LFW | face verification eval | open (scikit-learn downloads it) — **automatic** | `python -m trusthire.identity.eval face` |
| LibriSpeech test-clean | speaker verification eval | open (OpenSLR) — **automatic** | `python -m trusthire.identity.eval speaker` |

Unofficial re-uploads of gated datasets exist on the Hub; we do not use them — please obtain gated data
through the official channels and respect their licences (most are research-only).

## Heavy training on Kaggle / Colab

`notebooks/train_video_kaggle.ipynb` and `notebooks/train_audio_kaggle.ipynb` clone this repo, install it with
uv, mount the dataset (Kaggle "Add data" or Drive) and run the same `trusthire.video.train` /
`trusthire.audio.train` entry points on a T4/P100/A100. Copy the resulting `*_trained.pth` (and
`temporal_*.pt`) into `~/.trusthire/models/video|audio/`; the live detector picks them up automatically.
