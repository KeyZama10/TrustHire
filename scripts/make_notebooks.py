"""Generate the notebooks in notebooks/ (run with the project venv: python scripts/make_notebooks.py [--execute])."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
REPO = "https://github.com/Navyab27/TrustHire.git"  

SETUP = f"""# --- environment (Kaggle: enable GPU T4x2 / P100; Colab: Runtime > GPU) ---
!pip -q install uv
!git clone {REPO} trusthire || true
%cd trusthire
!uv pip install --system -e . 2>&1 | tail -2
!python -m trusthire.cli download-models --profile gpu"""


def md(s: str):  # type: ignore[no-untyped-def]
    return nbf.v4.new_markdown_cell(s)


def code(s: str):  # type: ignore[no-untyped-def]
    return nbf.v4.new_code_cell(s)


def video_nb() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# TrustHire — train video deepfake detectors on Kaggle/Colab\n\n"
            "Train on **FaceForensics++ c23**, test in-domain and **cross-dataset on Celeb-DF v2** (the standard "
            "generalisation protocol). Add the datasets via Kaggle *Add data* (your own private upload of the "
            "officially obtained data) or mount Drive. Nothing here downloads gated data on your behalf."
        ),
        code(SETUP),
        code(
            "FFPP = '/kaggle/input/ffpp-c23'      # official FaceForensics++ layout\n"
            "CELEB = '/kaggle/input/celeb-df-v2'  # Celeb-DF v2 layout\n"
            "!python -m trusthire.video.prepare manifest --ffpp $FFPP --celebdf $CELEB -o videos.csv\n"
            "!python -m trusthire.video.prepare crops videos.csv -o /kaggle/working/faces --frames 32"
        ),
        md("## 1. Baselines (no training): DeepfakeBench EfficientNet-B4 / Xception"),
        code(
            "!python -m trusthire.video.eval --crops /kaggle/working/faces/crops.csv --backbones effnb4 xception "
            "--test-datasets ffpp celebdf --conditions clean jpeg_q30 lowres_x3"
        ),
        md(
            "## 2. Modern backbones, fine-tuned with heavy augmentation\n"
            "ConvNeXt-V2 and DINOv2 are candidates to beat the FF++-only baselines on Celeb-DF; keep whichever wins."
        ),
        code(
            "for bb in ['convnextv2_tiny.fcmae_ft_in22k_in1k', 'vit_small_patch14_dinov2.lvd142m']:\n"
            "    !python -m trusthire.video.train stage=frame backbone={bb} data.crops=/kaggle/working/faces/crops.csv "
            "epochs=8 batch_size=64 compile=true\n"
            "!python -m trusthire.video.eval --crops /kaggle/working/faces/crops.csv "
            "--backbones convnextv2_tiny.fcmae_ft_in22k_in1k vit_small_patch14_dinov2.lvd142m --test-datasets ffpp celebdf"
        ),
        md("## 3. Temporal heads (BiLSTM vs transformer) on the winning backbone"),
        code(
            "for t in ['bilstm', 'transformer']:\n"
            "    !python -m trusthire.video.train stage=temporal backbone=effnb4 temporal={t} data.crops=/kaggle/working/faces/crops.csv"
        ),
        md(
            "## 4. Export\nDownload `~/.trusthire/models/video/*_trained.pth`, `temporal_*.pt` and "
            "`results/video/benchmark.*` and copy them into the same paths on the demo machine."
        ),
        code("!ls -la ~/.trusthire/models/video results/video && cat results/video/benchmark.md"),
    ]
    return nb


def audio_nb() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# TrustHire — train / fine-tune voice anti-spoofing on Kaggle/Colab\n\n"
            "ASVspoof 2019 LA is openly downloadable (Edinburgh DataShare, ODC-By). Compare fine-tuned SSL "
            "front-ends (WavLM / XLS-R) with AASIST and RawNet2; pick the best by **In-the-Wild EER** (cross-domain)."
        ),
        code(SETUP),
        code(
            "!wget -q https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip && unzip -q LA.zip -d /kaggle/working/"
        ),
        code(
            "for m in ['rawnet2', 'aasist']:\n"
            "    !python -m trusthire.audio.train model={m} data.asvspoof_root=/kaggle/working/LA epochs=20 batch_size=32\n"
            "!python -m trusthire.audio.train model=ssl encoder=microsoft/wavlm-base-plus data.asvspoof_root=/kaggle/working/LA "
            "epochs=6 batch_size=16\n"
            "!python -m trusthire.audio.train model=ssl encoder=facebook/wav2vec2-xls-r-300m data.asvspoof_root=/kaggle/working/LA "
            "epochs=4 batch_size=8"
        ),
        md("## Evaluate against the shipped models (In-the-Wild + ASVspoof19 LA eval, with degradations)"),
        code(
            "!python -m trusthire.audio.eval --datasets itw asv19la --n 1000 --backends spectra0 aasist "
            "--conditions clean opus_16k noise_snr10 telephone"
        ),
    ]
    return nb


def results_nb() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# TrustHire — results\n\nEvery number below is loaded from `results/` (real runs). "
            "Synthetic smoke-test outputs (`results/SYNTHETIC_DEMO`) are deliberately excluded."
        ),
        code(
            "import json, pathlib\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
            "R = pathlib.Path('..') / 'results'\nload = lambda p: json.loads((R / p).read_text()) if (R / p).exists() else None"
        ),
        md("## Voice anti-spoofing (public data)"),
        code(
            "b = load('audio/benchmark.json')\n"
            "df = pd.DataFrame(b.values())[['dataset','condition','backend','n','auc','eer','tpr_at_fpr1','ece']]\n"
            "df['eer_%'] = (100*df.eer).round(2); df.sort_values(['dataset','condition','eer'])"
        ),
        code(
            "rob = df[df.dataset=='itw'].pivot(index='condition', columns='backend', values='eer_%')\n"
            "ax = rob.plot.bar(figsize=(8,3.5), logy=True, rot=20); ax.set_ylabel('EER % (log)'); "
            "ax.set_title('In-the-Wild: robustness to degradations'); plt.tight_layout()"
        ),
        md("## Identity"),
        code(
            "f = load('identity/face_lfw.json'); s = load('identity/speaker_librispeech.json')\n"
            "pd.DataFrame([{k: d.get(k) for k in ['benchmark','n','auc','eer','eer_threshold_sim']} for d in (f, s) if d])"
        ),
        md("## AV sync desync study"),
        code("a = load('avsync/desync_study.json')\npd.DataFrame(a['per_shift']).T if a else 'pending'"),
        md("## Latency (per 3 s window)"),
        code(
            "l = load('latency/latency_cpu.json')\n"
            "pd.Series(l['modules_ms_p50']).sort_values().plot.barh(figsize=(6,3), title=f\"CPU window p50 {l['total_ms_p50']:.0f} ms\"); plt.tight_layout()"
        ),
        md("## Custom interview test set (ablations)"),
        code(
            "c = load('custom/results.json')\n"
            "if c and not c['synthetic']:\n"
            "    rows = [{'config': v['config'], 'fuser': v['fuser'], **{k: v['window'][k] for k in ('auc','eer','tpr_at_fpr5','ece')}} "
            "for v in c['ablation'].values()]\n"
            "    display(pd.DataFrame(rows).sort_values('auc', ascending=False))\n"
            "else:\n    print('pending: record the custom set (docs/DATA_COLLECTION.md) and run `make eval-custom`')"
        ),
    ]
    return nb


if __name__ == "__main__":
    NB.mkdir(exist_ok=True)
    for name, nb in (
        ("train_video_kaggle", video_nb()),
        ("train_audio_kaggle", audio_nb()),
        ("results", results_nb()),
    ):
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB / f"{name}.ipynb"
        if name == "results" and "--execute" in sys.argv:
            from nbclient import NotebookClient

            NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(NB)}}).execute()
        nbf.write(nb, path)
        print("wrote", path)
