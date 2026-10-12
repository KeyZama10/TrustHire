"""Train / fine-tune voice anti-spoofing models on ASVspoof 2019 LA (+ optional extra data).

    python -m trusthire.audio.train model=ssl encoder=microsoft/wavlm-base-plus data.asvspoof_root=/data/LA
    python -m trusthire.audio.train model=rawnet2 data.asvspoof_root=/data/LA
    python -m trusthire.audio.train model=aasist init_pretrained=true data.asvspoof_root=/data/LA   # fine-tune

Codec (Opus/MP3/AAC), noise, telephone band-pass, reverb and gain augmentation (common/augment.py) make the
models robust to WebRTC conditions. Best checkpoint by dev EER; exported to ONNX for inference.
Kaggle/Colab: notebooks/train_audio_kaggle.ipynb. Configs: configs/audio/train.yaml.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from trusthire.audio.model import RawNet2, SSLClassifier, build_aasist, pad_or_tile
from trusthire.common.augment import random_audio_aug
from trusthire.common.media import load_audio
from trusthire.common.registry import ensure, models_dir
from trusthire.common.settings import REPO_ROOT


def asvspoof_protocol(root: Path, split: str) -> pd.DataFrame:
    """ASVspoof2019 LA CM protocol -> DataFrame(path, label) with label 1 = spoof."""
    proto = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }[split]
    rows = []
    for line in (root / "ASVspoof2019_LA_cm_protocols" / proto).read_text().splitlines():
        spk, utt, _, system, key = line.split()
        rows.append(
            {
                "path": str(root / f"ASVspoof2019_LA_{split}" / "flac" / f"{utt}.flac"),
                "label": int(key == "spoof"),
                "system": system,
                "speaker": spk,
            }
        )
    return pd.DataFrame(rows)


class WavDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int, augment: bool, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        x = sf.read(r.path, dtype="float32")[0] if r.path.endswith(".flac") else load_audio(r.path)
        if x.ndim > 1:
            x = x.mean(1)
        if self.augment:
            if len(x) > self.max_len:  # random crop instead of always the first 4 s
                s = self.rng.integers(0, len(x) - self.max_len)
                x = x[s : s + self.max_len]
            x = random_audio_aug(x, self.rng)
        return pad_or_tile(torch.from_numpy(x.astype(np.float32)), self.max_len), torch.tensor(int(r.label))


class SpoofModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, pos_weight: float):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        if cfg.model == "ssl":
            self.net: nn.Module = SSLClassifier(cfg.encoder)
        elif cfg.model in ("aasist", "aasist_l"):
            self.net = build_aasist(
                cfg.model, ensure(f"audio.{cfg.model}") if cfg.init_pretrained else None
            ).train()
        else:
            self.net = RawNet2()
        # index 1 = bona fide everywhere; weight the minority (bona fide) class
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))
        self.scores: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # type: ignore[no-any-return]

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss(self(x), 1 - y)  # target index 1 = bona fide
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        self.scores.append(torch.softmax(self(x).float(), -1)[:, 0].cpu())  # P(spoof)
        self.labels.append(y.cpu())

    def on_validation_epoch_end(self) -> None:
        from trusthire.evaluation.metrics import report

        r = report(torch.cat(self.labels).numpy(), torch.cat(self.scores).numpy())
        self.scores.clear()
        self.labels.clear()
        self.log_dict({"val/eer": r.eer, "val/auc": r.auc}, prog_bar=True)

    def configure_optimizers(self):
        lr = self.cfg.lr if self.cfg.model == "ssl" else 1e-4
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=lr, weight_decay=self.cfg.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, int(self.trainer.estimated_stepping_batches))
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def train(cfg: DictConfig) -> Path:
    L.seed_everything(cfg.seed)
    root = Path(cfg.data.asvspoof_root)
    tr, dev = asvspoof_protocol(root, "train"), asvspoof_protocol(root, "dev")
    if cfg.data.extra_csv:
        tr = pd.concat([tr, pd.read_csv(cfg.data.extra_csv)], ignore_index=True)
    pos_weight = float((tr.label == 1).sum() / max((tr.label == 0).sum(), 1))
    dl_tr = DataLoader(
        WavDataset(tr, cfg.max_len, cfg.augment, cfg.seed),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        drop_last=True,
        persistent_workers=cfg.workers > 0,
    )
    dl_dev = DataLoader(
        WavDataset(dev, cfg.max_len, False), batch_size=cfg.batch_size, num_workers=cfg.workers
    )
    mod = SpoofModule(cfg, pos_weight)
    ckpt = L.pytorch.callbacks.ModelCheckpoint(monitor="val/eer", mode="min", save_top_k=1)
    logger = (
        L.pytorch.loggers.MLFlowLogger(
            experiment_name="trusthire-audio", tracking_uri=f"file:{REPO_ROOT / 'mlruns'}"
        )
        if cfg.tracking == "mlflow"
        else L.pytorch.loggers.CSVLogger(str(REPO_ROOT / "logs"), name="audio")
    )
    L.Trainer(
        max_epochs=cfg.epochs,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        logger=logger,
        callbacks=[ckpt],
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    ).fit(mod, dl_tr, dl_dev)
    best = (
        SpoofModule.load_from_checkpoint(ckpt.best_model_path, cfg=cfg, pos_weight=pos_weight)
        if ckpt.best_model_path
        else mod
    )
    out = models_dir() / "audio" / f"{cfg.model}_trained.pth"
    torch.save(best.net.state_dict(), out)
    print("saved", out)
    return out


@hydra.main(config_path=str(REPO_ROOT / "configs" / "audio"), config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
