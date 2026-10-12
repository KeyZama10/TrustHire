"""Train frame-level deepfake classifiers and temporal heads (Lightning + Hydra, AMP, torch.compile).

    # stage 1: frame classifier (train on FF++, validate on FF++ val)
    python -m trusthire.video.train stage=frame backbone=convnextv2_tiny.fcmae_ft_in22k_in1k data.crops=.../crops.csv
    # stage 2: temporal head over frozen per-frame embeddings
    python -m trusthire.video.train stage=temporal backbone=effnb4 temporal=transformer data.crops=.../crops.csv

Heavy augmentation (JPEG 30-95, blur, down/up-scaling, gamma, hue/saturation, flips) targets cross-dataset
generalisation (train FF++ -> test Celeb-DF). Configs: configs/video/train.yaml. Kaggle: notebooks/train_video_kaggle.ipynb
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from trusthire.common.augment import random_image_aug
from trusthire.common.registry import ensure, models_dir
from trusthire.common.settings import REPO_ROOT
from trusthire.video.infer import preprocess
from trusthire.video.model import INPUT_SIZE, build_frame_model, build_temporal_head


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


class CropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int, augment: bool, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = _read(r.crop)
        if self.augment:
            img = random_image_aug(img, self.rng)
        return torch.from_numpy(preprocess([img], self.size)[0]), torch.tensor(float(r.label))


class FrameModule(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        w = (
            ensure(f"video.{cfg.backbone}")
            if cfg.backbone in ("effnb4", "xception") and cfg.init_from_dfb
            else None
        )
        self.model = build_frame_model(cfg.backbone, w, pretrained_timm=True)
        if cfg.compile:
            self.model = torch.compile(self.model)  # type: ignore[assignment]
        self.loss = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.val_scores: list[torch.Tensor] = []
        self.val_labels: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[0]

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss(self(x), y.long())
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        self.val_scores.append(torch.softmax(self(x), -1)[:, 1].float().cpu())
        self.val_labels.append(y.cpu())

    def on_validation_epoch_end(self) -> None:
        from trusthire.evaluation.metrics import report

        s, y = torch.cat(self.val_scores).numpy(), torch.cat(self.val_labels).numpy()
        self.val_scores.clear()
        self.val_labels.clear()
        if len(np.unique(y)) == 2:
            r = report(y, s)
            self.log_dict({"val/auc": r.auc, "val/eer": r.eer}, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.cfg.lr, total_steps=int(self.trainer.estimated_stepping_batches), pct_start=0.1
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def _split(df: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = df[(df.dataset.isin(cfg.data.train_datasets)) & (df.split == "train")]
    va = df[(df.dataset.isin(cfg.data.train_datasets)) & (df.split.isin(["val", "test"]))]
    # balance real/fake by subsampling the majority class per epoch (FF++ is 1:5 real:fake)
    n = min((tr.label == 0).sum(), (tr.label == 1).sum()) * cfg.data.balance_ratio
    fake = tr[tr.label == 1].sample(n=min(int(n), int((tr.label == 1).sum())), random_state=cfg.seed)
    return pd.concat([tr[tr.label == 0], fake]), va


def train_frame(cfg: DictConfig) -> Path:
    L.seed_everything(cfg.seed)
    df = pd.read_csv(cfg.data.crops)
    tr, va = _split(df, cfg)
    size = INPUT_SIZE.get(cfg.backbone, cfg.img_size)
    dl_tr = DataLoader(
        CropDataset(tr, size, True, cfg.seed),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.workers > 0,
    )
    dl_va = DataLoader(CropDataset(va, size, False), batch_size=cfg.batch_size, num_workers=cfg.workers)
    mod = FrameModule(cfg)
    logger = _logger(cfg)
    ckpt = L.pytorch.callbacks.ModelCheckpoint(monitor="val/auc", mode="max", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        logger=logger,
        callbacks=[ckpt, L.pytorch.callbacks.EarlyStopping("val/auc", mode="max", patience=3)],
        gradient_clip_val=1.0,
        log_every_n_steps=20,
    )
    trainer.fit(mod, dl_tr, dl_va)
    best = FrameModule.load_from_checkpoint(ckpt.best_model_path, cfg=cfg) if ckpt.best_model_path else mod
    m: Any = getattr(best.model, "_orig_mod", best.model)  # unwrap torch.compile
    out = models_dir() / "video" / f"{cfg.backbone.split('.')[0]}_trained.pth"
    torch.save(m.state_dict(), out)
    print("saved", out)
    return out


def train_temporal(cfg: DictConfig) -> Path:
    """Freeze the frame model, embed every crop, train a BiLSTM / transformer over per-video sequences."""
    L.seed_everything(cfg.seed)
    df = pd.read_csv(cfg.data.crops)
    w = ensure(f"video.{cfg.backbone}") if cfg.backbone in ("effnb4", "xception") else cfg.frame_weights
    fm = build_frame_model(cfg.backbone, w).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fm.to(dev)
    size = INPUT_SIZE.get(cfg.backbone, cfg.img_size)
    seqs: dict[str, dict] = {}
    with torch.inference_mode(), torch.autocast(dev, enabled=dev == "cuda"):
        for vid, g in df.groupby("video_id"):
            g = g.sort_values("frame")
            imgs = [_read(p) for p in g.crop]
            emb = fm(torch.from_numpy(preprocess(imgs, size)).to(dev))[1].float().cpu()
            r = g.iloc[0]
            seqs[vid] = {"emb": emb, "label": float(r.label), "split": r.split, "dataset": r.dataset}
    head = build_temporal_head(cfg.temporal, fm.feat_dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    tr = [s for s in seqs.values() if s["split"] == "train" and s["dataset"] in cfg.data.train_datasets]
    t = cfg.temporal_len
    for _ep in range(cfg.temporal_epochs):
        tr = [tr[i] for i in np.random.permutation(len(tr))]
        for i in range(0, len(tr), 32):
            b = tr[i : i + 32]
            x = torch.stack([_crop_seq(s["emb"], t) for s in b]).to(dev)
            y = torch.tensor([s["label"] for s in b], device=dev)
            loss = nn.functional.binary_cross_entropy_with_logits(head(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    out = models_dir() / "video" / f"temporal_{cfg.temporal}_{cfg.backbone}.pt"
    torch.save({"kind": cfg.temporal, "in_dim": fm.feat_dim, "state_dict": head.cpu().state_dict()}, out)
    print("saved", out)
    return out


def _crop_seq(e: torch.Tensor, t: int) -> torch.Tensor:
    """Random contiguous crop of length t; short sequences are tiled."""
    if len(e) == 0:
        return torch.zeros(t, e.shape[1])
    if len(e) >= t:
        s = np.random.randint(0, len(e) - t + 1)
        return e[s : s + t]
    return e.repeat(int(np.ceil(t / len(e))), 1)[:t]


def _logger(cfg: DictConfig):
    if cfg.tracking == "mlflow":
        try:
            return L.pytorch.loggers.MLFlowLogger(
                experiment_name="trusthire-video", tracking_uri=f"file:{REPO_ROOT / 'mlruns'}"
            )
        except Exception:
            pass
    return L.pytorch.loggers.CSVLogger(str(REPO_ROOT / "logs"), name="video")


@hydra.main(config_path=str(REPO_ROOT / "configs" / "video"), config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train_frame(cfg) if cfg.stage == "frame" else train_temporal(cfg)


if __name__ == "__main__":
    main()
