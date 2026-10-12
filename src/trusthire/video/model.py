"""Video deepfake models.

Frame-level backbones
  * effnb4   — EfficientNet-B4, DeepfakeBench layout (FF++ c23 pretrained weights available)
  * xception — Xception, DeepfakeBench layout (FF++ c23 pretrained weights available)
  * any timm model (e.g. convnextv2_tiny, vit_base_patch14_dinov2, eva02) — ImageNet/SSL init,
    fine-tuned with `trusthire.video.train`

Temporal heads over per-frame embeddings: BiLSTM and a small Transformer encoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

DFB_CFG = {"num_classes": 2, "inc": 3, "dropout": False, "mode": "Original"}
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
INPUT_SIZE = {"effnb4": 256, "xception": 256}


class FrameClassifier(nn.Module):
    """Uniform wrapper: forward(x) -> (logits[B,2], embedding[B,D])."""

    def __init__(self, name: str, backbone: nn.Module, feat_dim: int, gradcam_layer: nn.Module):
        super().__init__()
        self.name = name
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.gradcam_layer = gradcam_layer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b: Any = self.backbone  # concrete backbone types differ per branch
        if self.name == "effnb4":
            f = b.efficientnet.extract_features(x)
            emb = F.adaptive_avg_pool2d(f, 1).flatten(1)
            return b.last_layer(emb), emb
        if self.name == "xception":
            f = b.features(x)
            emb = F.adaptive_avg_pool2d(F.relu(f), 1).flatten(1)
            return b.last_linear(emb), emb
        emb = b.forward_head(b.forward_features(x), pre_logits=True)
        return self.head(emb), emb  # type: ignore[operator]


def _effnb4() -> tuple[nn.Module, int, nn.Module]:
    from efficientnet_pytorch import EfficientNet

    class _B4(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.efficientnet = EfficientNet.from_name("efficientnet-b4")
            # DeepfakeBench replaces the stem with an unpadded conv — keep it identical for weight compat.
            self.efficientnet._conv_stem = nn.Conv2d(3, 48, kernel_size=3, stride=2, bias=False)
            self.efficientnet._fc = nn.Identity()
            self.efficientnet.set_swish(memory_efficient=False)  # exportable
            self.last_layer = nn.Linear(1792, 2)

    m = _B4()
    return m, 1792, m.efficientnet._conv_head


def _xception() -> tuple[nn.Module, int, nn.Module]:
    from trusthire.third_party.dfb_xception import Xception

    m = Xception(dict(DFB_CFG))
    return m, 2048, m.conv4


def build_frame_model(
    name: str, weights: str | Path | None = None, pretrained_timm: bool = True
) -> FrameClassifier:
    bb: Any
    if name == "effnb4":
        bb, d, layer = _effnb4()
    elif name == "xception":
        bb, d, layer = _xception()
    else:
        import timm

        bb = timm.create_model(name, pretrained=pretrained_timm, num_classes=0)
        d = int(bb.num_features)
        layer = _last_conv_or_block(bb)
    model = FrameClassifier(name, bb, d, layer)
    if name not in ("effnb4", "xception"):
        model.head = nn.Linear(d, 2)
    if weights:
        sd = torch.load(str(weights), map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        if name in ("effnb4", "xception"):
            sd = {k.removeprefix("backbone."): v for k, v in sd.items()}
            missing, _unexpected = bb.load_state_dict(sd, strict=False)
            missing = [k for k in missing if "adjust_channel" not in k]
            if missing:
                raise RuntimeError(f"{name}: missing keys {missing[:5]}…")
        else:
            model.load_state_dict(sd, strict=True)
    return model.eval()


def _last_conv_or_block(m: nn.Module) -> nn.Module:
    for attr in ("stages", "blocks", "layers"):
        seq = getattr(m, attr, None)
        if seq is not None and len(seq):
            return seq[-1]  # type: ignore[no-any-return]
    convs = [x for x in m.modules() if isinstance(x, nn.Conv2d)]
    return convs[-1]


# ---------------------------------------------------------------- temporal heads
class BiLSTMHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden))
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.lstm(self.proj(seq))
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled = h.mean(1)
        return self.out(self.drop(pooled)).squeeze(-1)


class TransformerHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        heads: int = 4,
        layers: int = 2,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = seq.shape
        x = torch.cat([self.cls.expand(b, -1, -1), self.proj(seq)], 1) + self.pos[:, : t + 1]
        pad = None
        if mask is not None:
            pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=seq.device), ~mask.bool()], 1)
        h = self.enc(x, src_key_padding_mask=pad)
        return self.out(h[:, 0]).squeeze(-1)


def build_temporal_head(kind: str, in_dim: int) -> nn.Module:
    if kind == "bilstm":
        return BiLSTMHead(in_dim)
    if kind == "transformer":
        return TransformerHead(in_dim)
    raise ValueError(kind)
