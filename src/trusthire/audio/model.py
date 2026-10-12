"""Voice anti-spoofing models.

* AASIST / AASIST-L  — raw-waveform spectro-temporal graph attention (Jung et al. 2022), pretrained on ASVspoof19 LA
* RawNet2            — raw-waveform SincNet + GRU baseline (Tak et al. 2021), trained with `audio.train`
* SSLClassifier      — wav2vec2 / WavLM / XLS-R encoder + attentive-pooling head, fine-tuned with `audio.train`
* Spectra-0          — XLS-R-300M + ECAPA head (lab260, Apache-2.0), shipped as ONNX; strongest public checkpoint

All torch models here output 2 logits with index 1 = bona fide (the AASIST/ASVspoof convention);
`infer.py` converts every backend to P(spoof) so higher always means more suspicious.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

AASIST_CFG = {
    "aasist": {
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    },
    "aasist_l": {
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    },
}
NB_SAMP = 64600  # ~4.04 s at 16 kHz (ASVspoof convention)


class AASISTWrapper(nn.Module):
    """AASIST returning logits only; index 1 = bona fide (upstream convention)."""

    def __init__(self, variant: str = "aasist"):
        super().__init__()
        from trusthire.third_party.aasist import Model

        self.net = Model(AASIST_CFG[variant])

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        _, out = self.net(wav)
        return out  # type: ignore[no-any-return]


def build_aasist(variant: str = "aasist", weights: str | Path | None = None) -> AASISTWrapper:
    m = AASISTWrapper(variant)
    if weights:
        m.net.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    return m.eval()


# ------------------------------------------------------------------ RawNet2
class SincConv(nn.Module):
    """Mel-initialised sinc band-pass filterbank (fixed), as in RawNet2."""

    def __init__(self, out_channels: int = 20, kernel_size: int = 1024, sample_rate: int = 16000):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        nfft = 512
        f = int(sample_rate / 2) * torch.linspace(0, 1, nfft // 2 + 1)
        fmel = 2595 * torch.log10(1 + f / 700)
        filbandwidthsmel = torch.linspace(fmel.min(), fmel.max(), out_channels + 1)
        mel = 700 * (10 ** (filbandwidthsmel / 2595) - 1)
        hsupp = torch.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
        filters = torch.zeros(out_channels, kernel_size)
        for i in range(out_channels):
            fmin, fmax = mel[i], mel[i + 1]
            h_high = (2 * fmax / sample_rate) * torch.sinc(2 * fmax * hsupp / sample_rate)
            h_low = (2 * fmin / sample_rate) * torch.sinc(2 * fmin * hsupp / sample_rate)
            filters[i] = torch.hamming_window(kernel_size, periodic=False) * (h_high - h_low)
        self.register_buffer("filters", filters.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.filters, padding=self.kernel_size // 2)  # type: ignore[arg-type]


class _FMS(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.fc = nn.Linear(c, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.fc(x.mean(-1))).unsqueeze(-1)
        return x * s + s


class _ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, first: bool = False):
        super().__init__()
        self.first = first
        self.bn1 = nn.BatchNorm1d(cin)
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.pool = nn.MaxPool1d(3)
        self.fms = _FMS(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x if self.first else F.leaky_relu(self.bn1(x), 0.3)
        h = self.conv2(F.leaky_relu(self.bn2(self.conv1(h)), 0.3))
        return self.fms(self.pool(h + self.down(x)))


class RawNet2(nn.Module):
    def __init__(self, sinc_channels: int = 20, filts: tuple[int, int] = (20, 128), gru: int = 1024):
        super().__init__()
        self.sinc = SincConv(sinc_channels)
        self.bn0 = nn.BatchNorm1d(sinc_channels)
        self.blocks = nn.Sequential(
            _ResBlock(filts[0], filts[0], first=True),
            _ResBlock(filts[0], filts[0]),
            _ResBlock(filts[0], filts[1]),
            _ResBlock(filts[1], filts[1]),
            _ResBlock(filts[1], filts[1]),
            _ResBlock(filts[1], filts[1]),
        )
        self.bn_gru = nn.BatchNorm1d(filts[1])
        self.gru = nn.GRU(filts[1], gru, num_layers=3, batch_first=True)
        self.fc = nn.Linear(gru, gru)
        self.out = nn.Linear(gru, 2)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.sinc(wav.unsqueeze(1))
        x = F.leaky_relu(self.bn0(F.max_pool1d(torch.abs(x), 3)), 0.3)
        x = self.blocks(x)
        x = F.leaky_relu(self.bn_gru(x), 0.3).transpose(1, 2)
        h, _ = self.gru(x)
        return self.out(self.fc(h[:, -1]))  # index 1 = bona fide


# ------------------------------------------------------------------ SSL front-end classifiers
class AttentivePool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.att(h), dim=1)
        mu = (w * h).sum(1)
        sd = torch.sqrt(((w * (h - mu.unsqueeze(1)) ** 2).sum(1)).clamp(min=1e-6))
        return torch.cat([mu, sd], -1)


class SSLClassifier(nn.Module):
    """wav2vec2 / WavLM / XLS-R encoder with learnable layer weighting + attentive stats pooling.

    Upstream checkpoints: facebook/wav2vec2-xls-r-300m, microsoft/wavlm-large, facebook/wav2vec2-base.
    """

    def __init__(self, encoder_id: str = "microsoft/wavlm-base-plus", freeze_feature_encoder: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_id, output_hidden_states=True)
        if freeze_feature_encoder and hasattr(self.encoder, "feature_extractor"):
            for p in self.encoder.feature_extractor.parameters():
                p.requires_grad = False
        n_layers = self.encoder.config.num_hidden_layers + 1
        d = self.encoder.config.hidden_size
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.pool = AttentivePool(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 2))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-5)
        hs = torch.stack(self.encoder(wav).hidden_states, 0)
        h = (torch.softmax(self.layer_w, 0)[:, None, None, None] * hs).sum(0)
        return self.head(self.pool(h))  # index 1 = bona fide


def pad_or_tile(x: torch.Tensor, n: int = NB_SAMP) -> torch.Tensor:
    """ASVspoof-style: tile short clips, crop long ones."""
    if x.shape[-1] >= n:
        return x[..., :n]
    reps = math.ceil(n / max(1, x.shape[-1]))
    return x.repeat(*([1] * (x.dim() - 1)), reps)[..., :n]
