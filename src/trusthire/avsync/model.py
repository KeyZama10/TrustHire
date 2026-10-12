"""SyncNet (Chung & Zisserman 2016) split into exportable lip / audio towers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from trusthire.third_party.syncnet import S


class LipTower(nn.Module):
    def __init__(self, s: S):
        super().__init__()
        self.s = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,5,224,224) BGR 0..255
        return self.s.forward_lip(x)  # type: ignore[no-any-return]


class AudioTower(nn.Module):
    def __init__(self, s: S):
        super().__init__()
        self.s = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,1,13,20) MFCC
        return self.s.forward_aud(x)  # type: ignore[no-any-return]


def build_syncnet(weights: str | Path) -> S:
    s = S(num_layers_in_fc_layers=1024)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    s.load_state_dict(state)
    return s.eval()


def mfcc(audio_f32: np.ndarray, sr: int = 16000) -> np.ndarray:
    """MFCC exactly as syncnet_python: python_speech_features on int16-scaled audio -> (13, T@100Hz)."""
    import python_speech_features

    a = (np.clip(audio_f32, -1, 1) * 32767).astype(np.int16)
    return np.asarray(python_speech_features.mfcc(a, sr)).T.astype(np.float32)
