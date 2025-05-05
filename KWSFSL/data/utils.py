import random
from functools import wraps
from time import time
from typing import Any, Callable, Dict, List, TypeVar

import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler


Json = Dict[str, Any]
JsonStr = Dict[str, str]
JsonInt = Dict[str, int]

def npow2(x: int):
    if x < 1: return 1
    is_pow2 = (x & (x-1)) == 0
    return x if is_pow2 else 1 << x.bit_length()


T = TypeVar('T')
def timeit(f0: Callable[..., T]):
    @wraps(f0)
    def f1(*args, **kwargs) -> T:
        start = time()
        ret = f0(*args, **kwargs)
        duration = time() - start

        logger.debug(f"{f0.__qualname__} takes {duration:.3f}s")
        return ret

    return f1


class LoaderDataset(Dataset):
    def __init__(self, dl_list: List[DataLoader]):
        self.dl_list = dl_list

    def __getitem__(self, i):
        return next(iter(self.dl_list[i]))

    def __len__(self):
        return len(self.dl_list)


class EpisodicBatchSampler(Sampler):
    def __init__(self, n_classes: int, n_way: int, n_episodes: int):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.sampling = [
            torch.randperm(self.n_classes)[: self.n_way]
            for i in range(self.n_episodes)
        ]

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield self.sampling[i]


def load_audio(file: str, sample_rate: int = -1, num_frames: int = -1):
    sound, sr = torchaudio.load(file, normalize=True, num_frames=num_frames)
    if sample_rate > 0 and sr != sample_rate:
        sound = torchaudio.functional.resample(sound, sr, sample_rate)
    return sound


def adjust_volume(audio: torch.Tensor, fg_volume: float = 1.0):
    return audio * fg_volume


def shift_and_pad(
    audio: torch.Tensor,
    shift: int,
    desired_samples: int,
):
    shift_amount = random.randint(-shift, shift) if shift > 0 else 0
    if shift_amount > 0:
        shift_padding = (shift_amount, 0)
        shift_offset = 0
    else:
        shift_padding = (0, -shift_amount)
        shift_offset = -shift_amount

    # Ensure data length is equal to the number of desired samples
    audio_len = audio.size(1)
    if audio_len < desired_samples:
        pad = (0, desired_samples - audio_len)
        audio = F.pad(audio, pad, 'constant', 0)

    padded_fg = F.pad(audio, shift_padding, 'constant', 0)
    sliced_fg = torch.narrow(padded_fg, 1, shift_offset, desired_samples)

    return sliced_fg


def mix_noise(
    fg: torch.Tensor,
    bg: torch.Tensor,
    frequency: float = 0.95,
    volume: int = 5,
    pow_scaled: bool = False,
):
    bg_samples = len(bg)
    desired_samples = fg.size(1)
    if bg_samples <= desired_samples:
        raise ValueError(
            f"Background sample is too short! Need more than {desired_samples} "
            f"samples but only {bg_samples} were found"
        )

    if random.uniform(0, 1) >= frequency: return fg

    bg_offset = random.randint(0, bg_samples - desired_samples)
    bg_clipped = bg[bg_offset : bg_offset + desired_samples]
    bg_reshaped = bg_clipped.reshape([1, desired_samples])

    bg_vol = random.uniform(0, volume)
    if pow_scaled:
        s_pow = fg.pow(2).sum()
        n_pow = bg_reshaped.pow(2).sum()
        bg_vol = (s_pow / ((10**bg_vol / 10) * n_pow)).sqrt().item()

    bg_add = bg_reshaped * bg_vol + fg
    bg_clamped = torch.clamp(bg_add, -1.0, 1.0)
    return bg_clamped

