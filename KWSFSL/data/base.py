import glob
import random
from functools import partial
from os.path import join as join_path
from typing import List

import torch
import torchaudio
from loguru import logger
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

from .utils import (
    Json,
    JsonInt,
    adjust_volume,
    load_audio,
    mix_noise,
    shift_and_pad,
)


class AudioDataset:
    SILENCE_LABEL = '_silence_'
    UNKNOWN_LABEL = '_unknown_'
    NOISE_POW_SCALED = False

    def __init__(self, args):
        self.num_frames = args['num_frames']
        self.sample_rate = args['sample_rate']
        self.desired_samples = self.sample_rate * args['clip_duration'] // 1000
        self.fg_volume = args['foreground_volume']
        self.time_shift_ms = args['time_shift']

        self.use_wav = args['use_wav']
        self.data_dir = args['default_datadir']
        self.noise_dir = args['noise_dir']

        self.use_noise = args['include_noise']
        self.noise_volume = args['noise_snr']
        self.noise_freq = args["noise_frequency"]
        if self.use_noise:
            logger.debug("Load noise data")
            self.noise_data = self.load_noise_data()
        else:
            self.noise_data = []

        logger.debug("Generate data dictionary")
        self.generate_data_dictionary()

        self.transforms = compose([
            partial(self._load_audio, 'file', 'label', 'data'),
            partial(self._adjust_volumn, 'data'),
            partial(self._shift_and_pad, 'data'),
            partial(self._mix_noise, 'data'),
            partial(self._label2idx, 'label', 'label_idx')
        ])


    @property
    def num_classes(self):
        return len(self.word2idx)

    def generate_data_dictionary(self):
        self.dataset = {}
        self.word2idx: JsonInt = {}


    def get_transform_dataset(self, data_list: List[Json]):
        ls_ds = ListDataset(data_list)
        ts_ds = TransformDataset(ls_ds, self.transforms)
        return ts_ds


    def load_noise_data(self):
        noise_path = join_path(self.noise_dir, '*.wav')
        if not (wav_paths := glob.glob(noise_path)):
            raise FileNotFoundError(f"No noise file found at {self.noise_dir}")

        noise_data: List[torch.Tensor] = []
        for wav_path in wav_paths:
            bg_sound, _ = torchaudio.load(wav_path)
            noise_data.append(bg_sound.flatten())
        return noise_data


    def _label2idx(self, label_key: str, out_key: str, d: Json):
        label_index = self.word2idx[d[label_key]]
        d[out_key] = torch.LongTensor([label_index]).squeeze()
        return d


    def _load_audio(self, file_key: str, label_key: str, out_key: str, d: Json):
        filepath = join_path(self.data_dir, str(d[file_key]))
        audio = load_audio(filepath, self.sample_rate, self.num_frames)
        if d[label_key] == self.SILENCE_LABEL:
            audio.zero_()
        d[out_key] = audio
        return d


    def _adjust_volumn(self, data_key: str, d: Json):
        audio = d[data_key]
        d[data_key] = adjust_volume(audio, self.fg_volume)
        return d


    def _shift_and_pad(self, data_key: str, d: Json):
        audio = d[data_key]
        shift = self.time_shift_ms * self.sample_rate // 1000
        d[data_key] = shift_and_pad(audio, shift, self.desired_samples)
        return d


    def _mix_noise(self, data_key: str, d: Json):
        if not self.use_noise: return d
        fg = d[data_key]
        bg = random.choice(self.noise_data)
        d[data_key] = mix_noise(
            fg, bg, self.noise_freq, self.fg_volume, self.NOISE_POW_SCALED
        )
        return d

