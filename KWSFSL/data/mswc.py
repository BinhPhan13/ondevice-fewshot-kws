from collections import defaultdict
from typing import Dict, List

from torch.utils.data import DataLoader

from .base import AudioDataset
from .utils import EpisodicBatchSampler, Json, LoaderDataset


class MSWCDataset(AudioDataset):
    NOISE_POW_SCALED = True
    DatasetType = Dict[str, List[Json]]

    def __init__(self, args):
        self.csv_file = args['csv_file']
        super().__init__(args)


    def generate_data_dictionary(self):
        self.dataset = defaultdict(list)
        with open(self.csv_file) as f:
            f.readline()
            for line in f:
                link, word, *rest = line.split(',', maxsplit=2)
                if self.use_wav: link = link.replace('.opus', '.wav')
                self.dataset[word].append({'label': word, 'file': link})

        self.dataset: MSWCDataset.DatasetType = dict(self.dataset)
        self.all_words = list(self.dataset.keys())
        self.word_to_index = {
            word: idx for idx, word in enumerate(self.all_words, 1)
        }

    def get_episodic_dataloader(
        self,
        n_way: int,
        n_shot: int,
        n_eps: int,
        n_workers: int = 8,
        pin_memory: bool = False,
    ):
        dl_list: List[DataLoader] = []
        for word in self.all_words:
            data_list = self.dataset[word]
            if len(data_list) < n_shot:
                raise ValueError(
                    f"The word {word} has {len(data_list)} samples in total, "
                    f"which < {n_shot = }"
                )

            ts_ds = self.get_transform_dataset(data_list)
            dl = DataLoader(
                ts_ds, # type:ignore
                batch_size=n_shot,
                shuffle=True,
                num_workers=0,
            )
            dl_list.append(dl)

        ds = LoaderDataset(dl_list)
        sampler = EpisodicBatchSampler(len(self.all_words), n_way, n_eps)
        dl = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=n_workers,
            pin_memory=pin_memory,
        )
        return dl

