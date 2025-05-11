import random
from collections import defaultdict
from typing import List

from torch.utils.data import DataLoader

from .base import AudioDataset
from .utils import EpisodicBatchSampler, LoaderDataset, Word2Data


class MSWCDataset(AudioDataset):
    NOISE_POW_SCALED = True

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

        self.dataset: Word2Data = dict(self.dataset)
        self.word2idx = {word: idx+1 for idx, word in enumerate(self.dataset)}


    def get_episodic_dataloader(
        self,
        n_way: int,
        n_shot: int,
        n_eps: int,
        n_workers: int = 8,
        pin_memory: bool = False,
    ):
        dataset = self.dataset
        dl_list: List[DataLoader] = []
        for word in dataset:
            data_list = dataset[word]
            if len(data_list) < n_shot:
                raise ValueError(
                    f"The word {word} has {len(data_list)} samples in total, "
                    f"which smaller than {n_shot = }"
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
        sampler = EpisodicBatchSampler(self.num_classes, n_way, n_eps)
        dl = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=n_workers,
            pin_memory=pin_memory,
        )
        return dl


    def get_eval_dataloaders(
        self,
        npos: int,
        nneg: int,
        nshot: int,
        threshold: int,
        batch_size: int = 256,
        nworkers: int = 8,
        pin_memory: bool = False,
    ):
        # make sure all random has the same seed
        random_state = random.getstate()

        random.setstate(random_state)
        wanted_words = random.sample(list(self.word2idx), npos + nneg)
        pos_words = wanted_words[:npos]
        neg_words = wanted_words[npos:]

        dataset = self.dataset
        ds_train: Word2Data = {}
        ds_test: Word2Data = {}

        for word in pos_words:
            random.setstate(random_state)
            files = random.sample(dataset[word], threshold)

            ds_train[word] = files[:nshot]
            ds_test[word] = files[nshot:]

        for word in neg_words:
            random.setstate(random_state)
            files = random.sample(dataset[word], threshold)
            ds_test[word] = files

        dl_list = [
            DataLoader(
                self.get_transform_dataset(ds_train[word]),  # type:ignore
                batch_size=nshot,
                num_workers=0,
            )
            for word in pos_words
        ]
        dl_train = DataLoader(
            LoaderDataset(dl_list),
            batch_size=npos,
            num_workers=nworkers,
            pin_memory=pin_memory,
        )

        data_list = sum([ds_test[word] for word in neg_words], [])
        dl_test = DataLoader(
            self.get_transform_dataset(data_list),  # type:ignore
            batch_size=batch_size,
            num_workers=nworkers,
            pin_memory=pin_memory,
            shuffle=True,
        )

        return dl_train, dl_test

