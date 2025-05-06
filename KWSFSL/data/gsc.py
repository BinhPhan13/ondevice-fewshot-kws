import hashlib
import random
import re
from os.path import basename, dirname
from os.path import join as join_path
from typing import Dict, List, Set

from loguru import logger
from torch.utils.data import DataLoader

from .base import AudioDataset
from .utils import EpisodicBatchSampler, LoaderDataset, Word2Data


def which_set(hashname: str, validation_pct: float, testing_pct: float):
    # Split dataset in training, validation, and testing set
    # Should be modified to load validation data from validation_list.txt
    # Should be modified to load testing data from testing_list.txt

    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    MAX_NUM_WAVS_PER_CLASS = (1 << 27) - 1  # ~134M
    N = MAX_NUM_WAVS_PER_CLASS

    hashname_hashed = hashlib.sha1(hashname.encode()).hexdigest()
    hash_pct = (int(hashname_hashed, 16) % (N + 1)) * (100.0 / N)

    if hash_pct < validation_pct:
        result = 'validation'
    elif hash_pct < (testing_pct + validation_pct):
        result = 'testing'
    else:
        result = 'training'
    return result


class GSCDataset(AudioDataset):
    NOISE_LABEL = '_background_noise_'
    RANDOM_SEED = 59185

    def __init__(self, args, task: str):
        self.silence = args['include_silence']
        self.unknown = args['include_unknown']

        self.resolve_target(task)

        self.silence_pct = 10.0
        self.unknown_pct = 10.0
        self.validation_pct = 10.0
        self.testing_pct = 10.0

        super().__init__(args)


    def get_iid_dataloader(
        self,
        set_idx: str,
        class_list: List[str],
        batch_size: int,
        n_workers: int = 8,
        pin_memory: bool = False,
    ):
        dataset = self.dataset[set_idx]
        data_list = sum([dataset[word] for word in class_list], [])
        ts_ds = self.get_transform_dataset(data_list)
        dl = DataLoader(
            ts_ds, # type: ignore
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=pin_memory,
        )
        return dl


    def get_episodic_dataloader(
        self,
        set_idx: str,
        class_list: List[str],
        n_way: int,
        n_shot: int,
        n_eps: int,
        n_workers: int = 8,
        pin_memory: bool = False,
    ):
        dataset = self.dataset[set_idx]
        dl_list: List[DataLoader] = []
        for word in class_list:
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
        sampler = EpisodicBatchSampler(len(class_list), n_way, n_eps)
        dl = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=n_workers,
            pin_memory=pin_memory,
        )
        return dl


    def get_class_list(
        self,
        include_silence: bool = True,
        include_unknown: bool = True,
    ):
        exclude: Set[str] = set()
        if not include_silence: exclude.add(self.SILENCE_LABEL)
        if not include_unknown: exclude.add(self.UNKNOWN_LABEL)

        class_list = [word for word in self.word2idx if word not in exclude]
        return class_list



    def resolve_target(self, task: str):
        unknown_words = ['backward', 'forward', 'visual', 'follow', 'learn']
        if task == 'GSC12':
            target='yes,no,up,down,left,right,on,off,stop,go'
            logger.debug("10 word")
        elif task == 'GSC22':
            self.silence = True
            target = 'bed,bird,cat,dog,eight,five,four,happy,house,marvin,nine,one,seven,sheila,six,three,tree,two,wow,zero'
            unknown_words = []
            logger.debug("20 word")
        elif task == 'GSC10':
            target='bed,bird,cat,dog,eight,five,four,nine,one,seven,six,three,tree,two,zero'
            unknown_words = []
            logger.debug("10 words for meta train task")
        elif task == 'GSC5':
            target='happy,house,marvin,sheila,wow'
            unknown_words = []
            logger.debug("5 words for meta val task")
        else:
            target='yes,no,up,down,left,right,on,off,stop,go,bed,bird,cat,dog,eight,five,four,happy,house,marvin,nine,one,seven,sheila,six,three,tree,two,wow,zero'
            logger.debug("35 word - 5 words")
        wanted_words: List[str] = target.split(',')  # type:ignore

        self.wanted_words = wanted_words
        self.unknown_words = unknown_words


    def generate_data_dictionary(self):
        dataset: Dict[str, Word2Data] = {
            'training': {},
            'validation': {},
            'testing': {},
        }
        unknown: Dict[str, Word2Data] = {
            'training': {},
            'validation': {},
            'testing': {},
        }

        all_words: Set[str] = set()
        wanted_words = set(self.wanted_words)
        unknown_words = set(self.unknown_words)

        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_path = ''

        for wav_path in self.get_audio_paths():
            if not silence_path: silence_path = wav_path

            word = basename(dirname(wav_path)).lower()
            # Ignore background noise, as it has been handled by generate_background_noise()
            if word == self.NOISE_LABEL: continue
            all_words.add(word)

            filename = basename(wav_path)
            assert (m := re.search(r'^[\da-f]{8}', filename))
            hashname = m.group(0)

            # Determine the set to which the word should belong
            set_idx = which_set(hashname, self.validation_pct, self.testing_pct)

            # If it's a known class, store its detail, otherwise add it to
            # the list we'll use to train the unknown label.
            # If we use 35 classes - all are known, hence no unknown samples
            data = {'file': wav_path, 'speaker': hashname}
            if word in wanted_words:
                data['label'] = word
                dataset[set_idx].setdefault(word, []).append(data)
            elif word in unknown_words:
                data['label'] = self.UNKNOWN_LABEL
                unknown[set_idx].setdefault(self.UNKNOWN_LABEL, []).append(data)

        if not all_words: raise ValueError("No words found")
        if not_found_words := wanted_words - all_words:
            raise ValueError(f"Unable to find {','.join(not_found_words)}")

        # Add silence and unknown words to each set
        for set_idx in dataset:
            set_size = len(dataset[set_idx])
            if self.silence:
                silence_data = {
                    'label': self.SILENCE_LABEL,
                    'file': silence_path,
                    'speaker': 'None',
                }
                silence_size = int(set_size * self.silence_pct / 100) + 1
                dataset[set_idx][self.SILENCE_LABEL] = [
                    silence_data for _ in range(silence_size)
                ]

            if self.unknown:
                unknown_size = int(set_size * self.unknown_pct / 100) + 1
                random.seed(self.RANDOM_SEED)
                dataset[set_idx][self.UNKNOWN_LABEL] = random.sample(
                    unknown[set_idx][self.UNKNOWN_LABEL], unknown_size
                )

        # Make sure the ordering is random.
        for set_data in dataset.values():
            for word_data in set_data.values():
                random.seed(self.RANDOM_SEED)
                random.shuffle(word_data)
        self.dataset = dataset

        # Make word to index mapping
        words_list: List[str] = []
        if self.silence:
            words_list.append(self.SILENCE_LABEL)
        if self.unknown:
            words_list.append(self.UNKNOWN_LABEL)
        words_list.extend(self.wanted_words)
        self.word2idx = {word: i for i, word in enumerate(words_list)}


    def get_audio_paths(self):
        index_file = '.audio.index.txt'
        with open(join_path(self.data_dir, index_file)) as f:
            for line in f: yield join_path(self.data_dir, line)

