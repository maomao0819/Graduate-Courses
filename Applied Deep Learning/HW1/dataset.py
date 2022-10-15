from typing import List, Dict

import torch
from torch.utils.data import Dataset

import utils
from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        mode: str = "train",
        sort: bool = False,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode = mode
        self.sort = sort

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        if self.sort:
            samples.sort(key=lambda sequence: len(sequence["text"].split()), reverse=True)
        batch = {key: [sample[key] for sample in samples] for key in samples[0]}
        batch["text"] = [text.split() for text in batch["text"]]
        batch["len"] = torch.tensor([min(len(text), self.max_len) for text in batch["text"]])
        batch["text_idx"] = self.vocab.encode_batch(batch["text"])
        batch["text_idx"] = torch.LongTensor(batch["text_idx"])
        if "train" in self.mode or "eval" in self.mode or "intent" in batch.keys():
            batch["intent_idx"] = [self.label2idx(intent) for intent in batch["intent"]]
            batch["intent_idx"] = torch.LongTensor(batch["intent_idx"])
        # else:
        #     batch["intent_idx"] = torch.zeros(len(batch), dtype=torch.long)
        return batch
        raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    def __init__(self, *args):
        super().__init__(*args)
        self.ignore_idx = -100

    def get_ignore_idx(self):
        return self.ignore_idx

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        if self.sort:
            samples.sort(key=lambda sequence: len(sequence["tokens"].split()), reverse=True)
        batch = {key: [sample[key] for sample in samples] for key in samples[0]}
        batch["len"] = torch.tensor([min(len(token), self.max_len) for token in batch["tokens"]])
        batch_seq_len = torch.max(batch["len"])
        batch["tokens_idx"] = self.vocab.encode_batch(batch["tokens"], batch_seq_len)
        batch["tokens_idx"] = torch.LongTensor(batch["tokens_idx"])

        if "train" in self.mode or "eval" in self.mode or "tags" in batch.keys():
            batch["tags_idx"] = [[self.label2idx(tag) for tag in tags] for tags in batch["tags"]]
            batch["tags_idx"] = torch.LongTensor(utils.pad_to_len(batch["tags_idx"], batch_seq_len.int(), self.ignore_idx))
        else:
            batch["tags_idx"] = torch.zeros((len(batch["tokens_idx"]), batch_seq_len), dtype=torch.long)
        batch['mask'] = batch['tokens_idx'].gt(0)
        return batch
        raise NotImplementedError
