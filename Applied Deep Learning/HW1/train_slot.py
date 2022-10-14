import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # TODO: implement main function
    raise NotImplementedError


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--save_ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    parser.add_argument(
        "--load_ckpt_dir",
        type=Path,
        help="Directory to load the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--workers", default=4, type=int, help="the number of data loading workers (default: 4)")

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument(
        "--model_out",
        default="output",
        type=str,
        choices=["output", "hidden"],
        help="type of model forward method (output, hidden)",
    )
    parser.add_argument(
        "--forward_method",
        default="basic",
        type=str,
        choices=["basic", "pad_pack"],
        help="type of model forward method (basic, pad_pack)",
    )

    # optimizer
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float, help="the initial learning rate")
    parser.add_argument(
        "-wd", "--weight_decay", default=0.01, type=float, help="the weight decay for L2-regularization"
    )
    parser.add_argument(
        "--optimizer_type", default="AdamW", type=str, choices=["AdamW", "Adam"], help="type of optimizer (AdamW, Adam)"
    )
    parser.add_argument(
        "--scheduler_type",
        default="reduce",
        type=str,
        choices=["reduce", "exponential", None],
        help="type of scheduler (ReduceLROnPlateau, exponentialLR)",
    )
    parser.add_argument(
        "--lr_patience", default=5, type=int, help="the learning rate patience for decreasing laerning rate (default 5)"
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("-seed", "--random_seed", default=888, type=int, help="the seed (default 888)")
    parser.add_argument(
        "--epoch_patience", default=40, type=int, help="the epoch patience for early stopping (default 40)"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.save_ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)