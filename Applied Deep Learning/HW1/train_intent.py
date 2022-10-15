import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import copy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab, save_checkpoint

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def run_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    mode: str,
) -> Dict:

    if mode == TRAIN:
        model.train()
    else:
        model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    epoch_loss = 0
    epoch_correct = 0
    n_batch = len(dataloader)
    tqdm_loop = tqdm((dataloader), total=n_batch)
    for batch_idx, sequences in enumerate(tqdm_loop, 1):
        with torch.set_grad_enabled(mode == TRAIN):
            # [batch_size, seq_len]
            sequences["text_idx"] = sequences["text_idx"].to(args.device)
            # [batch_size, num_class]
            texts = model(sequences)['logits']
            # [batch_size]
            labels = sequences["intent_idx"].to(args.device)
            optimizer.zero_grad()
            loss = criterion(texts, labels)

            if mode == TRAIN:
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            # [batch_size]
            # batch_correct = (torch.argmax(texts, dim=-1) == labels).float().sum().item()
            pred = texts.max(1)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            epoch_loss += batch_loss
            epoch_correct += batch_correct
            tqdm_loop.set_description(f"Batch [{batch_idx}/{n_batch}]")
            tqdm_loop.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{float(batch_correct) / float(labels.shape[0]):.4f}")

    if mode == DEV:
        if args.scheduler_type == "exponential":
            scheduler.step()
        elif args.scheduler_type == "reduce":
            scheduler.step(epoch_loss)

    performance = {}
    n_data = len(dataloader.dataset)
    performance["loss"] = epoch_loss / n_data
    performance["acc"] = epoch_correct / n_data
    return performance


def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len, split, args.forward_method == "pad_pack")
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    dataloaders = {
        split: DataLoader(
            dataset=split_dataset,
            batch_size=args.batch_size,
            shuffle=(split == TRAIN),
            num_workers=args.workers,
            collate_fn=split_dataset.collate_fn,
            pin_memory=True,
        )
        for split, split_dataset in datasets.items()
    }
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        datasets[TRAIN].num_classes,
        args.forward_method,
        args.model_out,
    ).to(args.device)

    # TODO: init optimizer
    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay,
            amsgrad=False,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.power(0.01, 1 / args.num_epoch))
    elif args.scheduler_type == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.lr_patience
        )
    else:
        scheduler = None

    best_val_loss = np.inf
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        performance_train = run_one_epoch(
            model, dataloader=dataloaders[TRAIN], optimizer=optimizer, scheduler=scheduler, mode=TRAIN
        )

        # TODO: Evaluation loop - calculate accuracy and save model weights
        performance_eval = run_one_epoch(
            model, dataloader=dataloaders[DEV], optimizer=optimizer, scheduler=scheduler, mode=DEV
        )

        if performance_eval["loss"] < best_val_loss:
            best_val_loss = performance_eval["loss"]
            best_model_weight = copy.deepcopy(model.state_dict())
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= args.epoch_patience:
                print("Early Stop")
                model.load_state_dict(best_model_weight)
                break

        epoch_pbar.set_description(f"Epoch [{epoch+1}/{args.num_epoch}]")
        epoch_pbar.set_postfix(
            train_loss=performance_train["loss"],
            train_acc=performance_train["acc"],
            eval_loss=performance_eval["loss"],
            eval_acc=performance_eval["acc"],
        )

    model.load_state_dict(best_model_weight)
    save_checkpoint(args.save_ckpt_dir / "best.pt", model)

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
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
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu")
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
