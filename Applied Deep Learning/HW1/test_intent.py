import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def predict(model: torch.nn.Module, dataloader: DataLoader, device:torch.device):
    model.eval()
    prediction = {}
    prediction["id"] = []
    prediction["intent_idx"] = []
    with torch.no_grad():
        n_batch = len(dataloader)
        tqdm_loop = tqdm((dataloader), total=n_batch)
        for batch_idx, sequences in enumerate(tqdm_loop, 1):
            # [batch_size]
            prediction["id"] += sequences["id"]
            # [batch_size, seq_len]
            sequences["text_idx"] = sequences["text_idx"].to(args.device)
            # [batch_size, num_class]
            preds = model(sequences)['logits']
            # [batch_size]
            preds_idx = torch.argmax(preds, dim=-1)

            preds_idx = preds_idx.int().tolist()

            prediction["intent_idx"] += [dataloader.dataset.idx2label(idx) for idx in preds_idx]
            tqdm_loop.set_description(f"Batch [{batch_idx}/{n_batch}]")
    return prediction

def val_performance(model: torch.nn.Module, dataloader: DataLoader, device:torch.device):
    model.eval()
    epoch_correct = 0
    n_batch = len(dataloader)
    tqdm_loop = tqdm((dataloader), total=n_batch)
    for batch_idx, sequences in enumerate(tqdm_loop, 1):
        with torch.no_grad():
            # [batch_size, seq_len]
            sequences["text_idx"] = sequences["text_idx"].to(device)
            # [batch_size, num_class]
            texts = model(sequences)['logits']
            # [batch_size]
            labels = sequences["intent_idx"].to(device)

            # [batch_size]
            # batch_correct = (torch.argmax(texts, dim=-1) == labels).float().sum().item()
            pred = texts.max(1)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            epoch_correct += batch_correct
            tqdm_loop.set_description(f"Batch [{batch_idx}/{n_batch}]")

    print(epoch_correct / len(dataloader.dataset))

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len, mode="test")
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        args.forward_method,
        args.model_out,
    ).to(args.device)

    model.eval()

    ckpt = torch.load(args.load_ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    if args.mode == "eval":
        val_performance(model=model, dataloader=dataloader, device=args.device)
    else:
        # TODO: predict dataset
        prediction = predict(model=model, dataloader=dataloader, device=args.device)
        # TODO: write prediction to file (args.pred_file)
        args.pred_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.pred_file, "w") as f:
            f.write("id,intent\n")
            for ids, intents_idx in zip(prediction["id"], prediction["intent_idx"]):
                f.write("%s,%s\n" % (ids, intents_idx))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=Path, help="Path to the test file.", required=True)
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
        "--load_ckpt_path",
        type=Path,
        help="Directory to load the model file.",
        default="./ckpt/intent/best.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")
    parser.add_argument("--mode", type=str, default="test", choices=["test", "eval"])

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
    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu")
    parser.add_argument("-seed", "--random_seed", default=888, type=int, help="the seed (default 888)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
