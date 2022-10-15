import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def predict(model: torch.nn.Module, dataloader: DataLoader):
    model.eval()
    prediction = {}
    prediction["id"] = []
    prediction["intent_idx"] = []
    with torch.no_grad():
        for sequences in dataloader:
            prediction["id"] += sequences["id"]
            if args.forward_method == "pad_pack":
                sequences["text_idx"] = sequences["text_idx"].to(args.device)
                sequences["intent_idx"] = sequences["intent_idx"].to(args.device)
                preds = model(sequences)
            else:
                preds = model(sequences["text_idx"].to(args.device))
            preds_idx = torch.argmax(preds, dim=-1)

            preds_idx = preds_idx.int().tolist()

            prediction["intent_idx"] += [dataloader.dataset.idx2label(idx) for idx in preds_idx]
    return prediction


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
        ruduce_seq=True,
    ).to(args.device)

    model.eval()

    ckpt = torch.load(args.load_ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    # TODO: predict dataset
    prediction = predict(model=model, dataloader=dataloader)
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
