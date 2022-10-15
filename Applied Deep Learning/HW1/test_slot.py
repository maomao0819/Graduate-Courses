import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def predict(model: torch.nn.Module, dataloader: DataLoader, tag2idx: Dict):
    model.eval()
    prediction = {}
    prediction["id"] = []
    prediction["tags_idx"] = []
    with torch.no_grad():
        n_batch = len(dataloader)
        tqdm_loop = tqdm((dataloader), total=n_batch)
        for batch_idx, sequences in enumerate(tqdm_loop, 1):
            # [batch_size]
            prediction["id"] += sequences["id"]
            # [batch_size, seq_len]
            sequences["tokens_idx"] = sequences["tokens_idx"].to(args.device)
            # [batch_size, num_class, seq_len]
            preds = model(sequences)["logits"]

            # [batch_size, seq_len]
            preds_idx = torch.argmax(preds, dim=1)
            preds_idx = preds.max(1)[1]  # get the index of the max log-probability

            # [batch_size, seq_len]
            mask = sequences["mask"]
            # [batch_size, seq_len]
            preds_idx = preds_idx * mask

            preds_idx = np.array(preds_idx.int().tolist())
            preds_ids = np.ndarray(preds_idx.shape, dtype="<U16")
            for key in tag2idx:
                value = tag2idx[key]
                preds_ids[preds_idx == value] = key

            for instance_id in range(len(preds_ids)):
                instance_length = sequences["len"][instance_id]
                prediction["tags_idx"] += [preds_ids[instance_id][:instance_length]]

            tqdm_loop.set_description(f"Batch [{batch_idx}/{n_batch}]")

    return prediction


def main(args):
    # implement main function
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len, "test", args.forward_method == "pad_pack")
    # crecate DataLoader for test dataset
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
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
    # predict dataset
    prediction = predict(model=model, dataloader=dataloader, tag2idx=tag2idx)
    # write prediction to file (args.pred_file)
    args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.pred_file, "w") as f:
        f.write("id,tags\n")
        for ids, tags_idx in zip(prediction["id"], prediction["tags_idx"]):
            tags_id_str = ""
            for tag_idx in tags_idx:
                tags_id_str += tag_idx + " "
            f.write("%s,%s\n" % (ids, tags_id_str[:-1]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=Path, help="Path to the test file.", required=True)
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
        default="./ckpt/tag/",
    )
    parser.add_argument(
        "--load_ckpt_path",
        type=Path,
        help="Directory to load the model file.",
        default="./ckpt/tag/best.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

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
