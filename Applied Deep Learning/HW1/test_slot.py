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


def predict(model: torch.nn.Module, dataloader: DataLoader, tag2idx: Dict, device:torch.device):
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
            sequences["tokens_idx"] = sequences["tokens_idx"].to(device)
            # [batch_size, num_class, seq_len]
            preds = model(sequences)["logits"]

            # [batch_size, seq_len]
            preds_idx = torch.argmax(preds, dim=1)
            preds_idx = preds.max(1)[1]  # get the index of the max log-probability

            # [batch_size, seq_len]
            mask = sequences["mask"].to(device)
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


def val_performance(model: torch.nn.Module, dataloader: DataLoader, tag2idx: Dict, device:torch.device):
    from seqeval.metrics import accuracy_score
    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2

    model.eval()
    predictions = []
    ground_truths = []
    n_seq = 0
    n_token = 0
    epoch_seq_correct = 0
    with torch.no_grad():
        n_batch = len(dataloader)
        tqdm_loop = tqdm((dataloader), total=n_batch)
        for batch_idx, sequences in enumerate(tqdm_loop, 1):
            if "tags" in sequences.keys():
                # [batch_size, seq_len]
                sequences["tokens_idx"] = sequences["tokens_idx"].to(device)
                # [batch_size, num_class, seq_len]
                preds = model(sequences)["logits"]
                # [batch_size, seq_len]
                tags = sequences["tags_idx"].to(device)

                # [batch_size, seq_len]
                preds_idx = torch.argmax(preds, dim=1)
                preds_idx = preds.max(1)[1]  # get the index of the max log-probability

                # [batch_size, seq_len]
                mask = sequences["mask"].to(device)

                batch_correct = 0
                # [batch_size, seq_len]
                preds_idx = preds_idx * mask
                tags = tags * mask

                batch_seq_correct = torch.all(torch.eq(preds_idx, tags), dim=1).sum().item()
                epoch_seq_correct += batch_seq_correct

                preds_idx = np.array(preds_idx.int().tolist())
                preds_ids = np.ndarray(preds_idx.shape, dtype="<U16")
                for key in tag2idx:
                    value = tag2idx[key]
                    preds_ids[preds_idx == value] = key

                for instance_id in range(len(preds_ids)):
                    n_seq += 1
                    instance_length = sequences["len"][instance_id]
                    n_token += instance_length
                    predictions += [preds_ids[instance_id][:instance_length].tolist()]
                    ground_truths += [sequences["tags"][instance_id]]
                tqdm_loop.set_description(f"Batch [{batch_idx}/{n_batch}]")
            else:
                return

        print(
            f"accuracy score: {int(accuracy_score(ground_truths, predictions) * n_token)} / {n_token} = {accuracy_score(ground_truths, predictions)}"
        )
        print(accuracy_score(ground_truths, predictions))
        print(f"joint accuracy : {epoch_seq_correct} / {n_seq} = {epoch_seq_correct / n_seq}")
        print("classification report: ")
        print(classification_report(ground_truths, predictions, scheme=IOB2, mode="strict"))
        accuracy_score


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
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len, args.mode, args.forward_method == "pad_pack")
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
    if args.mode == "eval":
        val_performance(model=model, dataloader=dataloader, tag2idx=tag2idx, device=args.device)
    else:
        # predict dataset
        prediction = predict(model=model, dataloader=dataloader, tag2idx=tag2idx, device=args.device)
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
