import argparse
import os
import torch

def boolean_string(str):
    return ("t" in str) or ("T" in str)

def arg_parse():
    parser = argparse.ArgumentParser()

    # data loader
    parser.add_argument("--workers", default=4, type=int, help="the number of data loading workers (default: 4)")

    # model
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("-seed", "--random_seed", default=888, type=int, help="the seed (default 888)")
    
    # path
    # datasets
    parser.add_argument(
        "--image_dir_pretrain",
        type=str,
        default=os.path.join("hw4_data", "mini", "train"),
        help="the directory to the image.",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join("hw4_data", "office", "val"),
        help="the directory to the image.",
    )

    parser.add_argument(
        "--csv_file_pretrain",
        type=str,
        default=os.path.join("hw4_data", "mini", "train.csv"),
        help="the path to the csv file.",
    )

    parser.add_argument(
        "--csv_file",
        type=str,
        default=os.path.join("hw4_data", "office", "val.csv"),
        help="the path to the csv file.",
    )

    parser.add_argument(
        "--pred_file",
        type=str,
        default=os.path.join("pred.csv"),
        help="the path to the prediction csv file.",
    )

    parser.add_argument(
        "--load",
        type=str,
        default=os.path.join("best_checkpoint", "SSL", "backbone.pth"),
        help="the path to load the checkpoint",
    )

    parser.add_argument(
        "--load_TA",
        type=str,
        default=os.path.join("hw4_data", "pretrain_model_SL.pt"),
        help="the path to load the checkpoint",
    )

    # checkpoint
    parser.add_argument(
        "--save",
        type=str,
        default=os.path.join("checkpoint"),
        help="the directory to save the checkpoint",
    )

    parser.add_argument("-is", "--image_size", default=128, type=int, help="the image size (default 128)")
    parser.add_argument(
        "--pretrain",
        default="Mine",
        type=str,
        choices=["None", "Mine", "TA"],
        help="pretrain weight (None, Mine, TA)",
    )
    # data loader
    parser.add_argument("--train_batch", default=512, type=int, help="the training batch size (default 32)")
    parser.add_argument("--test_batch", default=256, type=int, help="the testing batch size (default 32)")

    # optimizer
    parser.add_argument("-lr", "--learning_rate", default=7e-4, type=float, help="the initial learning rate")
    parser.add_argument(
        "-wd", "--weight_decay", default=1e-7, type=float, help="the weight decay for L2-regularization"
    )
    parser.add_argument(
        "--optimizer_type",
        default="AdamW",
        type=str,
        choices=["AdamW", "Adam", "SGD"],
        help="type of optimizer (AdamW, Adam, SGD)",
    )
    parser.add_argument(
        "--lr_patience",
        default=5,
        type=int,
        help="the learning rate patience for decreasing learning rate (default 5)",
    )
    parser.add_argument(
        "--scheduler_type",
        default="reduce",
        type=str,
        choices=["reduce", "exponential", "None"],
        help="type of scheduler (ReduceLROnPlateau, exponentialLR)",
    )

    # training
    parser.add_argument("--fix_backbone", type=boolean_string, default=False, help="fix backbone or not")
    parser.add_argument("--epoch", type=int, default=1000, help="the num of epoch (default 1000)")
    parser.add_argument(
        "--epoch_patience", default=40, type=int, help="the epoch patience for early stopping (default 40)"
    )

    # validation
    parser.add_argument(
        "--matrix", default="acc", type=str, choices=["acc", "loss"], help="validation matrix (default acc)"
    )
    parser.add_argument("--save_interval", default=5, type=int, help="the save interval (default 5)")

    args = parser.parse_args()

    return args
