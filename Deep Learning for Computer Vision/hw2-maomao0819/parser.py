import argparse
import os
import torch

def boolean_string(str):
    return ('t' in str) or ('T' in str)

def arg_parse(problem=1):
    parser = argparse.ArgumentParser()

    # data loader
    parser.add_argument("--workers", default=4, type=int, help="the number of data loading workers (default: 4)")

    # checkpoint
    parser.add_argument(
        "--save",
        type=str,
        default=os.path.join("checkpoint", f"p{problem}"),
        help="the directory to save the checkpoint",
    )

    # optimizer
    parser.add_argument(
        "--scheduler_type",
        default="reduce",
        type=str,
        choices=["reduce", "exponential", "None"],
        help="type of scheduler (ReduceLROnPlateau, exponentialLR)",
    )

    # training
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")

    if problem == 1:

        parser.add_argument("-t", "--task", type=str, choices=["A", "B"], default="B", help="the task in the problem.")

        # path
        # datasets
        parser.add_argument(
            "--data_dir",
            type=str,
            default=os.path.join("hw2_data", "face"),
            help="the directory to the dataset.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default=os.path.join("hw2_data", "face", "generate"),
            help="the directory to the output.",
        )

        # checkpoint
        parser.add_argument("--loadG", type=str, default=os.path.join("best_checkpoint", "P1", "P1_bestG.pth"), help="the directory to load the checkpointG")

        # data
        parser.add_argument("-is", "--image_size", type=int, default=64, help="the image size")

        # data loader
        parser.add_argument("--train_batch", default=32, type=int, help="the training batch size (default 32)")
        parser.add_argument("--test_batch", default=128, type=int, help="the testing batch size (default 32)")

        # optimizer
        parser.add_argument("-lr", "--learning_rate", default=0.0002, type=float, help="the initial learning rate")
        parser.add_argument(
            "-wd", "--weight_decay", default=0., type=float, help="the weight decay for L2-regularization"
        )
        parser.add_argument(
            "--optimizer_type",
            default="AdamW",
            type=str,
            choices=["AdamW", "Adam", "SGD", "SGD+AdamW"],
            help="type of optimizer (AdamW, Adam, SGD, SGD+AdamW)",
        )
        parser.add_argument(
            "--lr_patience",
            default=20,
            type=int,
            help="the learning rate patience for decreasing laerning rate (default 20)",
        )
        # training
        parser.add_argument("--D_steps", type=int, default=1, help="the num of steps of D (default 2)")
        parser.add_argument("--G_steps", type=int, default=1, help="the num of steps of G (default 2)")
        parser.add_argument("--epoch", type=int, default=1000, help="the num of epoch (default 1000)")
        parser.add_argument("-seed", "--random_seed", default=888, type=int, help="the seed (default 888)")
        parser.add_argument(
            "--epoch_patience", default=200, type=int, help="the epoch patience for early stopping (default 40)"
        )
        parser.add_argument("--label_smooth", type=boolean_string, default=True, help="label smoothing")
        parser.add_argument("--image_noise", type=boolean_string, default=True, help="Adding image noise")
        parser.add_argument("--reverse_train", type=int, default=5, help="Reverse trainning")

        # validation
        parser.add_argument(
            "--matrix", default="acc", type=str, choices=["acc", "loss"], help="validation matrix (default acc)"
        )
        parser.add_argument("--save_interval", default=1, type=int, help="the save interval (default 1)")

    elif problem == 2:
        # path
        # datasets
        parser.add_argument(
            "--data_dir",
            type=str,
            default=os.path.join("hw2_data", "digits", "mnistm"),
            help="the directory to the dataset.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default=os.path.join("hw2_data", "digits", "generate"),
            help="the directory to the output.",
        )
        parser.add_argument("--load", type=str, default=os.path.join("best_checkpoint", "P2", "P2_best_diffusion.pth"), 
            help="the directory to load the checkpoint")
        # data
        parser.add_argument("-is", "--image_size", type=int, default=28, help="the image size")

        # data loader
        parser.add_argument("--train_batch", default=128, type=int, help="the training batch size (default 32)")
        parser.add_argument("--test_batch", default=512, type=int, help="the testing batch size (default 32)")

        # model
        parser.add_argument("--time_length", type=int, default=200, help="the time length (default 500)")
        parser.add_argument("--channel", type=int, default=128, help="the number of channel (default 128)")
        parser.add_argument("--channel_multiply", default=[1, 2, 2], help="channel multiply")
        parser.add_argument("--n_residual_blocks", type=int, default=2, help="the number of the residual_blocks")
        parser.add_argument("--dropout", type=float, default=0.15, help="dropout")
        parser.add_argument("--multiplier", type=float, default=2.5, help="multiplier")
        parser.add_argument("--w", type=float, default=1.8, help="w")

        # optimizer
        parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float, help="the initial learning rate")
        parser.add_argument(
            "-wd", "--weight_decay", default=1e-4, type=float, help="the weight decay for L2-regularization"
        )
        parser.add_argument(
            "--optimizer_type",
            default="AdamW",
            type=str,
            choices=["AdamW", "Adam", "SGD", "SGD+AdamW"],
            help="type of optimizer (AdamW, Adam, SGD, SGD+AdamW)",
        )
        parser.add_argument(
            "--lr_patience",
            default=5,
            type=int,
            help="the learning rate patience for decreasing laerning rate (default 10)",
        )
        parser.add_argument("--beta_1", type=float, default=1e-4, help="beta_1")
        parser.add_argument("--beta_T", type=float, default=0.028, help="beta_T")

        # training
        parser.add_argument("--epoch", type=int, default=1000, help="the num of epoch (default 1000)")
        parser.add_argument("-seed", "--random_seed", default=888, type=int, help="the seed (default 888)")
        parser.add_argument(
            "--epoch_patience", default=200, type=int, help="the epoch patience for early stopping (default 40)"
        )
        parser.add_argument("--grad_clip", type=float, default=1.0, help="the value of gradient clipping")

        # validation
        parser.add_argument(
            "--matrix", default="acc", type=str, choices=["acc", "loss"], help="validation matrix (default acc)"
        )
        parser.add_argument("--save_interval", default=2, type=int, help="the save interval (default 5)")

    elif problem == 3:

        parser.add_argument("-t", "--task", type=str, choices=["A", "B"], default="A", help="the task in the problem.")

        # path
        # datasets
        parser.add_argument(
            "--data_dir",
            type=str,
            default=os.path.join("hw2_data", "digits"),
            help="the directory to the dataset.",
        )
        parser.add_argument(
            "--output_path",
            type=str,
            default="DANN_pred.csv",
            help="the directory to the output.",
        )
        parser.add_argument(
            "--target_dir",
            type=str,
            default=os.path.join("hw2_data", "digits", "svhn", "data"),
            help="the directory to the target images.",
        )
        # checkpoint
        parser.add_argument("--load", type=str, default=os.path.join("best_checkpoint", "P3", "P3_bestDANN_SVHN.pth"), help="the directory to load the checkpoint")

        # data
        parser.add_argument("-is", "--image_size", type=int, default=28, help="the image size")

        # model
        parser.add_argument(
            "-m", "--model_mode", type=str, default="DANN", choices=["cross", "DANN", "same"], help="the model mode"
        )

        # data loader
        parser.add_argument("--train_batch", default=4096, type=int, help="the training batch size (default 32)")
        parser.add_argument("--test_batch", default=128, type=int, help="the testing batch size (default 32)")

        # optimizer
        parser.add_argument("-lr", "--learning_rate", default=0.00025, type=float, help="the initial learning rate")
        parser.add_argument(
            "-wd", "--weight_decay", default=0.2, type=float, help="the weight decay for L2-regularization"
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
            help="the learning rate patience for decreasing laerning rate (default 5)",
        )

        # training
        parser.add_argument("--epoch", type=int, default=1000, help="the num of epoch (default 1000)")
        parser.add_argument("-seed", "--random_seed", default=888, type=int, help="the seed (default 888)")
        parser.add_argument(
            "--epoch_patience", default=100, type=int, help="the epoch patience for early stopping (default 40)"
        )

        # validation
        parser.add_argument(
            "--matrix", default="acc", type=str, choices=["acc", "loss"], help="validation matrix (default acc)"
        )
        parser.add_argument("--save_interval", default=5, type=int, help="the save interval (default 5)")

    args = parser.parse_args()

    return args
