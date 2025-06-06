import argparse
import os
import torch


def boolean_string(str):
    return ("t" in str) or ("T" in str)

def arg_parse(problem=1):
    parser = argparse.ArgumentParser()

    # data loader
    parser.add_argument("--workers", default=4, type=int, help="the number of data loading workers (default: 4)")

    # model
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("-seed", "--random_seed", default=888, type=int, help="the seed (default 888)")

    if problem == 1:

        # path
        # datasets
        parser.add_argument(
            "--image_dir",
            type=str,
            default=os.path.join('hw3_data', 'p1_data', 'val'),
            help="the directory to the dataset.",
        )

        parser.add_argument(
            "--id2label_path",
            type=str,
            default=os.path.join('hw3_data', 'p1_data', 'id2label.json'),
            help="the path to the id2label.",
        )

        parser.add_argument(
            "--predict_path",
            type=str,
            default="zero_shot_prediction.csv",
            help="the path to the output.",
        )

        # data
        parser.add_argument("--prompt_text_type", type=int, default=1, help="the prompt text type")

        # data loader
        parser.add_argument("--batch", default=32, type=int, help="the training batch size (default 32)")

        parser.add_argument("--analysis_top5", type=boolean_string, default=False, help="analysis top5")


    elif problem == 2:
        # path
        # datasets
        parser.add_argument(
            "--data_dir",
            type=str,
            default=os.path.join("hw3_data", "p2_data"),
            help="the directory to the dataset.",
        )
        parser.add_argument(
            "--tokenizer_path",
            type=str,
            default=os.path.join("hw3_data", "caption_tokenizer.json"),
            help="the directory to the dataset.",
        )
        
        parser.add_argument(
            "--pred_path",
            type=str,
            default="image_caption_prediction.json",
            help="the path to the output.",
        )

        parser.add_argument(
            "--load",
            type=str,
            default=os.path.join("best_checkpoint", "P2", "best.pth"),
            help="the directory to load the checkpoint",
        )

        # checkpoint
        parser.add_argument(
            "--save",
            type=str,
            default=os.path.join("checkpoint", f"p{problem}"),
            help="the directory to save the checkpoint",
        )
        
        # model
        parser.add_argument("--freeze", default=True, type=boolean_string, help="freeze ViT")
        parser.add_argument("--n_layers", default=6, type=int, help="the number of layers (default 2)")
        parser.add_argument("--d_model", default=1024, type=int, help="the dimention of feature (default 512)")
        parser.add_argument("--d_ff", default=2048, type=int, help="the dimention of ff (default 2048)")
        parser.add_argument("--n_heads", default=8, type=int, help="the number of heads (default 8)")
        parser.add_argument("--dropout", default=0.1, type=float, help="dropout (default 2)")
        
        # data loader
        parser.add_argument("--train_batch", default=24, type=int, help="the training batch size (default 32)")
        parser.add_argument("--test_batch", default=8, type=int, help="the testing batch size (default 32)")

        # optimizer
        parser.add_argument("-lr", "--learning_rate", default=5e-5, type=float, help="the initial learning rate")
        parser.add_argument(
            "-wd", "--weight_decay", default=0, type=float, help="the weight decay for L2-regularization"
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
            default=2,
            type=int,
            help="the learning rate patience for decreasing laerning rate (default 10)",
        )
        parser.add_argument(
            "--scheduler_type",
            default="reduce",
            type=str,
            choices=["reduce", "exponential", "None"],
            help="type of scheduler (ReduceLROnPlateau, exponentialLR)",
        )


        # training
        parser.add_argument("--epoch", type=int, default=1000, help="the num of epoch (default 1000)")
        parser.add_argument(
            "--epoch_patience", default=200, type=int, help="the epoch patience for early stopping (default 40)"
        )

        # validation
        parser.add_argument(
            "--matrix", default="loss", type=str, choices=["acc", "loss"], help="validation matrix (default acc)"
        )
        parser.add_argument("--save_interval", default=2, type=int, help="the save interval (default 5)")

    elif problem == 3:
        # path
        # datasets
        parser.add_argument(
            "--data_dir",
            type=str,
            default=os.path.join("hw3_data", "p2_data", "images", 'val'),
            help="the directory to the dataset.",
        )
        parser.add_argument(
            "--tokenizer_path",
            type=str,
            default=os.path.join("hw3_data", "caption_tokenizer.json"),
            help="the directory to the dataset.",
        )
        
        parser.add_argument(
            "--out_path",
            type=str,
            default="image_caption_prediction.json",
            help="the path to the output.",
        )

        parser.add_argument(
            "--load",
            type=str,
            default=os.path.join("best_checkpoint", "P2", "best.pth"),
            help="the directory to load the checkpoint",
        )
        
        # model
        parser.add_argument("--freeze", default=True, type=boolean_string, help="freeze ViT")
        parser.add_argument("--n_layers", default=6, type=int, help="the number of layers (default 2)")
        parser.add_argument("--d_model", default=1024, type=int, help="the dimention of feature (default 512)")
        parser.add_argument("--d_ff", default=2048, type=int, help="the dimention of ff (default 2048)")
        parser.add_argument("--n_heads", default=8, type=int, help="the number of heads (default 8)")
        parser.add_argument("--dropout", default=0.1, type=float, help="dropout (default 2)")
        
        # data loader
        parser.add_argument("--train_batch", default=24, type=int, help="the training batch size (default 32)")
        parser.add_argument("--test_batch", default=8, type=int, help="the testing batch size (default 32)")
        

    args = parser.parse_args()

    return args
