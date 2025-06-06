import argparse
import os

def arg_parse_1_1():
    parser = argparse.ArgumentParser()

    # Datasets parameters
    parser.add_argument(
        "--data_path",
        default=os.path.join("hw1_data", "hw1_data", "p1_data"),
        type=str,
        help="the root path to data directory",
    )
    parser.add_argument("--workers", default=4, type=int, help="the number of data loading workers (default: 4)")

    # training parameters
    parser.add_argument("--model_index", default=1, type=int, help="the index of model (0: Mine CNN, not 0: Pretrain Resnet, default 1)")
    parser.add_argument("--epoch", default=1000, type=int, help="the num of epoch (default 1000)")
    parser.add_argument("--train_batch", default=64, type=int, help="the training batch size (default 64)")
    parser.add_argument("--test_batch", default=64, type=int, help="the testing batch size (default 64)")
    parser.add_argument("--lr", default=1e-5, type=float, help="the initial learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="the weight decay for L2-regularization")
    parser.add_argument('--random_seed', default=888, type=int, help="the seed (default 888)")
    parser.add_argument("--epoch_patience", default=40, type=int, help="the epoch patience for early stopping (default 40)")
    parser.add_argument(
        "--lr_patience", default=5, type=int, help="the learning rate patience for decreasing laerning rate (default 5)"
    )
    parser.add_argument('--save_interval', default=5, type=int, help="the save interval (default 5)")
    parser.add_argument('--log_interval', default=1, type=int, help="the log interval (default 1)")

    # others
    parser.add_argument("--load", default=os.path.join('checkpoint', 'HW1_1', 'Pretrain_Resnet', 'best.pth'), type=str, help="the path to the directory to load the checkpoint")
    parser.add_argument("--save", default=os.path.join('checkpoint', 'HW1_1'), type=str, help="the path to the directory to save the checkpoint")

    # for test
    parser.add_argument('--input_dir', default=os.path.join("hw1_data", "hw1_data", "p1_data", "val_50"), type=str, help="Input directory to read images")
    parser.add_argument('--output_path', default='predictions.csv', type=str, help="Output directory to save images")

    args = parser.parse_args()

    return args

def arg_parse_1_2():
    parser = argparse.ArgumentParser()

    # Datasets parameters
    parser.add_argument(
        "--data_path",
        default=os.path.join("hw1_data", "hw1_data", "p2_data"),
        type=str,
        help="the root path to data directory",
    )
    parser.add_argument("--workers", default=4, type=int, help="the number of data loading workers (default: 4)")

    # training parameters
    parser.add_argument("--model_index", default=1, type=int, help="the index of model (0: VGG16_FCN32s, 1: DEEPLAB, default 1)")
    parser.add_argument("--epoch", default=1000, type=int, help="the num of epoch (default 1000)")
    parser.add_argument("--train_batch", default=4, type=int, help="the training batch size (default 16)")
    parser.add_argument("--test_batch", default=4, type=int, help="the testing batch size (default 16)")
    parser.add_argument("--lr", default=1e-4, type=float, help="the initial learning rate")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="the weight decay for L2-regularization")
    parser.add_argument('--random_seed', default=888, type=int, help="the seed (default 888)")
    parser.add_argument("--epoch_patience", default=40, type=int, help="the epoch patience for early stopping (default 40)")
    parser.add_argument(
        "--lr_patience", default=5, type=int, help="the learning rate patience for decreasing laerning rate (default 5)"
    )
    parser.add_argument('--save_interval', default=1, type=int, help="the save interval (default 5)")
    parser.add_argument('--log_interval', default=1, type=int, help="the log interval (default 1)")

    # others
    parser.add_argument("--load", default=os.path.join('checkpoint', 'HW1_2', 'DEEPLAB', 'best-0.73.pth'), type=str, help="the path to the directory to load the checkpoint")
    parser.add_argument("--save", default=os.path.join('checkpoint', 'HW1_2'), type=str, help="the path to the directory to save the checkpoint")

    # for test
    parser.add_argument('--input_dir', default=os.path.join("hw1_data", "hw1_data", "p2_data", "validation"), type=str, help="Input directory to read images")
    parser.add_argument('--output_dir', default='pred_mask', type=str, help="Output directory to save images")

    args = parser.parse_args()

    return args