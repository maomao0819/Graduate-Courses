import os
import copy
import numpy as np
import datetime
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import parser
import utils
import model
from dataset import DigitImageDataset
from typing import Dict


def eval(
    args,
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> Dict:

    model.eval()

    # Initialize CrossEntropyLoss function
    epoch_correct = 0.0
    n_batch = len(dataloader)
    batch_pbar = tqdm((dataloader), total=n_batch)
    with torch.no_grad():
        for batch_idx, data in enumerate(batch_pbar, 1):
            image = data[0].to(args.device)
            label = data[1].to(args.device)

            output = model(image)["class"]

            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(label.view_as(pred)).sum().item()
            epoch_correct += batch_correct

            batch_pbar.set_description(f"Batch [{batch_idx}/{n_batch}]")
            batch_pbar.set_postfix(acc=float(batch_correct) / float(label.shape[0]))

    n_data = len(dataloader.dataset)
    acc = epoch_correct / n_data
    return acc


def main(args):

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)


    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    image_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if args.task == "A":
        target_dataset = "svhn"
    else:
        target_dataset = "usps"

    target_dir = os.path.join(args.data_dir, target_dataset)
    testset = DigitImageDataset(dir=target_dir, mode="val", transform=image_transform)
    test_loader = DataLoader(
        testset, batch_size=args.test_batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )

    DANNModel = model.DANN_model().to(args.device)
    DANNModel = utils.load_checkpoint(args.load, DANNModel)
    
    acc = eval(args, DANNModel, test_loader)
    print(acc)

if __name__ == "__main__":
    args = parser.arg_parse(3)
    main(args)
