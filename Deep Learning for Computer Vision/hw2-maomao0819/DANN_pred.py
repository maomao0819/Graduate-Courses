import os
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import parser
import model
from dataset import DigitImagePredictDataset
from typing import Dict, List
import pandas as pd
import utils

def predict(
    args,
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> Dict:

    model.eval()
    n_batch = len(dataloader)
    image_names = []
    predictions = []
    batch_pbar = tqdm((dataloader), total=n_batch, desc="Batch")
    with torch.no_grad():
        for data in batch_pbar:
            alpha = 0
            images = data[0].to(args.device)
            basenames = data[1]
            # train on source domain
            output = model(images, alpha=alpha)
            pred_class = output["class"].max(1, keepdim=True)[1]  # get the index of the max log-probability
            image_names.extend(basenames)
            predictions.extend(pred_class.detach().cpu().numpy())
    df = pd.DataFrame(columns=['image_name', 'label'])
    df['image_name'] = image_names
    df['label'] = np.squeeze(predictions)
    df = df.sort_values(by=['image_name']).reset_index(drop=True)
    df.to_csv(args.output_path, index=False)
    return pred_class

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
     
    testset = DigitImagePredictDataset(dir=os.path.join(args.target_dir), transform=image_transform)
    testloader = DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    DANNModel = model.DANN_model().to(args.device)
    DANNModel = utils.load_checkpoint(args.load, DANNModel)
    predict(args, DANNModel, testloader)

if __name__ == "__main__":
    args = parser.arg_parse(3)
    main(args)
