import pandas as pd
import os
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import utils
from model import MyImageClassificationNet, Resnet50Model
from dataset import ImageClassificationPredictDataset
import parser
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

args = parser.arg_parse_1_1()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_transform = {
    'val': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

def predict_1_1(model):
    image_dir = args.input_dir
    predict_set = ImageClassificationPredictDataset(root=image_dir, transform=image_transform['val'])
    predict_loader = DataLoader(predict_set, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = model.to(device)
    model = utils.load_checkpoint(args.load, model)
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    filenames = []
    predictions = []
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        tqdm_loop = tqdm((predict_loader), total=len(predict_loader))
        for data, filename in tqdm_loop:
            data = data.to(device)
            output = model(data)['out']
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            filenames.extend(filename)
            predictions.extend(pred.cpu().detach().numpy())
    df = pd.DataFrame(columns=['filename', 'label'])
    df['filename'] = filenames
    df['label'] = np.squeeze(predictions)
    df.to_csv(args.output_path, index=False)
    return df

if __name__=='__main__':
    if args.model_index == 0 or 'Mine_CNN' in args.load:
        model = MyImageClassificationNet(n_classes=50).to(device)
    else:
        model = Resnet50Model(n_classes=50).to(device)
    predict_1_1(model)