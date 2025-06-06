import os
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import utils
from model import VGG16_FCN32s, DEEPLAB
from dataset import ImageSegmantationPredictionDataset
import parser
from viz_mask import cls_color
from PIL import Image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

args = parser.arg_parse_1_2()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_transform = {
        "val": transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
    }

def predict_1_2(model):
    image_dir = args.input_dir
    predict_set = ImageSegmantationPredictionDataset(root=image_dir, transform=image_transform['val'])
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
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            filenames.extend(filename)
            prediction_index = pred.cpu().detach().numpy()
            prediction_index = np.squeeze(prediction_index, axis=1)
            prediction_color_shape = list(prediction_index.shape)
            prediction_color_shape.append(3)
            prediction_color_shape = tuple(prediction_color_shape)
            prediction_color = np.zeros(prediction_color_shape)
            for color_key in cls_color:
                prediction_color[prediction_index == color_key] = cls_color[color_key]
            predictions.extend(prediction_color)
    utils.CheckDirectory(os.path.join(args.output_dir, ''))
    for mask_idx in range(len(predictions)):
        # print(np.shape(predictions[mask_idx]))
        prediction = predictions[mask_idx].astype(np.uint8)
        # print(np.shape(prediction))
        mask = Image.fromarray(prediction)
        mask.save(os.path.join(args.output_dir, filenames[mask_idx] + '.png'))
    return 

if __name__=='__main__':
    if args.model_index == 0 or 'VGG' in args.load or 'FCN' in args.load:
        model = VGG16_FCN32s().to(device)
    else:
        model = DEEPLAB().to(device)
    predict_1_2(model)