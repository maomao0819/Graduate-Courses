
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
import argparse

from torch.utils.data import DataLoader
import numpy as np
import random

from models import get_model
from engine import *
from config import *
from functions import *
import data_utils

cfg = Config()

parser = argparse.ArgumentParser()

parser.add_argument("--video_dir")
parser.add_argument("--seg_dir")
parser.add_argument("--bbox_dir")
parser.add_argument("--output_csv")

args = parser.parse_args()

cfg.video_dir = args.video_dir
cfg.seg_dir = args.seg_dir
cfg.bbox_dir = args.bbox_dir
cfg.output_csv = args.output_csv

print('###########################################')
print('batch_size:', cfg.batch_size)
print('init_lr:', cfg.init_lr)
print('num_epochs:', cfg.num_epochs)
print('input_size:', cfg.image_size)
print('frames sample:', cfg.samples)
print('backbone:', cfg.backbone)
print('###########################################')

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # set to False guarantee perfect reproducbility, but hurt performance

model = get_model(cfg.backbone, cfg)
model = model.to(cfg.device)
# print(net)
# exit()

# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=cfg.init_lr, weight_decay=1e-3)
# optimizer = optim.SGD(model.parameters(), lr=cfg.init_lr, momentum=0.9, weight_decay=5e-4)

# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)


train_transform = transforms.Compose([
    SquarePad(),
    # transforms.Resize((int(cfg.image_size), int(cfg.image_size))),
    transforms.RandomResizedCrop((cfg.image_size, cfg.image_size), scale=(0.49, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),]), p=0.2),
    transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur((3, 3), (1.0, 2.0)),]), p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=[0.9, 1.1], contrast=[0.9, 1.1], saturation=[0.9, 1.1]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
valid_transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# train_transform = transforms.Compose([transforms.Resize(int(cfg.image_size*1.1)), transforms.RandomRotation(15), transforms.RandomCrop(
#     int(cfg.image_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.RandomErasing()])
# valid_transform = transforms.Compose([transforms.Resize(cfg.image_size), transforms.ToTensor()])


train_data = data_utils.ttmData(seg_dir=cfg.seg_dir, bbox_dir=cfg.bbox_dir,
                                split='train', cfg=cfg, transform=train_transform)
valid_data = data_utils.ttmData(seg_dir=cfg.seg_dir, bbox_dir=cfg.bbox_dir,
                                split='valid', cfg=cfg, transform=valid_transform)

train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)


train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, cfg)
