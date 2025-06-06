
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os

from models import *

from torch.utils.data import DataLoader
import numpy as np
import random

from engine import *
from config import *
import data_utils

cfg = Config()

print('###########################################')
print('batch_size:', cfg.batch_size)
print('init_lr:', cfg.init_lr)
print('num_epochs:', cfg.num_epochs)
# print('backbone:', cfg.backbone)
print('###########################################')

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # set to False guarantee perfect reproducbility, but hurt performance


model = audio_net()
model = model.to(cfg.device)
# print(net)
# exit()
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.init_lr, weight_decay=1e-3)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5)


# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop((cfg.image_size, cfg.image_size), scale=(0.8, 1.0)),
#     transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),]), p=0.2),
#     transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(5),]), p=0.2),
#     transforms.RandomGrayscale(p=0.1),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=[0.9, 1.1], contrast=[0.9, 1.1], saturation=[0.9, 1.1]),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# valid_transform = transforms.Compose([
#     transforms.Resize((cfg.image_size, cfg.image_size)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# train_transform = transforms.Compose([transforms.Resize(int(cfg.image_size*1.1)), transforms.RandomRotation(15), transforms.RandomCrop(
#     int(cfg.image_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.RandomErasing()])
# valid_transform = transforms.Compose([transforms.Resize(cfg.image_size), transforms.ToTensor()])

train_data = data_utils.audioData(root=os.path.join(cfg.data_path), split='train')
valid_data = data_utils.audioData(root=os.path.join(cfg.data_path), split='valid')


train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)


train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, cfg)
