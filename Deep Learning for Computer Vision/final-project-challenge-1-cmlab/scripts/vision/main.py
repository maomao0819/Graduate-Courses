
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
import datetime

from torch.utils.data import DataLoader
import numpy as np
import random

from engine import *
from config import *
from functions import *
from models import *
import data_utils

cfg = Config()

print('###########################################')
print("Time stamp:", datetime.datetime.now())
print('batch_size:', cfg.batch_size)
print('init_lr:', cfg.init_lr)
print('num_epochs:', cfg.num_epochs)
print('input_size:', cfg.image_size)
print('frames sample:', cfg.samples)
print('backbone:', cfg.backbone)
print('TTA number:', cfg.tta_num)
print('use_checkpoint:', cfg.use_checkpoint)
print('###########################################')

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # set to False guarantee perfect reproducbility, but hurt performance

model = get_model(cfg.backbone)
model = model.to(cfg.device)
# for name, param in model.named_parameters():
#     if (name == 'conv1.weight') or ('layer1' in name) or ('layer2' in name) or ('layer3' in name):
#         if 'bn' not in name:
#             param.requires_grad = False
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of params: {n_parameters}")

# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
# criterion = nn.losses.FocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=cfg.init_lr, weight_decay=1e-3)

# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3)

if cfg.use_checkpoint and os.path.exists(cfg.model_path):
    print("\nLoading Checkpoint...\n")
    checkpoint = torch.load(cfg.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    cfg.start_epoch = checkpoint['epoch'] + 1

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
# train_transform = transforms.Compose([transforms.Resize(int(cfg.image_size*1.1)), transforms.RandomRotation(15), transforms.RandomCrop(
#     int(cfg.image_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.RandomErasing()])
# valid_transform = transforms.Compose([transforms.Resize(cfg.image_size), transforms.ToTensor()])


train_data = data_utils.ttmData(root=os.path.join(cfg.data_path, 'train'), split='train', transform=train_transform)
valid_data = data_utils.ttmData(root=os.path.join(cfg.data_path, 'train'), split='valid', transform=valid_transform)

train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)


train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, cfg)
# valid_model(model, valid_loader, criterion, cfg)
