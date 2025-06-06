import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

import parser
import utils
from model import MyImageClassificationNet, Resnet50Model
from dataset import ImageClassificationDataset
from test_1_1 import val_1_1, test_1_1

args = parser.arg_parse_1_1()

transform_set = [ 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomPerspective(distortion_scale=0.3, interpolation=2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.RandomRotation(30, interpolation=InterpolationMode.BICUBIC, expand=False)
]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_transform = {
    'train': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomApply(transform_set, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

training_path = os.path.join(args.data_path, 'train_50')
trainset = ImageClassificationDataset(root=training_path, transform=image_transform['train'])

validation_path = os.path.join(args.data_path, 'val_50')
valset = ImageClassificationDataset(root=validation_path, transform=image_transform['val'])

# # Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)
valset_loader = DataLoader(valset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

def train_save(model, model_name='Model'):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    model.train()  # set training mode
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = np.power(0.01, 1/args.epoch))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience)
    trigger_times = 0
    best_val_acc = 0
    best_model_weight = copy.deepcopy(model.state_dict())
    time = datetime.datetime.now()
    Date = f"{time.month}_{time.day}"
    save_checkpoint_root = os.path.join(args.save, model_name, Date)
    for epoch in range(1, args.epoch+1):
        print('Epoch {} / {}'.format(epoch, args.epoch-1))
        print('-' * 20)
        train_loss = 0
        train_correct = 0
        tqdm_loop = tqdm((trainset_loader), total=len(trainset_loader))
        for data, target in tqdm_loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)['out']
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()

            train_loss += batch_loss # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            train_correct += batch_correct
            
            tqdm_loop.set_description(f'Epoch [{epoch}/{args.epoch-1}]')
            tqdm_loop.set_postfix(loss=batch_loss, acc=float(batch_correct) / float(data.shape[0]))
            
        train_loss /= len(trainset_loader.dataset)
        
        if epoch % args.log_interval == 0:
            print('\nTrain set: Average loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)\n'.format(train_loss, train_correct, len(trainset_loader.dataset),
                100. * train_correct / len(trainset_loader.dataset)))

        val_loss, val_acc = val_1_1(model, valset_loader)
        scheduler.step(val_acc)

        if epoch % args.save_interval == 0 and epoch > 0:
            utils.save_checkpoint(
                ("%s/ckpt-%i-%.2f.pth" % (save_checkpoint_root, epoch, val_acc)), model,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weight = copy.deepcopy(model.state_dict())
            trigger_times = 0
            utils.save_checkpoint(("%s/best-%.2f.pth" % (save_checkpoint_root, val_acc)), model)
        else:
            trigger_times += 1

            if trigger_times >= args.epoch_patience:
                print('Early Stop')
                model.load_state_dict(best_model_weight)
                val_1_1(model, valset_loader)
                utils.save_checkpoint(("%s/best-%.2f.pth" % (save_checkpoint_root, val_acc)), model)
                return
    
    # save the final model
    utils.save_checkpoint(("%s/final-%.2f.pth" % (save_checkpoint_root, val_acc)), model)

if __name__=='__main__':
    if args.model_index == 0:
        model_name = 'Mine_CNN'
        model = MyImageClassificationNet(n_classes=50).to(device)
        train_save(model, model_name)
        val_1_1(model, valset_loader)
    else:
        model_name = 'Pretrain_Resnet'
        pretrain_model = Resnet50Model(n_classes=50).to(device)
        train_save(pretrain_model, model_name)
        val_1_1(pretrain_model, valset_loader)
    