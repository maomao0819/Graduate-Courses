import os
import glob
import copy
from tqdm import tqdm
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from mean_iou_evaluate import mean_iou_score
from viz_mask import read_masks
import datetime

import parser
import utils
from model import VGG16_FCN32s, DEEPLAB
from dataset import ImageSegmantationDataset
from test_1_2 import val_1_2, test_1_2

args = parser.arg_parse_1_2()

transform_set = [
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_transform = {
    "train": transforms.Compose(
        [transforms.RandomApply(transform_set, p=0.5), transforms.ToTensor(), transforms.Normalize(mean, std)]
    ),
    "val": transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
}


# Create the dataset.
# transforms.ToTensor() automatically converts PIL images to
# torch tensors with range [0, 1]
training_path = os.path.join(args.data_path, "train")
trainset = ImageSegmantationDataset(root=training_path, transform=image_transform["train"])

validation_path = os.path.join(args.data_path, "validation")
valset = ImageSegmantationDataset(root=validation_path, transform=image_transform["val"])

# # Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)
valset_loader = DataLoader(valset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

def train_save(model, model_name="Model"):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    model.train()  # set training mode
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.power(0.001, 1 / args.epoch))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience)
    trigger_times = 0
    best_val_miou = 0
    best_model_weight = copy.deepcopy(model.state_dict())
    metric = utils.metrix()
    time = datetime.datetime.now()
    Date = f"{time.month}_{time.day}"
    save_checkpoint_root = os.path.join(args.save, model_name, Date)
    for epoch in range(1, args.epoch+1):
        print("Epoch {} / {}".format(epoch, args.epoch - 1))
        print("-" * 20)
        train_loss = 0
        train_correct = 0
        train_miou = 0
        tqdm_loop = tqdm((trainset_loader), total=len(trainset_loader))
        for data, target in tqdm_loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()

            train_loss += batch_loss  # sum up batch loss

            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(target.view_as(pred)).sum().item() / 512 / 512
            train_correct += batch_correct
            batch_miou = mean_iou_score(
                pred.view_as(target).cpu().detach().numpy(), target.cpu().detach().numpy(), log=False
            )
            metric.update(pred.view_as(target).cpu().detach().numpy(), target.cpu().detach().numpy())
            train_miou += batch_miou * float(data.shape[0])
            tqdm_loop.set_description(f"Epoch [{epoch}/{args.epoch-1}]")
            tqdm_loop.set_postfix(loss=batch_loss, acc=float(batch_correct) / float(data.shape[0]), miou=batch_miou)

        train_loss /= len(trainset_loader.dataset)
        train_miou /= len(trainset_loader.dataset)

        if epoch % args.log_interval == 0:
            print(
                "\nTrain set: Average loss: {:.5f}, Accuracy: {}/{} ({:.2f}%), batch_MIoU:{:.5f}, total_MIoU:{:.5f}\n".format(
                    train_loss,
                    train_correct,
                    len(trainset_loader.dataset),
                    100.0 * train_correct / len(trainset_loader.dataset),
                    train_miou,
                    metric.mean_IoU(),
                )
            )
        
        val_loss, val_batch_miou, val_total_miou = val_1_2(model, valset_loader)
        val_miou = val_total_miou
        scheduler.step(val_miou)
        
        if epoch % args.save_interval == 0 and epoch > 0:
            utils.save_checkpoint(
                ("%s/ckpt-%i-%.2f.pth" % (save_checkpoint_root, epoch, val_miou)), model
            )

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_model_weight = copy.deepcopy(model.state_dict())
            trigger_times = 0
            utils.save_checkpoint(("%s/best-%.2f.pth" % (save_checkpoint_root, val_miou)), model)
        else:
            trigger_times += 1

            if trigger_times >= args.epoch_patience:
                print("Early Stop")
                model.load_state_dict(best_model_weight)
                val_1_2(model)
                utils.save_checkpoint(("%s/best-%.2f.pth" % (save_checkpoint_root, val_miou)), model)
                return
        metric.reset()

    # save the final model
    utils.save_checkpoint(("%s/final-%.2f.pth" % (save_checkpoint_root, val_miou)), model)

if __name__=='__main__':
    if args.model_index == 0:
        model_name = 'VGG16_FCN32s'
        VGG16_FCN32s_model = VGG16_FCN32s().to(device)
        train_save(VGG16_FCN32s_model, model_name)
        val_1_2(VGG16_FCN32s_model, valset_loader)
    else:
        model_name = 'DEEPLAB'
        DEEPLAB_model = DEEPLAB().to(device)
        train_save(DEEPLAB_model, model_name)
        val_1_2(DEEPLAB_model, valset_loader)