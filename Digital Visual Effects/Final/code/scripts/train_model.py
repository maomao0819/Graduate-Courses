# DISCLAIMER: this is a easy to use + slimmed down + refactored version of the training code used in the ECCV paper: X2Face
# It should give approximately similar results to what is in the paper (e.g. the frontalised unwrapped face
# and that the driving portion of the network transforms this frontalised face into the given view).
# It should also give a good idea of how to train the network.

# (c) Olivia Wiles

from VoxCelebData_withmask import VoxCeleb

import shutil
import os
import numpy as np
import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.transforms import ToTensor, Resize, Compose
import torch.optim as optim
from torch.autograd import Variable
from UnwrappedFace import UnwrappedFaceWeightedAverage
from tqdm import tqdm
import gpustat
from find_gpus import device, gpu_ids
from SSIM_Loss import SSIMLoss

parser = argparse.ArgumentParser(description='UnwrappedFace')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--sampler_lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--threads', type=int, default=8, help='Num Threads')
# parser.add_argument('--batchSize', type=int, default=64, help='Batch Size')
parser.add_argument('--SSIMLoss', action="store_true")
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--num_views', type=int, default=2, help='Num views')
parser.add_argument('--copy_weights', type=bool, default=False)
parser.add_argument('--model_type', type=str, default='UnwrappedFaceSampler_from1view')
parser.add_argument('--skip_net_backbone', type=str, choices=['unet_3+', 'unet_128', 'unet_256'], default='unet_3+')
parser.add_argument('--no_skip_net_backbone', type=str, choices=['unet_128', 'unet_256'], default='unet_128')
parser.add_argument('--inner_nc', type=int, default=128)
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--old_model', type=str, default='')
parser.add_argument('--results_folder', type=str, default='results/')  # Where temp results will be stored
parser.add_argument('--model_epoch_path', type=str, default='models/', help='Location to save to')
opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

writer = SummaryWriter(opt.results_folder)

opt.model_epoch_path = opt.model_epoch_path

model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3,
                                     inner_nc=opt.inner_nc, skip_net_backbone=opt.skip_net_backbone, 
                                     no_skip_net_backbone=opt.no_skip_net_backbone)

if opt.copy_weights:
    checkpoint_file = torch.load(opt.old_model)
    model.load_state_dict(checkpoint_file['state_dict'])
    opt.model_epoch_path = opt.model_epoch_path + 'copyWeights'
    del checkpoint_file


criterion_L1 = nn.L1Loss()
criterion_SSIM = SSIMLoss()

# Check if there are multiple GPUs
if torch.cuda.device_count() > 1:
    # model = nn.DataParallel(model)
    model = nn.DataParallel(model, device_ids=gpu_ids)

model = model.cuda()

criterion_L1 = criterion_L1.cuda()
criterion_SSIM = criterion_SSIM.cuda()
parameters = [{'params': model.parameters()}]
optimizer = optim.SGD(parameters, lr=opt.lr, momentum=0.9)


def run_batch(imgs):
    return model(imgs[1].cuda(), (imgs[0].cuda())), imgs


def get_unwrapped(imgs):
    if isinstance(model, nn.DataParallel):
        return model.module.get_unwrapped_oneimage(imgs[0].cuda())
    return model.get_unwrapped_oneimage(imgs[0].cuda())


def train(epoch, num_views):
    train_set = VoxCeleb(num_views, epoch, 1, dim=opt.dim)

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                      batch_size=opt.batch_size, shuffle=True)

    epoch_train_loss = 0

    model.train()
    # for iteration, batch in enumerate(training_data_loader, 1):
    iteration = 0
    with tqdm(training_data_loader, unit="batch") as tepoch:
        for batch in tepoch:
            iteration += 1
            tepoch.set_description(f"Epoch {epoch}")

            result, inputs = run_batch(batch[0])

            loss_L1 = criterion_L1(result, inputs[opt.num_views-1].cuda())
            loss_SSIM = criterion_SSIM(result, inputs[opt.num_views-1].cuda())
            loss = loss_L1 + loss_SSIM if opt.SSIMLoss else loss_L1

            optimizer.zero_grad()
            epoch_train_loss += loss.data.item()
            loss.backward()
            optimizer.step()
            loss_mean = epoch_train_loss / iteration
            if iteration % 1000 == 0 or iteration == 1:
                for i in range(0, len(inputs)):
                    input = inputs[i]
                    if input.size(1) == 2:
                        writer.add_image('Train/img_dim%d_%d1' % (i, iteration), input[:, 0:1, :, :].data.cpu(), epoch)
                        writer.add_image('Train/img_dim%d_%d2' % (i, iteration), input[:, 1:2, :, :].data.cpu(), epoch)
                    else:
                        writer.add_image('Train/img%d_%d1' % (i, iteration), input.data.cpu()[i], epoch)

                    writer.add_image('Train/result%d' % (iteration), result.data.cpu()[i], epoch)
                    writer.add_image('Train/gt%d' % (iteration), inputs[opt.num_views-1].data.cpu()[i], epoch)

                    unwrapped = get_unwrapped(batch[0])
                    writer.add_image('Train/unwrapped%d' % (iteration), unwrapped.data.cpu()[i], epoch)

            tepoch.set_postfix(loss=loss_mean)
            # print("===> Train Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration,
            #       len(training_data_loader), loss_mean))

            # if iteration == 2000:  # So we can see faster what's happening
            #     break
    return epoch_train_loss / iteration


def val(epoch, num_views):
    val_set = VoxCeleb(num_views, 0, 2, dim=opt.dim)

    validation_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads,
                                        batch_size=opt.batch_size, shuffle=False)

    model.eval()
    epoch_val_loss = 0

    # for iteration, batch in enumerate(validation_data_loader, 1):

    iteration = 0
    with tqdm(validation_data_loader, unit="batch") as tepoch:
        for batch in tepoch:
            iteration += 1
            tepoch.set_description(f"Epoch {epoch}")
            result, inputs = run_batch(batch[0])
            loss_L1 = criterion_L1(result, inputs[opt.num_views-1].cuda())
            loss_SSIM = criterion_SSIM(result, inputs[opt.num_views-1].cuda())
            loss = loss_L1 + loss_SSIM if opt.SSIMLoss else loss_L1
            epoch_val_loss += loss.data.item()
            loss_mean = epoch_val_loss / iteration

            if iteration % 1000 == 0 or iteration == 1:
                for i in range(0, len(inputs)):
                    input = inputs[i]
                    if input.size(1) == 2:
                        writer.add_image('Val/img_dim%d_%d1' % (i, iteration), input[:, 0:1, :, :].data.cpu(), epoch)
                        writer.add_image('Val/img_dim%d_%d2' % (i, iteration), input[:, 1:2, :, :].data.cpu(), epoch)
                    else:
                        writer.add_image('Val/img%d_%d1' % (i, iteration), input.data.cpu()[i], epoch)

                writer.add_image('Val/result%d' % (iteration), result.data.cpu()[i], epoch)
                writer.add_image('Val/gt%d' % (iteration), inputs[opt.num_views-1].data.cpu()[i], epoch)
                unwrapped = get_unwrapped(batch[0])
                writer.add_image('Val/unwrapped%d' % (iteration), unwrapped.data.cpu()[i], epoch)

            tepoch.set_postfix(loss=loss_mean)
            # print("===> Val Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration,
            #                                                        len(validation_data_loader), loss_mean))

            # if iteration == 2000:  # So we can see faster what's happening
            #     break

    return epoch_val_loss / iteration


def checkpoint(model, epoch):
    dict = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

    model_out_path = "{}model_epoch_{}.pth".format(opt.model_epoch_path, epoch)

    if not (os.path.exists(opt.model_epoch_path)):
        os.makedirs(opt.model_epoch_path)
    torch.save(dict, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    for i in range(0, epoch-1):
        if os.path.exists("{}model_epoch_{}.pth".format(opt.model_epoch_path, i)):
            os.remove("{}model_epoch_{}.pth".format(opt.model_epoch_path, i))


if __name__ == "__main__":
    if opt.copy_weights:
        checkpoint_file = torch.load(opt.old_model)
        model.load_state_dict(checkpoint_file['state_dict'])

    start_epoch = opt.start_epoch
    for epoch in range(start_epoch, 3000):
        if epoch > 0:
            checkpoint_file = torch.load("{}model_epoch_{}.pth".format(opt.model_epoch_path, epoch-1))
            model.load_state_dict(checkpoint_file['state_dict'])
            optimizer.load_state_dict(checkpoint_file['optimizer'])

        tloss = train(epoch, opt.num_views)
        with torch.no_grad():
            vloss = val(epoch, opt.num_views)

        writer.add_scalars('TrainVal/loss', {'train': tloss, 'val': vloss}, epoch)
        checkpoint(model, epoch)
