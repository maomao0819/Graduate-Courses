# DISCLAIMER: this is a easy to use + slimmed down + refactored version of the training code used in the ECCV paper: X2Face
# It should give approximately similar results to what is in the paper (e.g. the frontalised unwrapped face
# and that the driving portion of the network transforms this frontalised face into the given view).
# It should also give a good idea of how to train the network.

# (c) Olivia Wiles

import os
import numpy as np
import argparse
import torch
import torchvision
from tensorboardX import SummaryWriter
from torchvision.transforms import ToTensor, Resize, Compose
from torch.autograd import Variable
from UnwrappedFace import UnwrappedFaceWeightedAverage
from tqdm import tqdm

import matplotlib.pyplot as plt

from PIL import Image


def get_image_paths(path):
    # file
    if os.path.isfile(path):
        return [path]
    # dirs
    return [os.path.join(path, img_name) for img_name in os.listdir(path)]


def run_batch(model, source_images, pose_images):
    return model(pose_images, *source_images)


def load_img(file_path, dim):
    img = Image.open(file_path)
    transform = Compose([Resize((dim, dim)), ToTensor()])
    return Variable(transform(img)).cuda()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(opt):
    set_seed(opt.seed)
    os.makedirs(opt.results_folder, exist_ok=True)

    model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=opt.inner_nc, 
        skip_net_backbone=opt.skip_net_backbone, no_skip_net_backbone=opt.no_skip_net_backbone)

    if os.path.exists(opt.load_model):
        checkpoint_file = torch.load(opt.load_model)
        model.load_state_dict(checkpoint_file['state_dict'])

    model = model.cuda()
    model = model.eval()

    source_imgs = get_image_paths(opt.source_path)
    driver_imgs = get_image_paths(opt.driver_path)

    # Driving the source image with the driving sequence
    source_images = []
    for img in source_imgs:
        source_images.append(load_img(img, opt.dim).unsqueeze(0).repeat(len(driver_imgs), 1, 1, 1))

    driver_images = None
    for img in driver_imgs:
        if driver_images is None:
            driver_images = load_img(img, opt.dim).unsqueeze(0)
        else:
            driver_images = torch.cat((driver_images, load_img(img, opt.dim).unsqueeze(0)), 0)

    # Run the model for each
    with torch.no_grad():
        result = run_batch(model, source_images, driver_images)
    results = result.clamp(min=0, max=1)
    plt.axis('off')
    for id, result in enumerate(results):
        plt.imshow(result.detach().cpu().permute(1,2,0).numpy())
        plt.savefig(os.path.join(opt.results_folder, f'plot_{id}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UnwrappedFace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--inner_nc', type=int, default=128)
    parser.add_argument('--source_path', type=str, default='test_images/source/')
    parser.add_argument('--driver_path', type=str, default='test_images/driver/')
    parser.add_argument('--skip_net_backbone', type=str, choices=['unet_3+', 'unet_128', 'unet_256'], default='unet_3+')
    parser.add_argument('--no_skip_net_backbone', type=str, choices=['unet_128', 'unet_256'], default='unet_128')
    parser.add_argument('--load_model', type=str, default='model_weight/unet_3P.pth')
    parser.add_argument('--results_folder', type=str, default='result_images') 
    opt = parser.parse_args()
    main(opt)