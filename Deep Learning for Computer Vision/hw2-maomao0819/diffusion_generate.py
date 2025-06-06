import os
import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import TensorDataset, DataLoader
import parser
import utils
import model
from tqdm import tqdm

def generate_images(
    args,
    Unet_model: torch.nn.Module,
    noisyImage: torch,
    output_dir,
):

    Unet_model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        n_label = 10
        labelList = []
        test_label_batch = int(len(noisyImage) // n_label)
        for label in range(n_label - 1):
            labelList.extend([label] * test_label_batch)
        labelList.extend([n_label - 1] * (len(noisyImage) - len(labelList)))
        
        labels = torch.Tensor(labelList).long().to(args.device)
        dataset = TensorDataset(labels, noisyImage) # create your datset
        dataloader = DataLoader(dataset, batch_size=args.test_batch, shuffle=False)

        sampler = model.GaussianDiffusionSampler(Unet_model, args.beta_1, args.beta_T, args.time_length, w=args.w).to(args.device)
        batch_pbar = tqdm((dataloader), total=len(dataloader), desc="Batch")
        
        image_count = 0
        for labels, noisyImage in batch_pbar: 
            # Sampled from standard normal distribution
            sampled_images = sampler(noisyImage, labels)
            sampled_images = sampled_images * 0.5 + 0.5  # [0 ~ 1]
            for image_idx in range(sampled_images.size(0)):
                vutils.save_image(sampled_images[image_idx], os.path.join(output_dir, f'{labels[image_idx]}_{(image_count % test_label_batch):03d}.png'), padding=0)
                image_count += 1
    return 

def generate_grid(
    args,
    Unet_model: torch.nn.Module,
    output_dir,
):
    n_label = 10
    n_each_image = 10
    noisyImage = torch.randn(size=[n_each_image * n_label, 3, args.image_size, args.image_size], device=args.device)
    Unet_model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        labelList = []
        test_label_batch = int(len(noisyImage) // n_label)
        for label in range(n_label - 1):
            labelList.extend([label] * test_label_batch)
        labelList.extend([n_label - 1] * (len(noisyImage) - len(labelList)))

        labels = torch.Tensor(labelList).long().to(args.device)

        first_image = noisyImage[0].unsqueeze(0)
        sampled_first_images = first_image
        n_step = 6
        for step_idx in range(n_step):
            t = step_idx * int(args.time_length / (n_step - 1))
            sampler = model.GaussianDiffusionSampler(Unet_model, args.beta_1, args.beta_T, t, w=args.w).to(args.device)
            # Sampled from standard normal distribution
            sampled_images = sampler(noisyImage, labels)
            sampled_images = sampled_images * 0.5 + 0.5  # [0 ~ 1]
            sampled_first_images = torch.cat((sampled_first_images, sampled_images[0].unsqueeze(0)))
            if step_idx == (n_step - 1):
                vutils.save_image(sampled_images, os.path.join(output_dir, 'grid.png'), nrow=test_label_batch, padding=0)
        sampled_first_images = sampled_first_images[1:]
        vutils.save_image(sampled_first_images, os.path.join(output_dir, 'grid_time.png'), nrow=test_label_batch, padding=0)

    return 

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    Unet_model = model.UNet(time_length=args.time_length, n_labels=10, channel=args.channel, channel_multiply=args.channel_multiply,
        n_residual_blocks=args.n_residual_blocks, dropout=args.dropout).to(args.device)

    n_labels = 10
    n_each_image = 100
    noisyImage = torch.randn(size=[n_each_image * n_labels, 3, args.image_size, args.image_size], device=args.device)
    Unet_model = utils.load_checkpoint(args.load, Unet_model)

    generate_images(args, Unet_model, noisyImage, output_dir=args.output_dir)
    # generate_grid(args, Unet_model, output_dir=args.output_dir)

if __name__ == "__main__":
    args = parser.arg_parse(2)
    main(args)