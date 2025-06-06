import os
import numpy as np
import torch

import torchvision.utils as vutils
from tqdm import trange
import parser
import utils
import model


def get_DCGAN(verbose=False):

    # Create the generator
    net_G = model.DCGAN_Generator().to(args.device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    net_G.apply(model.weights_init)

    # Print the model
    if verbose:
        print(net_G)

    # Create the Discriminator
    net_D = model.DCGAN_Discriminator().to(args.device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    net_D.apply(model.weights_init)

    # Print the model
    if verbose:
        print(net_D)

    return net_G, net_D


def get_my_GAN(verbose=False):

    # Create the generator
    net_G = model.My_Generator(n_latent=100, n_feature_map=64, n_channel=3).to(args.device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    net_G.apply(model.weights_init)

    # Print the model
    if verbose:
        print(net_G)

    # Create the Discriminator
    net_D = model.My_Discriminator(n_channel=3, n_feature_map=64).to(args.device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    net_D.apply(model.weights_init)

    # Print the model
    if verbose:
        print(net_D)

    return net_G, net_D


def get_GAN(task="A", verbose=False):
    if "A" in task:
        return get_DCGAN(verbose)
    else:
        # return get_DCGAN(verbose)
        return get_my_GAN(verbose)


def generate(
    net_G: torch.nn.Module,
    noise: torch,
    output_dir: str,
):
    net_G.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        fake_data_valid = net_G(noise).detach()
        fake_data_valid = fake_data_valid * 0.5 + 0.5  # [0 ~ 1]
        data_pbar = trange(fake_data_valid.size(0), desc="data")
        for fake_data_idx in data_pbar:
            vutils.save_image(
                fake_data_valid[fake_data_idx], os.path.join(output_dir, "{}.png".format(fake_data_idx)), padding=0
            )
            data_pbar.set_description(f"Epoch [{fake_data_idx+1}/{fake_data_valid.size(0)}]")
    return


def main(args):

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    n_latent = 100
    n_output_image = 1000

    net_G, _ = get_GAN(task=args.task)
    net_G = utils.load_checkpoint(args.loadG, net_G)

    fixed_noise = torch.randn(n_output_image, n_latent, 1, 1, device=args.device)

    generate(net_G, fixed_noise, output_dir=args.output_dir)


if __name__ == "__main__":
    args = parser.arg_parse(1)

    main(args)
