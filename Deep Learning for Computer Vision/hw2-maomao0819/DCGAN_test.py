import os
import numpy as np
import torch

import parser
import utils
import model
from face_recog import face_recog
from fid_score import calculate_fid_given_paths
from DCGAN_generate import generate


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

def eval_one_epoch(
    args,
    net_G: torch.nn.Module,
    noise: torch,
    output_dir: str,
):
    net_G.eval()
    val_path = os.path.join(args.data_dir, "val")
    os.makedirs(output_dir, exist_ok=True)
    # Check how the generator is doing by saving G's output on fixed_noise_show
    generate(net_G, noise, output_dir)

    Acc_HOG = face_recog(output_dir)
    paths = [val_path, output_dir]
    Acc_FID = calculate_fid_given_paths(
        paths=paths, batch_size=args.test_batch, device=args.device, num_workers=args.workers
    )
    print("\nAcc_HOG:", Acc_HOG, "\tAcc_FID:", Acc_FID)
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

    eval_one_epoch(args, net_G, fixed_noise, output_dir=args.output_dir)


if __name__ == "__main__":
    args = parser.arg_parse(1)
    main(args)
