import os
import copy
import numpy as np
import datetime
from tqdm import tqdm
import torch

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm, trange
import parser
import utils
import model
from dataset import FaceImageDataset
from typing import Dict, List
from face_recog import face_recog
from fid_score import calculate_fid_given_paths


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


def train_one_epoch(
    args,
    net_D: torch.nn.Module,
    net_G: torch.nn.Module,
    dataloader: DataLoader,
    optimizer_D: torch.optim,
    optimizer_G: torch.optim,
    D_losses: List,
    G_losses: List,
    D_xs: List,
    D_G_z_fakes: List,
    D_G_z_reals: List,
    n_latent: int = 100,
    epoch_idx: int = 1,
) -> Dict:

    net_D.train()
    net_G.train()

    # Initialize BCELoss function
    criterion = torch.nn.BCELoss()
    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0
    if ("A" in args.task) == False and (epoch_idx + 1) % args.reverse_train == 0:
        real_label = 0.3
        fake_label = 0.7
    epoch_error_D = 0.0
    epoch_D_x = 0.0
    epoch_D_G_z_fake = 0.0
    epoch_error_G = 0.0
    epoch_D_G_z_real = 0.0
    n_batch = len(dataloader)
    batch_pbar = tqdm((dataloader), total=n_batch)
    d_steps = 1 if "A" in args.task else args.D_steps
    g_steps = 1 if "A" in args.task else args.G_steps
    for batch_idx, data in enumerate(batch_pbar, 1):
        real_data = data.to(args.device)
        # Format batch
        batch_size = data.size(0)
        real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=args.device)
        fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=args.device)

        if ("A" in args.task) == False and args.label_smooth == True:
            label_smooth = torch.clip(torch.randn_like(real_labels, device=args.device) * 0.2 / (epoch_idx * 0.2 + 1), min=-0.2, max=0.2)
            real_labels = real_labels - label_smooth
            label_smooth = torch.clip(torch.randn_like(fake_labels, device=args.device) * 0.2 / (epoch_idx * 0.2 + 1), min=0, max=0.3)
            fake_labels = fake_labels + label_smooth

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        batch_error_D = 0.0
        batch_D_x = 0.0
        batch_D_G_z_fake = 0.0
        fake_datas = []
        for D_index in range(d_steps):
            net_D.zero_grad()
            # Forward pass real batch through D
            if ("A" in args.task) == False and args.image_noise == True:
                image_noise = torch.clip(torch.randn_like(real_data, device=args.device) * 0.05 / (epoch_idx * 0.2 + 1), min=-0.05, max=0.05)
                real_data = real_data + image_noise
            output = net_D(real_data).view(-1)
            # Calculate loss on all-real batch
            err_D_real = criterion(output, real_labels)
            # Calculate gradients for D in backward pass

            err_D_real.backward()
            D_x = output.mean().item()
            batch_D_x += D_x

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, n_latent, 1, 1, device=args.device)
            # Generate fake image batch with G
            fake_data = net_G(noise)
            if ("A" in args.task) == False and args.image_noise == True:
                image_noise = torch.clip(torch.randn_like(fake_data, device=args.device) * 0.05 / (epoch_idx * 0.2 + 1), min=-0.05, max=0.05)
                fake_data = fake_data + image_noise
            fake_datas.append(fake_data)
            # Classify all fake batch with D
            output = net_D(fake_data.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            error_D_fake = criterion(output, fake_labels)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients

            error_D_fake.backward()
            D_G_z_fake = output.mean().item()
            batch_D_G_z_fake += D_G_z_fake
            # Compute error of D as sum over the fake and the real batches
            error_D = err_D_real + error_D_fake
            batch_error_D += error_D.item()
            # Update D
            optimizer_D.step()
        batch_error_D /= args.D_steps
        batch_D_x /= args.D_steps
        batch_D_G_z_fake /= args.D_steps
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        batch_error_G = 0.0
        batch_D_G_z_real = 0.0
        for G_index in range(g_steps):
            net_G.zero_grad()
            # Generate batch of latent vectors
            if len(fake_datas):
                fake_data = fake_datas.pop(0)
            else:
                noise = torch.randn(batch_size, n_latent, 1, 1, device=args.device)
                # Generate fake image batch with G
                fake_data = net_G(noise)
                if ("A" in args.task) == False and args.image_noise == True:
                    image_noise = torch.clip(torch.randn_like(fake_data, device=args.device) * 0.05 / (epoch_idx * 0.2 + 1), min=-0.05, max=0.05)
                    fake_data = fake_data + image_noise
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = net_D(fake_data).view(-1)
            # Calculate G's loss based on this output
            error_G = criterion(output, real_labels)
            # Calculate gradients for G
            error_G.backward()
            D_G_z_real = output.mean().item()
            batch_D_G_z_real += D_G_z_real
            batch_error_G += error_G.item()
            # Update G
            optimizer_G.step()
        batch_error_G /= args.G_steps
        batch_D_G_z_real /= args.G_steps

        # Save Losses for plotting later
        G_losses.append(batch_error_G)
        D_losses.append(batch_error_D)
        D_xs.append(batch_D_x)
        D_G_z_fakes.append(batch_D_G_z_fake)
        D_G_z_reals.append(batch_D_G_z_real)
        epoch_error_D += batch_error_D
        epoch_D_x += batch_D_x
        epoch_D_G_z_fake += batch_D_G_z_fake
        epoch_error_G += batch_error_G
        epoch_D_G_z_real += batch_D_G_z_real

        batch_pbar.set_description(f"Batch [{batch_idx}/{n_batch}]")
        # Output training stats
        batch_pbar.set_postfix(
            Loss_D=f"{batch_error_D:.4f}",
            loss_G=f"{batch_error_G:.4f}",
            D_x=f"{batch_D_x:.4f}",
            D_G_z=f"{batch_D_G_z_fake:.4f} / {batch_D_G_z_real:.4f}",
        )

    performance = {}
    n_data = len(dataloader.dataset)
    performance["epoch_error_D"] = epoch_error_D / n_data
    performance["epoch_error_G"] = epoch_error_G / n_data
    performance["epoch_D_x"] = epoch_D_x / n_batch
    performance["epoch_D_G_z_fake"] = epoch_D_G_z_fake / n_batch
    performance["epoch_D_G_z_real"] = epoch_D_G_z_real / n_batch
    return performance, D_losses, G_losses


def valid_one_epoch(
    args,
    net_G: torch.nn.Module,
    noise: torch,
):
    net_G.eval()
    val_path = os.path.join(args.data_dir, "val")
    output_path = os.path.join(args.output_dir, args.task)
    # Check how the generator is doing by saving G's output on fixed_noise_show
    with torch.no_grad():
        fake_data_valid = net_G(noise).detach()
        fake_data_valid = fake_data_valid * 0.5 + 0.5  # [0 ~ 1]
        for fake_data_idx in range(fake_data_valid.size(0)):
            vutils.save_image(fake_data_valid[fake_data_idx], os.path.join(output_path, "{}.png".format(fake_data_idx)), padding=0)
    Acc_HOG = face_recog(output_path)
    paths = [val_path, output_path]
    Acc_FID = calculate_fid_given_paths(
        paths=paths, batch_size=args.test_batch, device=args.device, num_workers=args.workers
    )
    return Acc_HOG, Acc_FID


def update_scheduler(scheduler_type, scheduler, matrix):
    if scheduler == None:
        return
    if scheduler_type == "exponential":
        scheduler.step()
    elif scheduler_type == "reduce":
        scheduler.step(matrix)
    return


def main(args):
    time = datetime.datetime.now()
    Date = f"{time.month}_{time.day}"

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    transform_set = [
        transforms.RandomHorizontalFlip(),
        # transforms.GaussianBlur(args.image_size / 2 - 1, sigma=(0.05, 0.1)),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomPerspective(distortion_scale=0.3, interpolation=2),
        # transforms.RandomResizedCrop(size=args.image_size, scale=(0.9, 1.0), ratio=(0.9, 1.0)),
        # # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(3, interpolation=InterpolationMode.BICUBIC, expand=False),
    ]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    image_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.RandomApply(transform_set, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    training_dir = os.path.join(args.data_dir, "train")
    trainset = FaceImageDataset(dir=training_dir, transform=image_transform)

    # # Use the torch dataloader to iterate through the dataset
    trainset_loader = DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )

    n_latent = 100
    n_output_image = 1000

    net_G, net_D = get_GAN(task=args.task)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    n_image_show = 32
    fixed_noise_show = torch.randn(n_image_show, n_latent, 1, 1, device=args.device)
    fixed_noise_valid = torch.randn(n_output_image, n_latent, 1, 1, device=args.device)

    # Setup Adam optimizers for both G and D
    if "A" in args.task:
        learning_rate_D = 0.0002
        learning_rate_G = 0.0002
        optimizer_D = torch.optim.Adam(net_D.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))
        optimizer_G = torch.optim.Adam(net_G.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
    elif args.optimizer_type == "SGD":
        optimizer_D = torch.optim.SGD(net_D.parameters(), lr=args.learning_rate * 10, weight_decay=args.weight_decay)
        optimizer_G = torch.optim.SGD(net_G.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adam":
        optimizer_D = torch.optim.Adam(
            net_D.parameters(), lr=args.learning_rate * 10, betas=(0.5, 0.999), weight_decay=args.weight_decay
        )
        optimizer_G = torch.optim.Adam(
            net_G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=args.weight_decay
        )
    elif args.optimizer_type == "AdamW":
        optimizer_D = torch.optim.AdamW(
            net_D.parameters(),
            lr=args.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay,
        )
        optimizer_G = torch.optim.AdamW(
            net_G.parameters(),
            lr=args.learning_rate * 3,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer_D = torch.optim.SGD(net_D.parameters(), lr=args.learning_rate * 5, weight_decay=args.weight_decay)
        optimizer_G = torch.optim.AdamW(
            net_G.parameters(),
            lr=args.learning_rate,
            betas=(0.3, 0.9),
            eps=1e-08,
            weight_decay=args.weight_decay,
            amsgrad=False,
        )

    if "A" in args.task:
        scheduler_D = None
        scheduler_G = None
    elif args.scheduler_type == "exponential":
        scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=np.power(0.01, 1 / args.epoch))
        scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=np.power(0.01, 1 / args.epoch))
    elif args.scheduler_type == "reduce":
        scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_D, mode="min" if args.matrix == "loss" else "min", factor=0.9, patience=args.lr_patience
        )
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_G, mode="min" if args.matrix == "loss" else "min", factor=0.9, patience=args.lr_patience
        )
    else:
        scheduler_D = None
        scheduler_G = None

    D_losses = []
    G_losses = []
    D_xs = []
    D_G_z_fakes = []
    D_G_z_reals = []
    best_error = np.inf
    best_fid = np.inf
    best_model_weight_D = copy.deepcopy(net_D.state_dict())
    best_model_weight_G = copy.deepcopy(net_G.state_dict())
    epoch_pbar = trange(args.epoch, desc="Epoch")
    trigger_times = 0
    # For each epoch
    for epoch in epoch_pbar:
        performance, D_losses, G_losses = train_one_epoch(
            args,
            net_D,
            net_G,
            trainset_loader,
            optimizer_D,
            optimizer_G,
            D_losses,
            G_losses,
            D_xs,
            D_G_z_fakes,
            D_G_z_reals,
            n_latent=100,
            epoch_idx=epoch
        )

        net_D.eval()
        net_G.eval()
        Acc_HOG, Acc_FID = valid_one_epoch(args, net_G, fixed_noise_valid)
        print("\nAcc_HOG:", Acc_HOG, "\tAcc_FID:", Acc_FID)
        # Check how the generator is doing by saving G's output on fixed_noise_show
        if epoch % args.save_interval == 0:
            with torch.no_grad():
                fake_data_show = net_G(fixed_noise_show).detach()
                fake_data_show = fake_data_show * 0.5 + 0.5  # [0 ~ 1]
            path = os.path.join(args.save, args.task, Date, f"{epoch+1}")
            utils.save_checkpoint(os.path.join(path, "D.pth"), net_D)
            utils.save_checkpoint(os.path.join(path, "G.pth"), net_G)
            vutils.save_image(
                vutils.make_grid(fake_data_show, nrow=8), os.path.join(path, "grid.png")
            )
            utils.plot_error(D_losses, G_losses, D_xs, D_G_z_fakes, D_G_z_reals, path)

        total_error = performance["epoch_error_D"] + performance["epoch_error_G"]

        if args.matrix == "loss":
            update_scheduler(args.scheduler_type, scheduler_D, total_error)
            update_scheduler(args.scheduler_type, scheduler_G, total_error)

            if total_error < best_error:
                best_error = total_error
                best_model_weight_D = copy.deepcopy(net_D.state_dict())
                best_model_weight_G = copy.deepcopy(net_G.state_dict())
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= args.epoch_patience:
                    print("Early Stop")
                    net_D.load_state_dict(best_model_weight_D)
                    net_G.load_state_dict(best_model_weight_G)
                    break
        else:
            update_scheduler(args.scheduler_type, scheduler_D, Acc_FID)
            update_scheduler(args.scheduler_type, scheduler_G, Acc_FID)

            if Acc_FID < best_fid:
                best_fid = Acc_FID
                best_model_weight_D = copy.deepcopy(net_D.state_dict())
                best_model_weight_G = copy.deepcopy(net_G.state_dict())
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= args.epoch_patience:
                    print("Early Stop")
                    net_D.load_state_dict(best_model_weight_D)
                    net_G.load_state_dict(best_model_weight_G)
                    break

        epoch_pbar.set_description(f"Epoch [{epoch+1}/{args.epoch}]")
        epoch_pbar.set_postfix(
            Acc_FID=f"{Acc_FID:.4f}",
            Acc_HOG=f"{Acc_HOG:.4f}",
            Loss_D=f'{performance["epoch_error_D"]:.4f}',
            loss_G=f'{performance["epoch_error_G"]:.4f}',
            D_x=f'{performance["epoch_D_x"]:.4f}',
            D_G_z=f'{performance["epoch_D_G_z_fake"]:.4f} / {performance["epoch_D_G_z_real"]:.4f}',
        )

    net_D.load_state_dict(best_model_weight_D)
    net_G.load_state_dict(best_model_weight_G)
    utils.save_checkpoint(os.path.join(args.save, args.task, Date, "best.pth"), net_D)


if __name__ == "__main__":
    
    args = parser.arg_parse(1)
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    main(args)

    # from torchsummary import summary
    # G, D = get_GAN("A", verbose=True)
    # G.to(args.device)
    # summary(G, (100, 1, 1))
    # D.to(args.device)
    # summary(D, (3, 64, 64))
    # G, D = get_GAN("B", verbose=True)
    # G.to(args.device)
    # summary(G, (100, 1, 1))
    # D.to(args.device)
    # summary(D, (3, 64, 64))