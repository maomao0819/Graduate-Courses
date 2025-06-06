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
from dataset import DigitImageDataset
from typing import Dict, List
import digit_classifier

def train_one_epoch(
    args,
    Unet_model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim,
) -> float:

    Unet_model.train()
    trainer = model.GaussianDiffusionTrainer(Unet_model, args.beta_1, args.beta_T, args.time_length).to(args.device)

    epoch_loss = 0.0
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for batch_idx, data in enumerate(tqdmDataLoader, 1):
            # train
            batch_size = data[0].shape[0]
            optimizer.zero_grad()
            images = data[0].to(args.device)
            labels = data[1].to(args.device)
            if np.random.rand() < 0.1:
                labels = torch.zeros_like(labels).to(args.device)
            loss = trainer(images, labels).sum() / batch_size ** 2.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Unet_model.parameters(), args.grad_clip)
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss  # sum up batch loss

            tqdmDataLoader.set_description(f"Batch [{batch_idx}/{len(dataloader)}]")
            tqdmDataLoader.set_postfix(loss=batch_loss)

    epoch_loss /= len(dataloader.dataset)
    return epoch_loss

def eval(
    args,
    Unet_model: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler,
    noisyImage: torch,
    grid_path: str,
) -> float:

    Unet_model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)
    with torch.no_grad():
        n_label = 10

        # test_step = int(args.test_batch // n_label)
        # labelList = []
        # k = 0
        # for i in range(1, args.test_batch + 1):
        #     labelList.append(torch.ones(size=[1]).long() * k)
        #     if i % test_step == 0:
        #         if k < 10 - 1:
        #             k += 1
        # labels = torch.cat(labelList, dim=0).long().to(args.device) + 1
        
        labelList = []
        test_label_batch = int(len(noisyImage) // n_label)
        for label in range(n_label - 1):
            labelList.extend([label] * test_label_batch)
        labelList.extend([n_label - 1] * (len(noisyImage) - len(labelList)))
        labels = torch.Tensor(labelList).long().to(args.device)

        sampler = model.GaussianDiffusionSampler(Unet_model, args.beta_1, args.beta_T, args.time_length, w=args.w).to(args.device)
        # Sampled from standard normal distribution
        sampled_images = sampler(noisyImage, labels)
        sampled_images = sampled_images * 0.5 + 0.5  # [0 ~ 1]
        for image_idx in range(sampled_images.size(0)):
            vutils.save_image(sampled_images[image_idx], os.path.join(args.output_dir, f'{labels[image_idx]}_{(image_idx % test_label_batch + 1):03d}.png'), padding=0)

        vutils.save_image(sampled_images, os.path.join(grid_path, 'grid.png'), nrow=test_label_batch, padding=0)

    classifier = digit_classifier.Classifier()
    classifier_path = "./Classifier.pth"
    digit_classifier.load_checkpoint(classifier_path, classifier)
    classifier = classifier.to(args.device)

    digit_data_loader = torch.utils.data.DataLoader(digit_classifier.DATA(args.output_dir), batch_size=32, num_workers=4, shuffle=False)

    correct = 0
    classifier.eval()
    #print('===> start evaluation ...')
    with torch.no_grad():
        for images, labels in digit_data_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            output = classifier(images)
            _, pred = torch.max(output, 1)
            correct += (pred == labels).detach().sum().item()
    n_data = len(digit_data_loader.dataset)
    acc = float(correct) / n_data
    update_scheduler(args.scheduler_type, scheduler, acc)
    print('acc = {} (correct/total = {}/{})'.format(acc, correct, n_data))
    return acc

def update_scheduler(scheduler_type, scheduler, matrix):
    if scheduler_type == "exponential":
        scheduler.step()
    elif scheduler_type == "reduce":
        scheduler.step(matrix)


def main(args):
    time = datetime.datetime.now()
    Date = f"{time.month}_{time.day}"

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    transform_set = [
        transforms.RandomResizedCrop(size=args.image_size, scale=(0.9, 1.0), ratio=(0.9, 1.0)),
        transforms.RandomRotation(3, interpolation=InterpolationMode.BICUBIC, expand=False),
    ]

    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    image_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.RandomApply(transform_set, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    
    training_dir = os.path.join(args.data_dir)
    trainset = DigitImageDataset(dir=training_dir, transform=image_transform)

    # # Use the torch dataloader to iterate through the dataset
    train_loader = DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )

    Unet_model = model.UNet(time_length=args.time_length, n_labels=10, channel=args.channel, channel_multiply=args.channel_multiply,
        n_residual_blocks=args.n_residual_blocks, dropout=args.dropout).to(args.device)

    # Setup optimizers
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(Unet_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(Unet_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            Unet_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    if args.scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.power(0.01, 1 / args.epoch))
    elif args.scheduler_type == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min" if args.matrix == "loss" else "max", factor=0.5, patience=args.lr_patience
        )
    else:
        scheduler = None

    output_path = os.path.join(args.save, Date)
    os.makedirs(output_path, exist_ok=True)
    del_keys = ["data_dir", "output_dir", "workers", "load", "device"]
    utils.saving_args(args, output_path, del_keys)

    n_labels = 10
    test_batch = 10
    noisyImage = torch.randn(size=[test_batch * n_labels, 3, args.image_size, args.image_size], device=args.device)

    best_loss = np.inf
    best_acc = -np.inf
    best_model_weight = copy.deepcopy(Unet_model.state_dict())
    trigger_times = 0
    epoch_pbar = trange(args.epoch, desc="Epoch")
    # For each epoch
    for epoch_idx in epoch_pbar:
        epoch_loss = train_one_epoch(args, Unet_model, train_loader, optimizer)
        save_path = os.path.join(output_path, f'{epoch_idx+1}')
        epoch_acc = eval(args, Unet_model, scheduler, noisyImage, save_path)

        if epoch_idx % args.save_interval == 0:
            utils.save_checkpoint(os.path.join(save_path, "ckpt.pth"), Unet_model)

            with open(os.path.join(output_path, "log.txt"), "a") as outfile:
                outfile.write(f"epoch: {epoch_idx+1}\n")
                outfile.write(f"loss: {epoch_loss}\n")
                outfile.write(f"acc: {epoch_acc}\n")
                outfile.write("-" * 20 + "\n\n")

        if args.matrix == "loss":

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weight = copy.deepcopy(Unet_model.state_dict())
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= args.epoch_patience:
                    print("Early Stop")
                    Unet_model.load_state_dict(best_model_weight)
                    break
        else:
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weight = copy.deepcopy(Unet_model.state_dict())
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= args.epoch_patience:
                    print("Early Stop")
                    Unet_model.load_state_dict(best_model_weight)
                    break

        epoch_pbar.set_description(f"Epoch [{epoch_idx+1}/{args.epoch}]")
        epoch_pbar.set_postfix(loss=f"{epoch_loss:.4f}", Acc=f"{epoch_acc:.4f}",)

    Unet_model.load_state_dict(best_model_weight)
    utils.save_checkpoint(os.path.join(output_path, "best.pth"), Unet_model)


if __name__ == "__main__":
    args = parser.arg_parse(2)
    main(args)