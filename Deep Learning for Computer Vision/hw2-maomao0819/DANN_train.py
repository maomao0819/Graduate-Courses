import os
import copy
import numpy as np
import datetime
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import parser
import utils
import model
from dataset import DigitImageDataset
from typing import Dict


def run_CNN_one_epoch(
    args,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    mode: str = "train",
) -> Dict:

    if mode == "train":
        model.train()
    else:
        model.eval()

    # Initialize CrossEntropyLoss function
    criterion = torch.nn.CrossEntropyLoss()
    epoch_loss = 0.0
    epoch_correct = 0.0
    n_batch = len(dataloader)
    batch_pbar = tqdm((dataloader), total=n_batch)
    with torch.set_grad_enabled(mode == "train"):
        for batch_idx, data in enumerate(batch_pbar, 1):
            image = data[0].to(args.device)
            label = data[1].to(args.device)
            if mode == "train":
                optimizer.zero_grad()
            output = model(image)["class"]
            loss = criterion(output, label)
            if mode == "train":
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(label.view_as(pred)).sum().item()
            epoch_correct += batch_correct

            batch_pbar.set_description(f"Batch [{batch_idx}/{n_batch}]")
            batch_pbar.set_postfix(loss=batch_loss, acc=float(batch_correct) / float(label.shape[0]))

    if mode != "train":
        update_scheduler(args.scheduler_type, scheduler, epoch_loss)

    performance = {}
    n_data = len(dataloader.dataset)
    performance["loss"] = epoch_loss / n_data
    performance["acc_class"] = epoch_correct / n_data
    return performance


def train_DANN_one_epoch(
    args,
    model: torch.nn.Module,
    source_dataloader: DataLoader,
    target_dataloader: DataLoader,
    optimizer: torch.optim,
    epoch_idx: int = 0,
) -> Dict:

    model.train()

    # Initialize CrossEntropyLoss function
    criterion = torch.nn.CrossEntropyLoss()
    epoch_loss = 0.0
    epoch_class_correct = 0.0
    epoch_domain_correct = 0.0

    target_iter = iter(target_dataloader)
    batch_pbar = tqdm((source_dataloader), total=len(source_dataloader))
    source_batch_size = 0
    target_batch_size = 0
    for batch_idx, source_data in enumerate(batch_pbar):
        try:
            target_data = next(target_iter)
        except StopIteration:
            target_iter = iter(target_dataloader)
            target_data = next(target_iter)
        p = float(batch_idx + epoch_idx * len(source_dataloader)) / args.epoch / len(source_dataloader)
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        source_image = source_data[0].to(args.device)
        source_class_label = source_data[1].to(args.device)
        target_image = target_data[0].to(args.device)

        source_batch_size = len(source_image)
        target_batch_size = len(target_image)

        source_domain_label = torch.zeros(source_batch_size).long().to(args.device)  # source 0
        target_domain_label = torch.ones(target_batch_size).long().to(args.device)  # target 1

        optimizer.zero_grad()

        # train on source domain
        source_output = model(source_image, alpha=alpha)
        source_loss_class = criterion(source_output["class"], source_class_label)
        source_loss_domain = criterion(source_output["domain"], source_domain_label)

        # train on target domain
        target_output = model(target_image, alpha=alpha)
        target_loss_domain = criterion(target_output["domain"], target_domain_label)

        # domain loss
        loss = source_loss_class + source_loss_domain + target_loss_domain

        # backward + optimize
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss  # sum up batch loss
        pred_class = source_output["class"].max(1, keepdim=True)[1]  # get the index of the max log-probability
        batch_class_correct = pred_class.eq(source_class_label.view_as(pred_class)).sum().item()
        pred_source_domain = source_output["domain"].max(1, keepdim=True)[1]  # get the index of the max log-probability
        batch_source_domain_correct = (
            pred_source_domain.eq(source_domain_label.view_as(pred_source_domain)).sum().item()
        )
        pred_target_domain = target_output["domain"].max(1, keepdim=True)[1]  # get the index of the max log-probability
        batch_target_domain_correct = (
            pred_target_domain.eq(target_domain_label.view_as(pred_target_domain)).sum().item()
        )

        batch_domain_correct = batch_source_domain_correct + batch_target_domain_correct
        epoch_class_correct += batch_class_correct
        epoch_domain_correct += batch_domain_correct

        batch_pbar.set_description(f"Batch [{batch_idx+1}/{len(source_dataloader)}]")
        batch_pbar.set_postfix(
            loss=batch_loss,
            acc_class=float(batch_class_correct) / float(source_batch_size),
            acc_domain=float(batch_domain_correct) / float(source_batch_size + target_batch_size),
        )

    performance = {}
    n_source_data = len(source_dataloader.dataset)
    n_target_data = float(len(target_dataloader.dataset)) * len(source_dataloader) / len(target_dataloader)
    performance["loss"] = epoch_loss / (n_source_data * 2 + n_target_data)
    performance["acc_class"] = epoch_class_correct / n_source_data
    performance["acc_domain"] = epoch_domain_correct / (n_source_data + n_target_data)
    return performance


def val_DANN(
    args,
    model: torch.nn.Module,
    dataloader: DataLoader,
    scheduler: torch.optim.lr_scheduler,
) -> Dict:

    model.eval()

    # Initialize CrossEntropyLoss function
    criterion = torch.nn.CrossEntropyLoss()
    epoch_loss = 0.0
    epoch_class_correct = 0.0
    epoch_domain_correct = 0.0
    n_batch = len(dataloader)
    batch_pbar = tqdm((dataloader), total=n_batch)
    with torch.no_grad():
        for batch_idx, data in enumerate(batch_pbar, 1):
            alpha = 0
            image = data[0].to(args.device)
            class_label = data[1].to(args.device)

            batch_size = len(image)

            domain_label = torch.ones(batch_size).long().to(args.device)  # target 1

            # train on source domain
            output = model(image, alpha=alpha)
            loss_class = criterion(output["class"], class_label)
            loss_domain = criterion(output["domain"], domain_label)

            # domain loss
            loss = loss_class + loss_domain

            batch_loss = loss.item()
            epoch_loss += batch_loss  # sum up batch loss
            pred_class = output["class"].max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_class_correct = pred_class.eq(class_label.view_as(pred_class)).sum().item()
            pred_domain = output["domain"].max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_domain_correct = pred_domain.eq(domain_label.view_as(pred_domain)).sum().item()

            epoch_class_correct += batch_class_correct
            epoch_domain_correct += batch_domain_correct

            batch_pbar.set_description(f"Batch [{batch_idx+1}/{n_batch}]")
            batch_pbar.set_postfix(
                loss=batch_loss,
                acc_class=float(batch_class_correct) / float(batch_size),
                acc_domain=float(batch_domain_correct) / float(batch_size),
            )

    update_scheduler(args.scheduler_type, scheduler, epoch_loss)

    performance = {}
    n_data = len(dataloader.dataset)
    performance["loss"] = epoch_loss / (n_data * 2)
    performance["acc_class"] = epoch_class_correct / n_data
    performance["acc_domain"] = epoch_domain_correct / n_data
    return performance


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
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30, interpolation=InterpolationMode.BICUBIC, expand=False),
        transforms.RandomPerspective(distortion_scale=0.3, interpolation=2),
        transforms.RandomResizedCrop(size=args.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    ]

    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    image_transform = {
        "train": transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.RandomApply(transform_set, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }
    source_dataset = "mnistm"
    if args.task == "A":
        target_dataset = "svhn"
    else:
        target_dataset = "usps"

    target_dir = os.path.join(args.data_dir, target_dataset)
    if args.model_mode == "cross":
        source_dir = os.path.join(args.data_dir, source_dataset)
        source_trainset = DigitImageDataset(dir=source_dir, mode="train", transform=image_transform["train"])
        # Use the torch dataloader to iterate through the dataset
        train_loader = DataLoader(
            source_trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True
        )

    if args.model_mode == "same":
        target_trainset = DigitImageDataset(dir=target_dir, mode="train", transform=image_transform["train"])
        # Use the torch dataloader to iterate through the dataset
        train_loader = DataLoader(
            target_trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True
        )

    else:
        source_dir = os.path.join(args.data_dir, source_dataset)
        source_trainset = DigitImageDataset(dir=source_dir, mode="train", transform=image_transform["train"])
        target_trainset = DigitImageDataset(dir=target_dir, mode="train", transform=image_transform["train"])
        # Use the torch dataloader to iterate through the dataset
        source_train_loader = DataLoader(
            source_trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True
        )
        target_train_loader = DataLoader(
            target_trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True
        )

    target_valset = DigitImageDataset(dir=target_dir, mode="val", transform=image_transform["val"])
    val_loader = DataLoader(
        target_valset, batch_size=args.test_batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )

    DANNModel = model.DANN_model().to(args.device)

    # Setup optimizers
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(DANNModel.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(DANNModel.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            DANNModel.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            amsgrad=False,
        )

    if args.scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.power(0.01, 1 / args.epoch))
    elif args.scheduler_type == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min" if args.matrix == "loss" else "max", factor=0.5, patience=args.lr_patience
        )
    else:
        scheduler = None

    output_path = os.path.join(args.save, args.task, args.model_mode, Date)
    os.makedirs(output_path, exist_ok=True)
    del_keys = ["data_dir", "output_path", "task", "workers", "load", "device", "model_mode"]
    utils.saving_args(args, output_path, del_keys)

    loss = {}
    loss["train"] = []
    loss["val"] = []
    acc_class = {}
    acc_class["train"] = []
    acc_class["val"] = []

    if args.model_mode == "DANN":
        acc_domain = {}
        acc_domain["train"] = []
        acc_domain["val"] = []
    best_loss = np.inf
    best_acc = -np.inf
    best_model_weight = copy.deepcopy(DANNModel.state_dict())
    trigger_times = 0
    epoch_pbar = trange(args.epoch, desc="Epoch")
    # For each epoch
    for epoch_idx in epoch_pbar:
        if args.model_mode == "DANN":
            performance_train = train_DANN_one_epoch(
                args, DANNModel, source_train_loader, target_train_loader, optimizer, epoch_idx
            )
            performance_val = val_DANN(args, DANNModel, val_loader, scheduler)
        else:
            performance_train = run_CNN_one_epoch(args, DANNModel, train_loader, optimizer, scheduler, "train")
            performance_val = run_CNN_one_epoch(args, DANNModel, val_loader, optimizer, scheduler, "val")

        epoch_loss_train = performance_train["loss"]
        loss["train"].append(epoch_loss_train)
        epoch_loss_val = performance_val["loss"]
        loss["val"].append(epoch_loss_val)

        epoch_acc_class_train = performance_train["acc_class"]
        acc_class["train"].append(epoch_acc_class_train)
        epoch_acc_class_val = performance_val["acc_class"]
        acc_class["val"].append(epoch_acc_class_val)

        epoch_acc_domain_train = 0.0
        epoch_acc_domain_val = 0.0
        if args.model_mode == "DANN":
            epoch_acc_domain_train = performance_train["acc_domain"]
            acc_domain["train"].append(epoch_acc_domain_train)
            epoch_acc_domain_val = performance_val["acc_domain"]
            acc_domain["val"].append(epoch_acc_domain_val)

        # Check how the generator is doing by saving G's output on fixed_noise_show
        if epoch_idx % args.save_interval == 0:
            path = os.path.join(output_path, f"{epoch_idx+1}")
            utils.save_checkpoint(os.path.join(path, "ckpt.pth"), DANNModel)
            utils.plot_performance(loss["train"], loss["val"], path, "loss")
            utils.plot_performance(acc_class["train"], acc_class["val"], path, "accuracy_class")
            if args.model_mode == "DANN":
                utils.plot_performance(acc_domain["train"], acc_domain["val"], path, "accuracy_domain")

            with open(os.path.join(output_path, "log.txt"), "a") as outfile:
                outfile.write(f"epoch: {epoch_idx+1}\n")
                outfile.write(f"loss train: {epoch_loss_train}\n")
                outfile.write(f"loss val: {epoch_loss_val}\n")
                outfile.write(f"acc class train: {epoch_acc_class_train}\n")
                outfile.write(f"acc class val: {epoch_acc_class_val}\n")
                if args.model_mode == "DANN":
                    outfile.write(f"acc domain train: {epoch_acc_domain_train}\n")
                    outfile.write(f"acc domain val: {epoch_acc_domain_val}\n")
                outfile.write("-" * 20 + "\n\n")

        if args.matrix == "loss":

            if epoch_loss_val < best_loss:
                best_loss = epoch_loss_val
                best_model_weight = copy.deepcopy(DANNModel.state_dict())
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= args.epoch_patience:
                    print("Early Stop")
                    DANNModel.load_state_dict(best_model_weight)
                    break
        else:
            if epoch_acc_class_val + epoch_acc_domain_val > best_acc:
                best_acc = epoch_acc_class_val + epoch_acc_domain_val
                best_model_weight = copy.deepcopy(DANNModel.state_dict())
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= args.epoch_patience:
                    print("Early Stop")
                    DANNModel.load_state_dict(best_model_weight)
                    break

        epoch_pbar.set_description(f"Epoch [{epoch_idx+1}/{args.epoch}]")
        if args.model_mode == "DANN":
            epoch_pbar.set_postfix(
                loss_train=f"{epoch_loss_train:.4f} / {epoch_loss_val:.4f}",
                Acc_class=f"{epoch_acc_class_train:.4f} / {epoch_acc_class_val:.4f}",
                Acc_domain=f"{epoch_acc_domain_train:.4f} / {epoch_acc_domain_val:.4f}",
            )
        else:
            epoch_pbar.set_postfix(
                loss_train=f"{epoch_loss_train:.4f} / {epoch_loss_val:.4f}",
                Acc_class=f"{epoch_acc_class_train:.4f} / {epoch_acc_class_val:.4f}",
            )

    DANNModel.load_state_dict(best_model_weight)
    utils.save_checkpoint(os.path.join(output_path, "best.pth"), DANNModel)


if __name__ == "__main__":
    args = parser.arg_parse(3)
    main(args)
