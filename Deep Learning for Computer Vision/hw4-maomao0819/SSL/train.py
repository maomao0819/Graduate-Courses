
import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import parser
from dataset import ImageClassificationDataset
from model import Resnet50Model
import utils
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

def run_one_epoch(
    args,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    mode: str = "train",
):

    if mode == 'train':
        model.train()
    else:
        model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    epoch_loss = 0.0
    epoch_correct = 0.0
    n_batch = len(dataloader)
    batch_pbar = tqdm((dataloader), total=n_batch, desc="Batch")
    with torch.set_grad_enabled(mode == "train"):
        for batch_idx, data in enumerate(batch_pbar, 1):
            images, labels, _ = data
            images = images.to(args.device)
            labels = labels.to(args.device)
            if mode == "train":
                optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)

            if mode == "train":
                loss.backward()
                optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            epoch_correct += batch_correct
            batch_pbar.set_description(f"Batch [{batch_idx}/{n_batch}]")
            batch_pbar.set_postfix(loss=f"{batch_loss:.4f}")

    if mode != "train":
        update_scheduler(args.scheduler_type, scheduler, epoch_loss)

    performance = {}
    n_data = len(dataloader.dataset)
    performance["loss"] = epoch_loss / n_data
    performance["acc"] = epoch_correct / n_data
    return performance

def update_scheduler(scheduler_type, scheduler, matrix):
    if scheduler == None:
        return
    if scheduler_type == "exponential":
        scheduler.step()
    elif scheduler_type == "reduce":
        scheduler.step(matrix)
    return

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    
    transform_set = [
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(args.image_size / 2 - 1, sigma=(0.005, 0.01)),
        transforms.RandomVerticalFlip(),
        transforms.RandomPerspective(distortion_scale=0.05, interpolation=2),
        transforms.RandomResizedCrop(size=args.image_size, scale=(0.9, 1.0), ratio=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(3, interpolation=InterpolationMode.BICUBIC, expand=False),
    ]

    TRANSFORM_IMG = {
        'train': transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.RandomApply(transform_set, p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]),
            ]),
        'val': transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]),
            ])
    }
    
    trainset = ImageClassificationDataset(image_path_root=args.image_dir, csv_file_path=args.csv_file, split='train', transform=TRANSFORM_IMG['train'])
    valset = ImageClassificationDataset(image_path_root=args.image_dir, csv_file_path=args.csv_file, split='val', transform=TRANSFORM_IMG['val'])

    # Use the torch dataloader to iterate through the dataset
    train_loader = DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model = Resnet50Model(fix_backbone=args.fix_backbone).to(args.device)

    if args.pretrain == 'Mine':
        model.backbone = utils.load_checkpoint(args.load, model.backbone)
    elif args.pretrain == 'TA':
        model.backbone = utils.load_checkpoint(args.load_TA, model.backbone)
    
    # Setup optimizers
    if args.fix_backbone:
        if args.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        if args.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.power(0.01, 1 / args.epoch))
    elif args.scheduler_type == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min" if args.matrix == "loss" else "max", factor=0.7, patience=args.lr_patience
        )
    else:
        scheduler = None

    best_loss = np.inf
    best_acc = -np.inf
    best_model_weight = copy.deepcopy(model.state_dict())
    trigger_times = 0
    epoch_pbar = trange(args.epoch, desc="Epoch")
    # For each epoch
    for epoch_idx in epoch_pbar:
        performance_train = run_one_epoch(args, model, dataloader=train_loader, optimizer=optimizer, scheduler=scheduler, mode='train')
        performance_eval = run_one_epoch(args, model, dataloader=val_loader, optimizer=optimizer, scheduler=scheduler, mode='val')

        if epoch_idx % args.save_interval == 0:
            utils.save_checkpoint(os.path.join(args.save, 'train', f"pretrain_{args.pretrain}", f"fix-backbone_{args.fix_backbone}", f"{epoch_idx+1}.pth"), model)
            
        if args.matrix == "loss":
            if performance_eval["loss"] < best_loss:
                best_loss = performance_eval["loss"]
                best_model_weight = copy.deepcopy(model.state_dict())
                trigger_times = 0
                utils.save_checkpoint(os.path.join(args.save, 'train', f"pretrain_{args.pretrain}", f"fix-backbone_{args.fix_backbone}", "better.pth"), model)
            else:
                trigger_times += 1
                if trigger_times >= args.epoch_patience:
                    print("Early Stop")
                    model.load_state_dict(best_model_weight)
                    break
        else:
            if performance_eval["acc"] > best_acc:
                best_acc = performance_eval["acc"]
                best_model_weight = copy.deepcopy(model.state_dict())
                trigger_times = 0
                utils.save_checkpoint(os.path.join(args.save, 'train', f"pretrain_{args.pretrain}", f"fix-backbone_{args.fix_backbone}", "better.pth"), model)
            else:
                trigger_times += 1
                if trigger_times >= args.epoch_patience:
                    print("Early Stop")
                    model.load_state_dict(best_model_weight)
                    break

        epoch_pbar.set_description(f"Epoch [{epoch_idx+1}/{args.epoch}]")
        epoch_pbar.set_postfix(
            train_loss=performance_train["loss"],
            train_acc=performance_train["acc"],
            eval_loss=performance_eval["loss"],
            eval_acc=performance_eval["acc"],
        )

    model.load_state_dict(best_model_weight)
    utils.save_checkpoint(os.path.join(args.save, 'train', f"pretrain_{args.pretrain}", f"fix-backbone_{args.fix_backbone}", "best.pth"), model)

if __name__ == "__main__":
    args = parser.arg_parse()
    main(args)