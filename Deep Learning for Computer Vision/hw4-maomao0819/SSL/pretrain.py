
import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import parser
from dataset import ImageClassificationDataset
import utils
import torchvision.transforms as transforms
from byol_pytorch import BYOL
import torchvision

def run_one_epoch(
    args,
    learner,
    dataloader: DataLoader,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
):
    epoch_loss = 0
    n_batch = len(dataloader)
    tqdm_loop = tqdm((dataloader), total=n_batch)
    for batch_idx, data in enumerate(tqdm_loop, 1):
        images, _, _ = data
        images = images.to(args.device)
        optimizer.zero_grad()
        loss = learner(images)
        loss.backward()
        optimizer.step()
        learner.update_moving_average()
        
        batch_loss = loss.item()

        epoch_loss += batch_loss
        tqdm_loop.set_description(f"Batch [{batch_idx}/{n_batch}]")
        tqdm_loop.set_postfix(loss=f"{batch_loss:.4f}")

    update_scheduler(args.scheduler_type, scheduler, epoch_loss)

    performance = {}
    performance["loss"] = epoch_loss / len(dataloader.dataset)
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
    
    TRANSFORM_IMG = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ]
    )
    
    dataset = ImageClassificationDataset(image_path_root=args.image_dir_pretrain, csv_file_path=args.csv_file_pretrain, split=None, transform=TRANSFORM_IMG)

    # Use the torch dataloader to iterate through the dataset
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )

    resnet50 = torchvision.models.resnet50(pretrained=False).to(args.device)
    learner = BYOL(
        resnet50,
        image_size = args.image_size,
        hidden_layer = 'avgpool'
    )

    # Setup optimizers
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(learner.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(learner.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(learner.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.power(0.01, 1 / args.epoch))
    elif args.scheduler_type == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=args.lr_patience
        )
    else:
        scheduler = None

    best_loss = np.inf
    best_model_weight = copy.deepcopy(resnet50.state_dict())
    trigger_times = 0
    epoch_pbar = trange(args.epoch, desc="Epoch")
    # For each epoch
    for epoch_idx in epoch_pbar:
        performance = run_one_epoch(args, learner, dataloader=dataloader, optimizer=optimizer, scheduler=scheduler)

        if epoch_idx % args.save_interval == 0:
            utils.save_checkpoint(os.path.join(args.save, "pretrain", f"{epoch_idx+1}.pth"), resnet50)

        if performance["loss"] < best_loss:
            best_loss = performance["loss"]
            best_model_weight = copy.deepcopy(resnet50.state_dict())
            trigger_times = 0
            utils.save_checkpoint(os.path.join(args.save, "pretrain", "better.pth"), resnet50)
        else:
            trigger_times += 1
            if trigger_times >= args.epoch_patience:
                print("Early Stop")
                resnet50.load_state_dict(best_model_weight)
                break

        epoch_pbar.set_description(f"Epoch [{epoch_idx+1}/{args.epoch}]")
        epoch_pbar.set_postfix(loss=performance["loss"])

    resnet50.load_state_dict(best_model_weight)
    utils.save_checkpoint(os.path.join(args.save, "pretrain", "best.pth"), resnet50)

if __name__ == "__main__":
    args = parser.arg_parse()
    main(args)