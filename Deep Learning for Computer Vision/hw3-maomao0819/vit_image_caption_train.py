
import os
import copy
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer
import parser
from dataset import ImageCaptionLabelDataset
from model import make_model, subsequent_mask, make_std_mask
import utils
from vit_image_caption_eval import eval
n_pixels = 197

def run_one_epoch(
    args,
    model: torch.nn.Module,
    tokenizer: Tokenizer, 
    dataloader: DataLoader,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    mode: str,
):

    if mode == 'train':
        model.train()
    else:
        model.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataloader.dataset.ignore_idx)
    # criterion = torch.nn.CrossEntropyLoss()
    # nn.KLDivLoss(reduction="sum")
    epoch_loss = 0
    epoch_correct = 0
    n_batch = len(dataloader)
    tqdm_loop = tqdm((dataloader), total=n_batch)
    for batch_idx, data in enumerate(tqdm_loop, 1):
        with torch.set_grad_enabled(mode == 'train'):
            # [batch_size, 3, 224, 224]
            image = data["image"].to(args.device)
            # [batch_size, seq_len]
            tgt = data["tokens_id"].to(args.device)
            tgt_next = data["tokens_id_next"].to(args.device)
            batch_size = image.size(0)
            src_mask = torch.ones(batch_size, 1, 14*14).to(args.device)
            tgt_mask = make_std_mask(tgt, pad=dataloader.dataset.get_pad_id())

            # [batch_size, seq_len, d_model]
            decode = model(image, tgt, src_mask, tgt_mask)
            # [batch_size, tgt_vocab, seq_len]
            prediction = model.generator(decode).permute(0, 2, 1)

            optimizer.zero_grad()
            loss = criterion(prediction, tgt_next)

            if mode == 'train':
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            
            # [batch_size, seq_len]
            pred = (torch.argmax(prediction, dim=1))
            pred = prediction.max(1)[1]  # get the index of the max log-probability
            # [batch_size, seq_len]
            mask = data['mask'].to(args.device)
            batch_correct = 0
            # [batch_size, seq_len]

            pred = pred * mask
            tgt_next = tgt_next * mask
            
            batch_correct = torch.all(torch.eq(pred, tgt_next), dim=1).sum().item()

            epoch_loss += batch_loss
            epoch_correct += batch_correct
            tqdm_loop.set_description(f"Batch [{batch_idx}/{n_batch}]")
            tqdm_loop.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{float(batch_correct) / float(tgt_next.shape[0]):.4f}")

    if mode == 'val':
        update_scheduler(args.scheduler_type, scheduler, epoch_loss)
        # eval(args, model, dataloader, tokenizer)
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
    config = resolve_data_config({}, model=timm.create_model('vit_large_patch16_224', pretrained=True))
    transform = create_transform(**config)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    trainset = ImageCaptionLabelDataset(data_dir=args.data_dir, tokenizer=tokenizer, split_set='train', transform=transform)
    valset = ImageCaptionLabelDataset(data_dir=args.data_dir, tokenizer=tokenizer, split_set='val', transform=transform)

    # Use the torch dataloader to iterate through the dataset
    train_loader = DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, collate_fn=trainset.collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, collate_fn=valset.collate_fn, pin_memory=True
    )

    model = make_model(args.freeze, tokenizer.get_vocab_size(), args.n_layers, args.d_model, args.d_ff, args.n_heads, args.dropout).to(args.device)
    # Setup optimizers
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            amsgrad=False,
        )

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
        performance_train = run_one_epoch(args, model, tokenizer=tokenizer, dataloader=train_loader, optimizer=optimizer, scheduler=scheduler, mode='train')
        performance_eval = run_one_epoch(args, model, tokenizer=tokenizer, dataloader=val_loader, optimizer=optimizer, scheduler=scheduler, mode='val')

        if epoch_idx % args.save_interval == 0:
            utils.save_checkpoint(os.path.join(args.save, f"{epoch_idx+1}.pth"), model)
            
        if args.matrix == "loss":

            if performance_eval["loss"] < best_loss:
                best_loss = performance_eval["loss"]
                best_model_weight = copy.deepcopy(model.state_dict())
                trigger_times = 0
                utils.save_checkpoint(os.path.join(args.save, "better.pth"), model)
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
                utils.save_checkpoint(os.path.join(args.save, "better.pth"), model)
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
    utils.save_checkpoint(os.path.join(args.save, "best.pth"), model)

if __name__ == "__main__":
    args = parser.arg_parse(2)
    main(args)