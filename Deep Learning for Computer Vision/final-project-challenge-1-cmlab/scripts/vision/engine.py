import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, cfg):
    torch.set_printoptions(precision=3)
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        if epoch <= 15:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.3*(1-epoch/15))
        model.train()
        correct_all = torch.tensor([], device=cfg.device)
        train_loss = []
        iters = len(train_loader)
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                inputs, labels = data.to(cfg.device), target.to(cfg.device)

                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                train_loss.append(loss.item())

                correct_all = torch.cat((correct_all, ((outputs > 0.5).int() == labels)), 0)
                accuracy = correct_all.sum().item() / correct_all.shape[0]

                optimizer.zero_grad()
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                if cfg.scheduler == 'cos':
                    scheduler.step(epoch + i / iters)

                tepoch.set_postfix(loss=sum(train_loss)/len(train_loss), accuracy=accuracy)
        if cfg.scheduler == 'exp':
            scheduler.step()

        accuracy = valid_model(model, valid_loader, criterion, cfg)

        if accuracy > cfg.benchmark:
            cfg.benchmark = accuracy
            print('current accuracy', accuracy, 'beat benchmark, saving model')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }, cfg.model_path)
    print('\nBest Accuracy: ', cfg.benchmark)
    return model


@torch.no_grad()
def valid_model(model, valid_loader, criterion, cfg):
    torch.set_printoptions(precision=3)
    correct_all = torch.tensor([], device=cfg.device)
    outputs_all = torch.tensor([], device=cfg.device)
    labels_all = torch.tensor([], device=cfg.device)
    valid_loss = []

    model.eval()
    with tqdm(total=cfg.tta_num*len(valid_loader)) as pbar:
        for i in range(cfg.tta_num):
            for data, target in valid_loader:
                pbar.set_description(f"Valid  ")

                inputs, labels = data.to(cfg.device), target.to(cfg.device)

                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                valid_loss.append(loss.item())

                outputs_all = torch.cat((outputs_all, outputs), 0)
                if i == 0:
                    labels_all = torch.cat((labels_all, labels), 0)
                correct_all = torch.cat((correct_all, ((outputs > 0.5).int() == labels)), 0)
                accuracy = correct_all.sum().item() / correct_all.shape[0]

                pbar.set_postfix(accuracy=accuracy, loss=sum(valid_loss)/len(valid_loss))
                pbar.update(1)
    outputs_all = torch.reshape(outputs_all, (cfg.tta_num, -1))
    outputs_all = torch.mean(outputs_all, dim=0)
    correct_all = ((outputs_all > 0.5).int() == labels_all)
    accuracy = correct_all.sum().item() / correct_all.shape[0]

    print('total validation accuracy:', accuracy)

    return accuracy
