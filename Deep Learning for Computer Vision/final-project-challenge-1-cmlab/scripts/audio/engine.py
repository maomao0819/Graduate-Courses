import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, cfg):

    torch.set_printoptions(precision=3)
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        iters = len(train_loader)
        loss_values = []
        correct_all = torch.tensor([], device=cfg.device)
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                inputs, labels = data.to(cfg.device), target.to(cfg.device)
                inputs = torch.squeeze(inputs)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                correct_all = torch.cat((correct_all, (torch.argmax(outputs, dim=1) == labels)), 0)
                accuracy = correct_all.sum().item() / correct_all.shape[0]

                optimizer.zero_grad()
                loss.backward()
                # if cfg.grad_clip > 0:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                if cfg.scheduler == 'cos':
                    scheduler.step(epoch + i / iters)

                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)
                epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        loss_values.append(epoch_loss)
        if cfg.scheduler == 'exp':
            scheduler.step()

        accuracy = valid_model(model, valid_loader, criterion, cfg)

        if accuracy > cfg.benchmark:
            cfg.benchmark = accuracy
            print('current accuracy', accuracy, 'beat benchmark, saving model')
            torch.save(model, cfg.model_path)
    print('\nBest Accuracy: ', cfg.benchmark)
    return model


@torch.no_grad()
def valid_model(model, valid_loader, criterion, cfg):
    torch.set_printoptions(precision=3)
    correct_all = torch.tensor([], device=cfg.device)
    valid_loss = []
    valid_acc = []

    model.eval()
    with tqdm(valid_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Valid  ")

            inputs, labels = data.to(cfg.device), target.to(cfg.device)

            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            valid_loss.append(loss.item())

            correct_all = torch.cat((correct_all, (torch.argmax(outputs, dim=1) == labels)), 0)
            accuracy = correct_all.sum().item() / correct_all.shape[0]
            valid_acc.append(accuracy)

            tepoch.set_postfix(accuracy=sum(valid_acc)/len(valid_acc), loss=sum(valid_loss)/len(valid_loss))

    print('total validation accuracy:', sum(valid_acc)/len(valid_acc))

    return sum(valid_acc)/len(valid_acc)
