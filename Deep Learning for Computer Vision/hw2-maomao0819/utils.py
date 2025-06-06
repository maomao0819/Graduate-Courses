import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

def layer_debug_log(tensor, layer_name='layer'):
    print(f'Tensor size and type after {layer_name}:', tensor.shape, tensor.dtype)

def dataset_debug(dataset):
    print(len(dataset))

def dataloader_debug(dataloader):
    batch = next(iter(dataloader))
    print(batch)

def save_checkpoint(checkpoint_path, model):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    state = model.state_dict()
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    print('model loaded from %s' % checkpoint_path)
    return model

def update_checkpoint(checkpoint_path, model, layer_name='conv'):
    checkpoint = torch.load(checkpoint_path)
    states_to_load = {}
    for name, param in checkpoint.items():
        if name.startswith(layer_name):
            states_to_load[name] = param

    model_state = model.state_dict()
    model_state.update(states_to_load)
    model.load_state_dict(model_state)
    return model

def plot_error(D_losses, G_losses, D_xs, D_G_z_fakes, D_G_z_reals, path):
    plt.plot(D_losses, color='red')
    plt.plot(G_losses, color='blue')
    plt.plot(D_xs, color='yellow')
    plt.plot(D_G_z_fakes, color='green')
    plt.plot(D_G_z_reals, color='purple')
    plt.title('Loss')
    plt.legend(['D', 'G', 'D(x)', 'D(G(z_fake))', 'D(G(z_real))'])
    plt.xticks(range(len(D_losses))) #set the tick frequency on x-axis
    plt.ylabel('error') #set the label for y axis
    plt.xlabel('index') #set the label for x-axis
    plt.savefig(os.path.join(path, "error.png"))
    plt.clf()

def plot_performance(train, val, path, matrix='loss'):
    plt.plot(train, color='red')
    plt.plot(val, color='blue')
    plt.title(matrix)
    plt.legend(['train', 'val'])
    plt.xticks(range(len(train))) #set the tick frequency on x-axis
    plt.ylabel(matrix) #set the label for y axis
    plt.xlabel('index') #set the label for x-axis
    plt.savefig(os.path.join(path, f"{matrix}.png"))
    plt.clf()

def saving_args(args, output_path, del_keys):
    saving_args = args.__dict__.copy()
    with open(os.path.join(output_path, "arg.json") , "w") as outfile:
        for del_key in del_keys:
            del saving_args[del_key]
        json.dump(saving_args, outfile, indent=2)