import os
import pandas as pd
import torch
import json

def layer_debug_log(tensor, layer_name='layer'):
    print(f'Tensor size and type after {layer_name}:', tensor.shape, tensor.dtype)

def dataset_debug(dataset):
    print(len(dataset))

def dataloader_debug(dataloader):
    batch = next(iter(dataloader))
    print(batch)

def load_json(json_path):
    if (json_path is not None) and os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            json_content = json.load(json_file)
        return json_content

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

def predict_to_csv(basenames, predictions, output_path):
    df = pd.DataFrame(columns=['filename', 'label'])
    df['filename'] = basenames
    df['label'] = predictions
    df['real_label'] = df['filename'].apply(lambda basename: int(basename.split('_')[0]))
    df = df.sort_values(by=['real_label', 'filename']).reset_index(drop=True).drop(columns='real_label')
    df.to_csv(output_path, index=False)

def save_json(data, json_path):
    if json_path.count("/"):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    # with open(json_path, 'w') as f:
    #     json.dump(data, f)
    json.dump(data, open(json_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)