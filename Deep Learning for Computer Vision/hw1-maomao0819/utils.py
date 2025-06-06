import os
import numpy as np
import torch

def layer_debug_log(tensor):
    print('Tensor size and type after layer:', tensor.shape, tensor.dtype)

def CheckDirectory(path):
    
    """
    Checking the path exists or creating the path.
    
    path: the path to create or save the file at.
    """
    parents_dir = os.path.join(os.path.split(path)[0])
    if not os.path.exists(parents_dir):
        os.makedirs(parents_dir)
        # pathlib.Path(path).mkdir()

def save_checkpoint(checkpoint_path, model):
    CheckDirectory(checkpoint_path)
    state = {'state_dict': model.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)
    return model

def update_checkpoint(checkpoint_path, model, optimizer, layer_name='conv'):
    checkpoint = torch.load(checkpoint_path)
    states_to_load = {}
    for name, param in checkpoint['state_dict'].items():
        if name.startswith(layer_name):
            states_to_load[name] = param

    # Construct a new state dict in which the layers we want
    # to import from the checkpoint is update with the parameters
    # from the checkpoint
    model_state = model.state_dict()
    model_state.update(states_to_load)
    model.load_state_dict(model_state)

class metrix:
    def __init__(self, n_classes=7):
        self.n_classes = n_classes
        self.tp_fp = np.zeros((self.n_classes))
        self.tp_fn = np.zeros((self.n_classes))
        self.tp = np.zeros((self.n_classes))

    def update(self, pred, labels):
        """
        Compute score over 6 classes
        """
        for i in range(self.n_classes):
            tp_fp = np.sum(pred == i)
            tp_fn = np.sum(labels == i)
            tp = np.sum((pred == i) * (labels == i))
            self.tp_fp[i] += tp_fp
            self.tp_fn[i] += tp_fn
            self.tp[i] += tp

    def mean_IoU(self):
        mean_iou = 0
        for i in range(self.n_classes):
            iou = self.tp[i] / ((self.tp_fp[i] + self.tp_fn[i] - self.tp[i]) + 1e-8)
            mean_iou += iou / 6
        return mean_iou

    def reset(self):
        self.tp_fp = np.zeros((self.n_classes))
        self.tp_fn = np.zeros((self.n_classes))
        self.tp = np.zeros((self.n_classes))