import os
from tqdm import tqdm 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import utils
from model import MyImageClassificationNet, Resnet50Model
from dataset import ImageClassificationDataset, ImageClassificationPredictDataset
import parser

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

args = parser.arg_parse_1_1()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_transform = {
    'val': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

def val_1_1(model, valset_loader, feature_visualize=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    val_loss = 0
    val_correct = 0
    logits_list = []
    target_list = []
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        tqdm_loop = tqdm((valset_loader), total=len(valset_loader))
        for data, target in tqdm_loop:
            data, target = data.to(device), target.to(device)
            output = model(data)['out']
            logits = model(data)['logits']
            logits_list.extend(logits.cpu().detach().numpy())
            target_list.extend(target.cpu().detach().numpy())
            batch_loss = criterion(output, target).item() 
            val_loss += batch_loss # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            val_correct += batch_correct
            tqdm_loop.set_postfix(loss=batch_loss, acc=float(batch_correct) / float(data.shape[0]))
    if feature_visualize:
        return logits_list, target_list

    val_loss /= len(valset_loader.dataset)
    print('\nVal set: Average loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        val_loss, val_correct, len(valset_loader.dataset),
        100. * val_correct / len(valset_loader.dataset)))
    
    return val_loss, val_correct / len(valset_loader.dataset)

def test_1_1(checkpoint_path, model, dataloader):
    model = model.to(device)
    model = utils.load_checkpoint(checkpoint_path, model)
    test_loss, test_acc = val_1_1(model, dataloader)
    print('Acc:', test_acc)

if __name__=='__main__':
    

    checkpoint_path = args.load
    if args.model_index == 0 or 'Mine_CNN' in checkpoint_path:
        model = MyImageClassificationNet(n_classes=50).to(device)
    else:
        model = Resnet50Model(n_classes=50).to(device)
    validation_path = os.path.join(args.data_path, 'val_50')
    valset = ImageClassificationDataset(root=validation_path, transform=image_transform['val'])
    valset_loader = DataLoader(valset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_1_1(checkpoint_path, model, valset_loader)