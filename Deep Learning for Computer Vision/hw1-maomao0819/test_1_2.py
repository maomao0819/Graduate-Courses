import os
from tqdm import tqdm 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import utils
from model import VGG16_FCN32s, DEEPLAB
from dataset import ImageSegmantationDataset
import parser
from mean_iou_evaluate import mean_iou_score

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def val_1_2(model, valset_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    val_loss = 0
    val_correct = 0
    val_miou = 0
    metric = utils.metrix()
    with torch.no_grad():  # This will free the GPU memory used for back-prop
        tqdm_loop = tqdm((valset_loader), total=len(valset_loader))
        for data, target in tqdm_loop:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_loss = criterion(output, target).item()
            val_loss += batch_loss  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(target.view_as(pred)).sum().item() / 512 / 512
            val_correct += batch_correct
            batch_miou = mean_iou_score(
                pred.view_as(target).cpu().detach().numpy(), target.cpu().detach().numpy(), log=False
            )
            val_miou += batch_miou * float(data.shape[0])
            metric.update(pred.view_as(target).cpu().detach().numpy(), target.cpu().detach().numpy())
            tqdm_loop.set_postfix(loss=batch_loss, acc=float(batch_correct) / float(data.shape[0]), miou=batch_miou)

    val_loss /= len(valset_loader.dataset)
    val_miou /= len(valset_loader.dataset)
    print(
        "\nVal set: Average loss: {:.5f}, Accuracy: {}/{} ({:.2f}%), MIoU:{:.5f}\n".format(
            val_loss,
            val_correct,
            len(valset_loader.dataset),
            100.0 * val_correct / len(valset_loader.dataset),
            # val_miou,
            metric.mean_IoU(),
        )
    )
    return val_loss, val_miou, metric.mean_IoU()

def test_1_2(checkpoint_path, model, dataloader):
    model = model.to(device)
    model = utils.load_checkpoint(checkpoint_path, model)
    test_loss, test_batch_miou, test_total_miou = val_1_2(model, dataloader)
    print('miou', test_total_miou)

if __name__=='__main__':

    args = parser.arg_parse_1_2()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_transform = {
        "val": transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
    }

    checkpoint_path = args.load
    if args.model_index == 0 or 'VGG' in checkpoint_path or 'FCN' in checkpoint_path:
        model = VGG16_FCN32s().to(device)
    else:
        model = DEEPLAB().to(device)
    validation_path = os.path.join(args.data_path, 'validation')
    valset = ImageSegmantationDataset(root=validation_path, transform=image_transform['val'])
    valset_loader = DataLoader(valset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_1_2(checkpoint_path, model, valset_loader)