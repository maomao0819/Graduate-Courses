import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import parser
from dataset import ImageClassificationDataset
from model import Resnet50Model
import utils
import torchvision.transforms as transforms

def evaluate(
    args,
    model: torch.nn.Module,
    dataloader: DataLoader,
):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    epoch_loss = 0.0
    epoch_correct = 0.0
    n_batch = len(dataloader)
    batch_pbar = tqdm((dataloader), total=n_batch, desc="Batch")
    with torch.no_grad():
        for batch_idx, data in enumerate(batch_pbar, 1):
            images, labels, _ = data
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            output = model(images)
            loss = criterion(output, labels)
            
            batch_loss = loss.item()
            epoch_loss += batch_loss  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            epoch_correct += batch_correct
            batch_pbar.set_description(f"Batch [{batch_idx}/{n_batch}]")
            batch_pbar.set_postfix(loss=f"{batch_loss:.4f}")

    performance = {}
    n_data = len(dataloader.dataset)
    performance["loss"] = epoch_loss / n_data
    performance["acc"] = epoch_correct / n_data
    return performance

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageClassificationDataset(image_path_root=args.image_dir, csv_file_path=args.csv_file, split=None, transform=TRANSFORM_IMG)

    # Use the torch dataloader to iterate through the dataset
    dataloader = DataLoader(
        dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model = Resnet50Model(fix_backbone=args.fix_backbone).to(args.device)
    model = utils.load_checkpoint(args.load, model)
    performance = evaluate(args, model, dataloader)
    print('loss:', performance['loss'], '\tacc:', performance["acc"])
    
if __name__ == "__main__":
    args = parser.arg_parse()
    main(args)