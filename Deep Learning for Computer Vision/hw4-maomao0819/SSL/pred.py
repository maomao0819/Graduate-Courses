import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import parser
from dataset import ImageClassificationDataset
from model import Resnet50Model
import utils
import torchvision.transforms as transforms
import pandas as pd

def predict(
    args,
    model: torch.nn.Module,
    dataloader: DataLoader,
):
    model.eval()
    filenames = []
    predictions = []
    n_batch = len(dataloader)
    batch_pbar = tqdm((dataloader), total=n_batch, desc="Batch")
    with torch.no_grad():
        for data in batch_pbar:
            images, _, basenames = data
            images = images.to(args.device)
            output = model(images)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            filenames.extend(basenames)
            # print(np.shape(pred.detach().cpu().numpy().squeeze(-1)))
            predictions.extend(pred.detach().cpu().numpy().squeeze(-1).tolist())
    label2index = dataloader.dataset.label2index
    index2label = {index: label for label, index in label2index.items()}
    predictions = [index2label[prediction] for prediction in predictions]
    df = pd.DataFrame(columns=['id', 'filename', 'label'])
    df['filename'] = filenames
    df['label'] = np.squeeze(predictions)
    # df = df.sort_values(by=['filename']).reset_index(drop=True)
    df['id'] = range(0, len(df))
    df.to_csv(args.pred_file, index=False)
    
    return predictions

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
    predict(args, model, dataloader)

    
if __name__ == "__main__":
    args = parser.arg_parse()
    main(args)