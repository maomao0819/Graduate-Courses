import numpy as np
import utils
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import model
from dataset import DigitImageDataset
import parser
import sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from typing import List

plt.style.use('seaborn')

def get_latent(
    args,
    model: torch.nn.Module,
    dataloader: DataLoader,    
    domain: str = 'source',
    intents: List = [],
    class_labels: List = [],
    domain_labels: List = [],
):
    n_batch = len(dataloader)
    batch_pbar = tqdm((dataloader), total=n_batch, desc="Batch")
    with torch.no_grad():
        for data in batch_pbar:
            alpha = 0
            image = data[0].to(args.device)
            class_label = data[1].to(args.device)
            intent = model(image, alpha)
            batch_size = len(image)
            if domain == 'source':
                domain_label = torch.ones(batch_size).long().to(args.device)
            else:
                domain_label = torch.zeros(batch_size).long().to(args.device)
            intents.append(intent)
            class_labels.append(class_label)
            domain_labels.append(domain_label)
    return intents, class_labels, domain_labels

def main(args):

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    image_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    
    source_dataset = "mnistm"
    if args.task == "A":
        target_dataset = "svhn"
    else:
        target_dataset = "usps"

    source_dir = os.path.join(args.data_dir, source_dataset)
    target_dir = os.path.join(args.data_dir, target_dataset)
    source_valset = DigitImageDataset(dir=source_dir, mode="val", transform=image_transform)
    target_valset = DigitImageDataset(dir=target_dir, mode="val", transform=image_transform)
    # Use the torch dataloader to iterate through the dataset
    source_val_loader = DataLoader(
        source_valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    target_val_loader = DataLoader(
        target_valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    ExtractedDANNModel = model.ExtractedDANN().to(args.device)
    ExtractedDANNModel = utils.load_checkpoint(args.load, ExtractedDANNModel)

    intents = []
    class_labels = []
    domain_labels = []
    intents, class_labels, domain_labels = get_latent(args, ExtractedDANNModel, source_val_loader, 'source', intents, class_labels, domain_labels)
    intents, class_labels, domain_labels = get_latent(args, ExtractedDANNModel, target_val_loader, 'target', intents, class_labels, domain_labels)

    intents = torch.cat(intents).cpu().numpy()
    class_labels = torch.cat(class_labels).cpu().numpy()
    domain_labels = torch.cat(domain_labels).cpu().numpy()

    intent_TSNE = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50, n_jobs=4).fit_transform(intents)

    plt.scatter(intent_TSNE[:, 0], intent_TSNE[:, 1], c=class_labels, cmap=plt.cm.get_cmap("jet", 50), marker='.')
    plt.colorbar(ticks=range(50))
    plt.savefig(f'DANN_class_TSNE.png')
    plt.clf()

    plt.scatter(intent_TSNE[:, 0], intent_TSNE[:, 1], c=domain_labels, cmap=plt.cm.get_cmap("jet", 50), marker='.')
    plt.colorbar(ticks=range(50))
    plt.savefig(f'DANN_domain_TSNE.png')

if __name__ == "__main__":
    args = parser.arg_parse(3)
    main(args)