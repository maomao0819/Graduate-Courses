from test_1_1 import val_1_1, test_1_1
import pandas as pd
import numpy as np
import utils
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import MyImageClassificationNet, Resnet50Model
from dataset import ImageClassificationDataset
import parser
import sklearn
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.style.use('seaborn')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if __name__=='__main__':
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

    checkpoint_path = args.load
    if args.model_index == 0 or 'Mine_CNN' in checkpoint_path:
        model_name = 'Mine_CNN'
        model = MyImageClassificationNet(n_classes=50).to(device)
    else:
        model_name = 'Pretrain_Resnet'
        model = Resnet50Model(n_classes=50).to(device)
    validation_path = os.path.join(args.data_path, 'val_50')
    valset = ImageClassificationDataset(root=validation_path, transform=image_transform['val'])
    valset_loader = DataLoader(valset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = utils.load_checkpoint(checkpoint_path, model)
    # test_1_1(checkpoint_path, model, valset_loader)
    logits, labels = val_1_1(model, valset_loader, feature_visualize=True)
    logits = np.array(logits)

    pca = PCA(n_components=2)
    pca.fit(logits)
    logits_PCA = pca.transform(logits)
    # df = pd.DataFrame()
    # df['label'] = labels
    # df['X'] = logits_pca[:, 0]
    # df['Y'] = logits_pca[:, 1]
    # print(df.head())
    # plot = sns.scatterplot(x='X', y='Y', data=df, hue='label')
    # fig = plot.get_figure()
    # fig.savefig("out.png") 
    plt.scatter(logits_PCA[:, 0], logits_PCA[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 50), marker='.')
    plt.colorbar(ticks=range(50))
    plt.savefig(f'{model_name}_PCA.png')
    plt.clf()

    logits_TSNE = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(logits)
    plt.scatter(logits_TSNE[:, 0], logits_TSNE[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 50), marker='.')
    plt.colorbar(ticks=range(50))

    plt.savefig(f'{model_name}_TSNE.png')