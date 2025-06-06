import os
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FaceImageDataset(Dataset):
    def __init__(self, dir, transform=None):
        """ Intialize the image dataset """
        self.filenames = []
        self.transform = transform
        
        # read filenames
        self.filenames = glob.glob(os.path.join(dir, '*.png'))
 
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        filename = self.filenames[index]
        image = Image.open(filename)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.filenames)

class DigitImageDataset(Dataset):
    def __init__(self, dir, mode='train', transform=None):
        """ Intialize the image dataset """
        self.image_paths = []
        self.transform = transform
        self.labels = []

        csv_path = os.path.join(dir, f'{mode}.csv')
        df = pd.read_csv(csv_path)
        image_names = df.iloc[:, 0].tolist()
        self.image_paths = [os.path.join(dir, 'data', image_name) for image_name in image_names]
        self.labels = df.iloc[:, 1].tolist()
 
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.labels)

class DigitImagePredictDataset(Dataset):
    def __init__(self, dir, transform=None):
        """ Intialize the image dataset """
        self.filenames = []
        self.transform = transform
        
        # read filenames
        self.filenames = glob.glob(os.path.join(dir, '*.png'))
 
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        filename = self.filenames[index]
        basename = os.path.basename(filename)
        image = Image.open(filename).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, basename

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.filenames)