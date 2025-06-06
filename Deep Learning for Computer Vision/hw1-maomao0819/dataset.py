import os
import glob
import numpy as np
from PIL import Image
import imageio
import torch
from torch.utils.data import Dataset
from viz_mask import read_masks

class ImageClassificationDataset(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the image dataset """
        self.filenames = []
        self.labels = []
        self.transform = transform
        self.len = 0
        
        # read filenames
        filenames = glob.glob(os.path.join(root, '*.png'))
        for filename in filenames:
            basename = os.path.splitext(os.path.basename(filename))[0]
            label = int(basename.split('_')[0])
            self.filenames.append(filename)
            self.labels.append(label)
        
        self.len = len(self.labels)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        filename = self.filenames[index]
        image = Image.open(filename)
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class ImageClassificationPredictDataset(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the image dataset """
        self.filenames = []
        self.transform = transform
        self.len = 0
        if root.endswith('.png'):
            self.filenames.append(image_dir)
        else:
            # read filenames
            self.filenames = glob.glob(os.path.join(root, '*.png'))
        
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        filename = self.filenames[index]
        basename = os.path.basename(filename)
        image = Image.open(filename)
        if self.transform is not None:
            image = self.transform(image)
        return image, basename

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

colorMap = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]
colorMap_to_label = np.zeros(256**3)
for idx, cm in enumerate(colorMap):
    colorMap_to_label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = idx

class ImageSegmantationDataset(Dataset):
    def __init__(self, root, transform=None):
        """Intialize the image dataset"""
        self.images = []
        self.labels = []
        self.transform = transform
        self.len = 0

        images_extension = "sat.jpg"
        labels_extension = "mask.png"
        # read filenames
        images_path = glob.glob(os.path.join(root, f"*{images_extension}"))
        labels_path = glob.glob(os.path.join(root, f"*{labels_extension}"))

        images_index = [int(os.path.basename(image_path).split("_")[0]) for image_path in images_path]
        labels_index = [int(os.path.basename(label_path).split("_")[0]) for label_path in labels_path]

        index = list(set(images_index).intersection(set(labels_index)))
        index.sort()

        for idx in index:
            self.images.append(os.path.join(root, f"{idx:04d}_{images_extension}"))
            self.labels.append(os.path.join(root, f"{idx:04d}_{labels_extension}"))

        self.len = len(self.labels)

    def image2label(self, image):
        img = np.array(image, dtype="int32")
        idx = (img[:, :, 0] * 256 + img[:, :, 1]) * 256 + img[:, :, 2]
        return np.array(colorMap_to_label[idx], dtype="int64")

    def __getitem__(self, index):
        """Get a sample from the dataset"""
        images = self.images[index]
        images = Image.open(images).convert("RGB")
        labels = self.labels[index]
        # labels = Image.open(labels)
        # labels = np.array(labels)
        # labels = self.image2label(labels)
        # labels = torch.from_numpy(labels)
        labels = imageio.imread(labels)
        labels = read_masks(labels, labels.shape).astype(np.int64)
        labels = torch.from_numpy(labels)
        if self.transform is not None:
            images = self.transform(images)
        # print(images.size())
        # print(images[:, :, 0], images[:, :, 1], images[:, :, 2])
        # print(labels.size())
        return images, labels

    def __len__(self):
        """Total number of samples in the dataset"""
        return self.len

class ImageSegmantationPredictionDataset(Dataset):
    def __init__(self, root, transform=None):
        """Intialize the image dataset"""
        self.images_path = []
        self.transform = transform
        self.len = 0

        images_extension = ".jpg"

        if root.endswith(images_extension):
            self.images_path.append(root)
        else:
            # read filenames
            self.images_path = glob.glob(os.path.join(root, f"*{images_extension}"))

        self.len = len(self.images_path)

    def __getitem__(self, index):
        """Get a sample from the dataset"""
        image_path = self.images_path[index]
        basename = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, basename

    def __len__(self):
        """Total number of samples in the dataset"""
        return self.len