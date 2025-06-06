import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

class ImageClassificationDataset(Dataset):
    def __init__(self, image_path_root, csv_file_path, split=None, transform=None):
        """ Intialize the image dataset """
        if split != None:
            data = pd.read_csv(os.path.join(csv_file_path, f'{split}.csv'))
            basenames = data['filename']
            self.basenames = basenames.tolist()
            self.filenames = basenames.apply(lambda basename: os.path.join(image_path_root, split, basename)).tolist()
        else:
            data = pd.read_csv(csv_file_path)
            basenames = data['filename']
            self.basenames = basenames.tolist()
            self.filenames = basenames.apply(lambda basename: os.path.join(image_path_root, basename)).tolist()
        self.labels = data['label'].tolist()
        self.transform = transform
        # labels_type = list(set(self.labels))
        # labels_type.sort()
        # self.label2index = {label: index for index, label in enumerate(labels_type)}
        self.label2index = {'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 
            'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 
            'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 
            'Fork': 24, 'Glasses': 25, 'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 
            'Marker': 33, 'Monitor': 34, 'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 
            'Pencil': 43, 'Postit_Notes': 44, 'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 
            'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 
            'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}
           
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        filename = self.filenames[index]
        image = Image.open(filename).convert('RGB')
        label = self.label2index[self.labels[index]]
        basename = self.basenames[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, basename

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.labels)
