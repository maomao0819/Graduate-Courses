import os
import pandas as pd
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
from tokenizers import Tokenizer
import utils
import torchvision.transforms as transforms
class ImageClassificationDataset(Dataset):
    def __init__(self, image_path_root, preprocess=None):
        """ Intialize the image dataset """
        self.filenames = []
        self.labels = []
        self.preprocess = preprocess
        self.basenames = []
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.PILToTensor()])
        # read filenames
        filenames = glob.glob(os.path.join(image_path_root, '*.png'))
        for filename in filenames:
            basename = os.path.basename(filename)
            label = int(os.path.splitext(basename)[0].split('_')[0])

            self.filenames.append(filename)
            self.labels.append(label)
            self.basenames.append(basename)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        filename = self.filenames[index]
        image_ori = Image.open(filename)
        label = self.labels[index]
        if self.transform is not None:
            img_tensor = self.transform(image_ori)
        if self.preprocess is not None:
            image = self.preprocess(image_ori)
        basename = self.basenames[index]
        return image, label, basename, img_tensor

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.labels)

class ImageCaptionLabelDataset(Dataset):
    def __init__(self, data_dir, tokenizer, split_set='train', transform=None):
        """ Intialize the image dataset """
        self.transform = transform
        self.tokenizer = tokenizer
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.ignore_idx = self.pad_id
        self.max_len = 50
        # read filenames
        json_path = os.path.join(data_dir, f"{split_set}.json")
        data = utils.load_json(json_path)
        df_annotations = pd.DataFrame(data["annotations"])
        df_images = pd.DataFrame(data["images"])
        df_images = df_images.rename(columns={'id': 'image_id'})
        dataframe = pd.merge(df_annotations, df_images)
        image_path_root = os.path.join(data_dir, "images", split_set)

        self.basenames = dataframe['file_name'].apply(lambda filename: os.path.splitext(filename)[0]).tolist()
        self.filenames = dataframe['file_name'].apply(lambda filename: os.path.join(image_path_root, filename)).tolist()
        self.captions = dataframe['caption'].tolist()
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        filename = self.filenames[index]
        image = Image.open(filename).convert('RGB')
        caption = self.captions[index]
        if self.transform is not None:
            image = self.transform(image)
        basename = self.basenames[index]
        data = {}
        data['image'] = image
        data['caption'] = caption
        data['basename'] = basename
        return data

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.captions)
    
    def get_pad_id(self):
        return self.pad_id

    def collate_fn(self, samples):
        batch = {key: [sample[key] for sample in samples] for key in samples[0]}
        batch['image'] = torch.stack(batch['image'])
        batch['tokens_id'], batch['tokens_id_next'] = self.encode_batch(batch['caption'])
        batch['tokens_id'] = torch.LongTensor(batch["tokens_id"])
        batch['tokens_id_next'] = torch.LongTensor(batch["tokens_id_next"])
        batch["len"] = torch.tensor([min(len(caption_id), self.max_len) for caption_id in batch["tokens_id"]])
        batch_seq_len = torch.max(batch["len"])
        batch['mask'] = batch['tokens_id_next'].gt(self.pad_id)

        # tgt_list = []
        # for caption in batch['caption']:
        #     processed = torch.cat([self.bos_id, torch.tensor(self.tokenizer.encode(caption).ids, dtype=torch.int64, 
        #         device='cuda', ), self.eos_id, ], 0, )
        #     tgt_list.append(pad(processed, (0, self.max_len - len(processed)), value=self.pad_id, ))
        # tgt = torch.stack(tgt_list)
        return batch

    def encode_batch(self, batch_captions, to_len: int = None):
        batch_ids = [self.tokenizer.encode(captions).ids[:-1] for captions in batch_captions]
        batch_ids_next = [self.tokenizer.encode(captions).ids[1:] for captions in batch_captions]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        padded_ids = self.pad_to_len(batch_ids, to_len, self.pad_id)
        padded_ids_next = self.pad_to_len(batch_ids_next, to_len, self.pad_id)
        return padded_ids, padded_ids_next

    def pad_to_len(self, seqs, to_len: int, padding: int):
        paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
        return paddeds

class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir, tokenizer, split_set='train', transform=None):
        """ Intialize the image dataset """
        self.transform = transform
        self.tokenizer = tokenizer
        # self.pad_id = 0
        # self.unk_id = 1
        # self.bos_id = 2
        # self.eos_id = 3
        self.pad_id = tokenizer.token_to_id('[PAD]')
        # self.unk_id = tokenizer.token_to_id('[UNK]')
        self.bos_id = tokenizer.token_to_id('[BOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')
        self.ignore_idx = self.pad_id
        self.max_len = 50
        # read filenames
        json_path = os.path.join(data_dir, f"{split_set}.json")
        data = utils.load_json(json_path)

        df_images = pd.DataFrame(data["images"])

        image_path_root = os.path.join(data_dir, "images", split_set)

        self.basenames = df_images['file_name'].apply(lambda filename: os.path.splitext(filename)[0]).tolist()
        self.filenames = df_images['file_name'].apply(lambda filename: os.path.join(image_path_root, filename)).tolist()

                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        filename = self.filenames[index]
        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        basename = self.basenames[index]
        data = {}
        data['image'] = image
        data['basename'] = basename
        return data

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.filenames)
    
    def get_pad_id(self):
        return self.pad_id

class ImageDataset(Dataset):
    def __init__(self, data_dir, tokenizer, transform=None):
        """ Intialize the image dataset """
        self.transform = transform
        self.tokenizer = tokenizer
        self.transform_ori = transforms.Compose([transforms.Resize((224, 224)), transforms.PILToTensor()])
        self.pad_id = tokenizer.token_to_id('[PAD]')
        # self.unk_id = tokenizer.token_to_id('[UNK]')
        self.bos_id = tokenizer.token_to_id('[BOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')
        self.ignore_idx = self.pad_id
        self.max_len = 50
        
        # read filenames
        self.filenames = glob.glob(os.path.join(data_dir, '*.jpg'))

                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        filename = self.filenames[index]
        image = Image.open(filename)
        if self.transform_ori is not None:
            image_ori = self.transform_ori(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_ori

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.filenames)
    
    def get_pad_id(self):
        return self.pad_id