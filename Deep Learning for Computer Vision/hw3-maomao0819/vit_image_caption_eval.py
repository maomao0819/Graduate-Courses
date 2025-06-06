import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer
import parser
from dataset import ImageCaptionDataset
from model import make_model
import utils
from p2_evaluate import evaluation
from vit_image_caption_pred import predict
n_pixels = 197

def eval(args, model, dataloader, tokenizer, split_set='val'):
    prediction = predict(args, model, dataloader, tokenizer)
    annotations = utils.load_json(os.path.join(args.data_dir, f'{split_set}.json'))
    images_root = os.path.join(args.data_dir, 'images', split_set)
    evaluation(prediction, annotations, images_root)

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    config = resolve_data_config({}, model=timm.create_model('vit_large_patch16_224', pretrained=True))
    transform = create_transform(**config)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    trainset = ImageCaptionDataset(data_dir=args.data_dir, tokenizer=tokenizer, split_set='train', transform=transform)
    valset = ImageCaptionDataset(data_dir=args.data_dir, tokenizer=tokenizer, split_set='val', transform=transform)

    # Use the torch dataloader to iterate through the dataset

    val_loader = DataLoader(
        valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model = make_model(args.freeze, tokenizer.get_vocab_size(), args.n_layers, args.d_model, args.d_ff, args.n_heads, args.dropout).to(args.device)
    model = utils.load_checkpoint(args.load, model)
    eval(args, model, val_loader, tokenizer)
    

if __name__ == "__main__":
    args = parser.arg_parse(2)
    main(args)