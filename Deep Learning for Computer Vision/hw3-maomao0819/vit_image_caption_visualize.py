import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer
import parser
from dataset import ImageDataset
from model import make_model
import utils
from p2_evaluate import evaluation
from model import make_model, make_std_mask, subsequent_mask
from vit_image_caption_pred import autoregress_decode
import matplotlib.pyplot as plt
n_pixels = 197


def visualize(args, model, dataloader, tokenizer, split_set='val'):
    batch_data = next(iter(dataloader))

    # image: [batch_size, 3, 224, 224]
    image = batch_data[0].to(args.device)
    image_ori = batch_data[1]
    batch_size = image.size(0)
    # memory: [batch_size, 196, 1024]
    memory = model.encode(image)

    # word_generate_id: [batch_size, seq_len]
    word_generate_id = autoregress_decode(args, batch_size, model, dataloader, memory)
    word_generate_id = word_generate_id.tolist()

    # stop_tokens = [tokenizer.token_to_id('[EOS]'), tokenizer.token_to_id('.')]
    tokens_ids = []
    stop_tokens = [tokenizer.token_to_id('[EOS]')]
    for batch_id in range(batch_size):

        tokens_id = word_generate_id[batch_id]
        for stop_token in stop_tokens:
            if stop_token in tokens_id:
                tokens_id = tokens_id[:tokens_id.index(stop_token)+1]
        # tokens = tokenizer.decode(tokens_id, skip_special_tokens=True)

        # [seq_len, 196]
        cross_attn_maps = torch.mean(model.decoder.layers[-1].cross_attn.attn[batch_id], 0,)[:len(tokens_id)]
        plt.imshow(image_ori[batch_id].permute(1,2,0))
        plt.title("[BOS]")
        os.makedirs("images", exist_ok=True)
        os.makedirs(f"images/object{batch_id+1}", exist_ok=True)
        plt.savefig(f"images/object{batch_id+1}/token0.png")
        plt.clf()
        for token_idx in range(len(tokens_id)-1):
            token = tokenizer.decode([tokens_id[token_idx+1]], skip_special_tokens=False)
            # [14, 14]
            heatmap = cross_attn_maps[token_idx+1].reshape((14, 14)).detach()
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
            heatmap = torch.nn.functional.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
            heatmap = heatmap.squeeze(0).squeeze(0)
            heatmap = heatmap.cpu().detach().numpy()
            plt.title(token)
            plt.imshow(image_ori[batch_id].permute(1,2,0))
            plt.imshow(heatmap, alpha=0.6, cmap='rainbow')
            
            plt.savefig(f"images/object{batch_id+1}/token{token_idx+1}.png")
            plt.clf()

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    config = resolve_data_config({}, model=timm.create_model('vit_large_patch16_224', pretrained=True))
    transform = create_transform(**config)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    valset = ImageDataset(data_dir=args.data_dir, tokenizer=tokenizer, transform=transform)

    # Use the torch dataloader to iterate through the dataset

    val_loader = DataLoader(
        valset, batch_size=5, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model = make_model(args.freeze, tokenizer.get_vocab_size(), args.n_layers, args.d_model, args.d_ff, args.n_heads, args.dropout).to(args.device)
    model = utils.load_checkpoint(args.load, model)

    visualize(args, model, val_loader, tokenizer)
    

if __name__ == "__main__":
    args = parser.arg_parse(3)
    main(args)