import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer
import parser
from dataset import ImageCaptionDataset
from model import make_model, make_std_mask, subsequent_mask
import utils

n_pixels = 197

def autoregress_decode(args, batch_size, model, dataloader, memory):
    word_generate_id = (torch.ones(batch_size, 1) * dataloader.dataset.bos_id).type(torch.int64).to(args.device)
    src_mask = torch.ones(batch_size, 1, 14*14).to(args.device)

    for _ in range(dataloader.dataset.max_len):

        # out: [batch_size, n_out_token, d_model]
        # prob: [batch_size, tgt_vocab]
        # next_word [batch_size]
        # word_generate_id: [batch_size, n_out_token]
        word_generate_id_mask = make_std_mask(word_generate_id, pad=dataloader.dataset.get_pad_id())
        out = model.decode(memory, src_mask, word_generate_id, word_generate_id_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data
        word_generate_id = torch.cat((word_generate_id, next_word.unsqueeze(-1)), dim=-1)
        # word_generate_id = torch.cat(
        #     [word_generate_id, torch.ones(batch_size, 1).type(torch.int64).fill_(next_word).to(args.device)],
        #     dim=1,
        # )
        stop_condition  = (next_word == 3) | (next_word == 13)
        if torch.all(stop_condition):
            break
    return word_generate_id

def predict(
    args,
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
):
    model.eval()
    prediction = {}
    n_batch = len(dataloader)
    tqdm_loop = tqdm((dataloader), total=n_batch)
    for batch_idx, data in enumerate(tqdm_loop, 1):
        with torch.no_grad():

            # image: [batch_size, 3, 224, 224]
            image = data["image"].to(args.device)
            batch_size = image.size(0)
            # memory: [batch_size, 196, 1024]
            memory = model.encode(image)

            word_generate_id = autoregress_decode(args, batch_size, model, dataloader, memory)
            word_generate_id = word_generate_id.tolist()

            # stop_tokens = [tokenizer.token_to_id('[EOS]'), tokenizer.token_to_id('.')]
            stop_tokens = [tokenizer.token_to_id('[EOS]')]
            for i in range(batch_size):
                tokens_id = word_generate_id[i]
                for stop_token in stop_tokens:
                    if stop_token in tokens_id:
                        tokens_id = tokens_id[:tokens_id.index(stop_token)+1]
                        
                prediction[data['basename'][i]] = tokenizer.decode(tokens_id, skip_special_tokens=True)
            tqdm_loop.set_description(f"Batch [{batch_idx}/{n_batch}]")

    utils.save_json(prediction, args.pred_path)
    return prediction


def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    config = resolve_data_config({}, model=timm.create_model("vit_large_patch16_224", pretrained=True))
    transform = create_transform(**config)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    trainset = ImageCaptionDataset(
        data_dir=args.data_dir, tokenizer=tokenizer, split_set="train", transform=transform
    )
    valset = ImageCaptionDataset(data_dir=args.data_dir, tokenizer=tokenizer, split_set="val", transform=transform)

    # Use the torch dataloader to iterate through the dataset

    val_loader = DataLoader(
        trainset,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = make_model(args.freeze, tokenizer.get_vocab_size(), args.n_layers, args.d_model, args.d_ff, args.n_heads, args.dropout).to(args.device)
    
    model = utils.load_checkpoint(args.load, model)

    predict(args, model, val_loader, tokenizer)


if __name__ == "__main__":
    args = parser.arg_parse(2)
    main(args)
