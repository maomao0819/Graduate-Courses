import os
import numpy as np
import torch

from tqdm import tqdm
import clip
from dataset import ImageClassificationDataset
from torch.utils.data import DataLoader

import parser
import utils


def run_one_epoch(
    args,
    dataloader,
    model, 
    labels_texts,
):
    if args.prompt_text_type == 1:
        text = torch.cat([clip.tokenize(f"a photo of a {label}") for label in labels_texts]).to(args.device)
    elif args.prompt_text_type == 2:
        text = torch.cat([clip.tokenize(f"This is a {label} image.") for label in labels_texts]).to(args.device)
    else:
        text = torch.cat([clip.tokenize(f"No {label}, no score.") for label in labels_texts]).to(args.device)
    

    n_batch = len(dataloader)
    batch_pbar = tqdm((dataloader), total=n_batch)
    # epoch_correct = 0
    predictions = []
    basenames = []
    for batch_idx, data in enumerate(batch_pbar, 1):
        # Prepare the inputs
        # [batch_size, 3, 224, 224]
        image = data[0].to(args.device)
        # label = data[1]
        basename = data[2]

        basenames.extend(basename)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_image, logits_text = model(image, text)
            probs = logits_image.softmax(dim=-1).cpu().numpy()

        prediction = np.argmax(probs, axis=1)
        predictions.extend(prediction)

        # batch_correct = np.sum(prediction == np.array(label))
        # epoch_correct += batch_correct

        if args.analysis_top5:
            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            values, indices = similarity.topk(5, dim=1)

            # Print the result
            print("\nTop predictions:\n")
            for value, index in zip(values, indices):
                print(f"{labels_texts[index]:>16s}: {100 * value.item():.2f}%")
        
        batch_pbar.set_description(f'Batch [{batch_idx}/{n_batch}]')
        # batch_pbar.set_postfix(acc=float(batch_correct) / float(len(label)))

    # accuracy = epoch_correct / len(dataloader.dataset)
    # print(f"{100 * accuracy:.2f}%")

    utils.predict_to_csv(basenames, predictions, args.predict_path)

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    image_dir = os.path.join(args.image_dir)
    id2label_path = os.path.join(args.id2label_path)
    id2label = utils.load_json(id2label_path)
    labels_texts = list(id2label.values())
    

    # Load the model
    model, preprocess = clip.load('ViT-B/32', args.device)

    imageset = ImageClassificationDataset(image_path_root=image_dir, preprocess=preprocess)

    # Use the torch dataloader to iterate through the dataset
    dataloader = DataLoader(
        imageset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    
    run_one_epoch(args, dataloader, model, labels_texts)


if __name__ == "__main__":
    args = parser.arg_parse(1)
    main(args)