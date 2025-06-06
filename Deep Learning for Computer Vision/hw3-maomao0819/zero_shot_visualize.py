import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip
from dataset import ImageClassificationDataset
from torch.utils.data import DataLoader

import parser
import utils

def sample_prediction(
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
    
    batch_data = next(iter(dataloader))

    # Prepare the inputs
    # [batch_size, 3, 224, 224]
    image = batch_data[0].to(args.device)
    label = batch_data[1]
    basename = batch_data[2]
    image_ori = batch_data[-1]


    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)


    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    batch_values, batch_indices = similarity.topk(5, dim=1)

    # Print the result
    predictions = {}
    for idx, (values, indices) in enumerate(zip(batch_values, batch_indices)):
        predictions[idx] = {}
        predictions[idx]['image'] = image_ori[idx]
        predictions[idx]['pred'] = []
        predictions[idx]['prob'] = []
        predictions[idx]['label_id'] = -1
        for value, index in zip(values, indices):
            if index == label[idx]:
                predictions[idx]['label_id'] = idx
            if args.prompt_text_type == 1:
                predictions[idx]['pred'].append(f"a photo of a {labels_texts[index]}")
            elif args.prompt_text_type == 2:
                predictions[idx]['pred'].append(f"This is a {labels_texts[index]} image.")
            else:
                predictions[idx]['pred'].append(f"No {labels_texts[index]}, no score.")
            predictions[idx]['prob'].append(value.item())
            # print(f"{labels_texts[index]:>16s}: {100 * value.item():.2f}%")
    return predictions

def visualize_prediction(predictions):
    width = 0.75 # the width of the bars 
    for idx in predictions.keys():
        prediction = predictions[idx]
        n_data = len(prediction['prob'])
        n_ticks = n_data + 1
        color = ['blue'] * 5
        color[0] = 'red'
        if prediction['label_id'] != -1:
            color[prediction['label_id']] = 'green'
        y_length = np.array(prediction['prob']) * n_data
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ind = np.arange(len(y_length))  # the x locations for the groups
        ind = ind[::-1]
        ax[0].imshow(predictions[idx]['image'].cpu().detach().permute(1, 2, 0).numpy())
        ax[0].set_xticklabels('', minor=False)
        ax[0].set_yticklabels('', minor=False)
        ax[1].barh(ind, y_length, width, color=color, align='edge')
        ax[1].set_yticks(ind + width / 2)
        ax[1].set_yticklabels('', minor=False)
        for bar, text in zip(ax[1].patches, prediction['pred']):
            ax[1].text(0.1, bar.get_y() + bar.get_height() / 2, text, color = 'black', ha = 'left', va = 'center') 
        plt.xticks(np.arange(n_ticks), np.arange(0, 101, 20))
        plt.title(f'correct probability: {100 * prediction["prob"][0]:.2f}%')
        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/sample{idx+1}.png")

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
    
    predictions = sample_prediction(args, dataloader, model, labels_texts)
    visualize_prediction(predictions)


if __name__ == "__main__":
    args = parser.arg_parse(1)
    main(args)