# DISCLAIMER: this is a easy to use + slimmed down + refactored version of the training code used in the ECCV paper: X2Face
# It should give approximately similar results to what is in the paper (e.g. the frontalised unwrapped face
# and that the driving portion of the network transforms this frontalised face into the given view).
# It should also give a good idea of how to train the network.

# (c) Olivia Wiles

from VoxCelebData_withmask import VoxCeleb
from UnwrappedFace import UnwrappedFaceWeightedAverage

import os
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from find_gpus import device, gpu_ids
from torchvision.transforms import ToTensor, Resize, Compose


def prepare_data(path, dim=256, num_sources=3):
    transform = Compose([Resize((dim, dim)), ToTensor()])
    face_list = []
    face_per_souce = []
    source_now = 0
    for sources in os.listdir(path):
        source_now += 1
        source_path = os.path.join(path, sources)
        for face in os.listdir(source_path):
            face_path = os.path.join(source_path, face)
            face = Image.open(face_path)
            face = transform(face)
            face = face.unsqueeze(0)
            face_per_souce.append(face)
        if source_now == num_sources:
            face_per_souce = torch.cat(face_per_souce, dim=0)
            face_list.append(face_per_souce.unsqueeze(1))
            face_per_souce = []
    face_per_souce = torch.cat(face_per_souce, dim=0)
    face_list.append(face_per_souce.unsqueeze(1))

    return face_list


def main(args):
    model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3,
                                         inner_nc=args.inner_nc, skip_net_backbone=args.skip_net_backbone,
                                         no_skip_net_backbone=args.no_skip_net_backbone,)

    if args.pretrained_weight:
        checkpoint_file = torch.load(args.pretrained_weight)
        model.load_state_dict(checkpoint_file['state_dict'])
        del checkpoint_file

    # Check if there are multiple GPUs
    if torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model)
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model = model.cuda()
    model.eval()

    # Save embeddings
    embeddings = {}
    for ID in tqdm(os.listdir(args.input_path)):
        # Load face
        face_path = os.path.join(args.input_path, ID)
        face_path = os.path.join(face_path, '1.6')
        face_imgs = prepare_data(face_path, dim=args.dim, num_sources=3)

        # Get embedding
        with torch.no_grad():
            imgs_total, confidents_total = None, None
            for face in face_imgs:
                face = face.cuda()
                _, imgs, confidents = model.get_unwrapped(*face)
                if imgs_total is None:
                    imgs_total = imgs
                    confidents_total = confidents
                # else:
                #     imgs_total += imgs
                #     confidents_total += confidents
        imgs_final = imgs_total / confidents_total.expand_as(imgs_total)
        # import cv2
        # print(ID)
        # imgs_final = imgs_final.cpu().numpy()
        # imgs_final = imgs_final.transpose(0, 2, 3, 1)
        # imgs_final = imgs_final * 255
        # imgs_final = imgs_final.astype(np.uint8)
        # imgs_final = imgs_final[0]
        # imgs_final = cv2.cvtColor(imgs_final, cv2.COLOR_BGR2RGB)
        # cv2.imshow('img', imgs_final)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        embeddings[ID] = imgs_final.cpu().numpy()

    # Save embeddings
    with open(os.path.join(args.results_folder, 'embedded_faces.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UnwrappedFace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_views', type=int, default=2, help='Num views')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of image')
    parser.add_argument('--model_type', type=str, default='UnwrappedFaceSampler_from1view')
    parser.add_argument('--inner_nc', type=int, default=128)
    parser.add_argument('--pretrained_weight', type=str, default='')
    parser.add_argument('--input_path', type=str, default='./faces/')
    parser.add_argument('--results_folder', type=str, default='./results/')
    parser.add_argument('--skip_net_backbone', type=str, choices=['unet_3+', 'unet_128', 'unet_256'], default='unet_3+')
    parser.add_argument('--no_skip_net_backbone', type=str, choices=['unet_128', 'unet_256'], default='unet_128')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
