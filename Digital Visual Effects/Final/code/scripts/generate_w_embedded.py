import os
import argparse
import pickle
import numpy as np
import cv2
from PIL import Image
from find_gpus import device, gpu_ids
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Resize, Compose
from torch.autograd import Variable

from UnwrappedFace import UnwrappedFaceWeightedAverage


def main(args):
    # Load embeddings
    with open(args.embedding_path, 'rb') as f:
        embeddings = pickle.load(f)

    inferenced_name = args.inferenced_name
    inferenced_embedding = embeddings[inferenced_name]

    # Draw inference face
    # inferenced_face = inferenced_embedding[0]
    # inferenced_face = inferenced_face.transpose(1, 2, 0)
    # inferenced_face = inferenced_face * 255
    # inferenced_face = inferenced_face.astype(numpy.uint8)
    # inferenced_face = cv2.cvtColor(inferenced_face, cv2.COLOR_BGR2RGB)
    # cv2.imshow('inferenced', inferenced_face)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3,
                                         inner_nc=args.inner_nc, skip_net_backbone=args.skip_net_backbone,
                                         no_skip_net_backbone=args.no_skip_net_backbone)

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

    # Get driving frame and feed into model
    inferenced_embedding = Variable(torch.from_numpy(inferenced_embedding)).cuda()
    transform = Compose([Resize((args.dim, args.dim)), ToTensor()])

    generated_frames = []
    f_name = list(os.listdir(args.driving_video))
    # sort the frame names
    f_name.sort(key=lambda x: int(x.split('.')[0].split('Frame')[1]))
    for frame in tqdm(f_name):
        frame_path = os.path.join(args.driving_video, frame)
        img = Image.open(frame_path)
        driving_img = Variable(transform(img)).cuda()
        driving_img = driving_img.unsqueeze(0)

        # Get output
        with torch.no_grad():
            output = model.get_result_w_embedding(inferenced_embedding, driving_img)

        # Draw output
        output = output.data.cpu().numpy()
        output = output.transpose(0, 2, 3, 1)
        output = output[0]
        output = output * 255
        output = output.astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        generated_frames.append(output)
        # cv2.imshow('output', output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # Save output as images
    output_dir = os.path.join(args.results_folder, f'{inferenced_name}_w_embedded')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, frame in enumerate(generated_frames):
        cv2.imwrite(os.path.join(output_dir, f_name[i]), frame)

    # Save output as video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(
        output_dir, f'{inferenced_name}_w_embedded.mp4'), fourcc, 30, (args.dim, args.dim))
    for frame in generated_frames:
        video.write(frame)
    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UnwrappedFace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_views', type=int, default=2, help='Num views')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of image')
    parser.add_argument('--model_type', type=str, default='UnwrappedFaceSampler_from1view')
    parser.add_argument('--inner_nc', type=int, default=128)
    parser.add_argument('--pretrained_weight', type=str, default='')
    parser.add_argument('--embedding_path', type=str, default='./results/embedded_faces.pkl')
    parser.add_argument('--inferenced_name', type=str, default='Taylor_Swift')
    parser.add_argument('--driving_video', type=str, default='./driver_test/')
    parser.add_argument('--results_folder', type=str, default='./results/')
    parser.add_argument('--skip_net_backbone', type=str, choices=['unet_3+', 'unet_128', 'unet_256'], default='unet_3+')
    parser.add_argument('--no_skip_net_backbone', type=str, choices=['unet_128', 'unet_256'], default='unet_128')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
