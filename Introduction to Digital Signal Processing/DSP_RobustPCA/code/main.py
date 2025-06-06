import os
import numpy as np
import skvideo.io
from RPCA_ALM import IALM, ALM, RPCA_ALM
import argparse
from util import save_to_csv, write_video, resize
from task2 import get_result

def main(args):
    video = skvideo.io.vread(fname=args.test_video)
    # video = resize(video)
    N, H, W, C = np.shape(video)
    video = video.flatten().reshape(N, H * W * C)

    L, S = IALM(video, lmbda=0.01, maxIter=100)
    flatten_L = L.astype(np.uint8).flatten()
    save_to_csv(flatten_L, args.save_task1_csv)

    if args.save_video:
        write_video(np.reshape(L, (N, H, W, C)), args.save_video_clear)
        write_video(np.reshape(S, (N, H, W, C)), args.save_video_noise)

    test_data = np.load(args.test_data)

    L, S = ALM(test_data, lmbda=0.01, maxIter=100)
    results = get_result(test_data, L)
    save_to_csv(results, args.save_task2_csv)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_video', type=str, default='../dataset/Task1/test_noisy.mp4')
    parse.add_argument('--test_data', type=str, default='../dataset/Task2/test_data.npy')
    parse.add_argument('--save_task1_csv', type=str, default='result_task1.csv')
    parse.add_argument('--save_task2_csv', type=str, default='result_task2.csv')
    parse.add_argument('--save_video', action='store_true')
    parse.add_argument('--save_video_clear', type=str, default='Task1_clear.mp4')
    parse.add_argument('--save_video_noise', type=str, default='Task1_noise.mp4')
    args = parse.parse_args()
    main(args)
    