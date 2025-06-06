import os
import numpy as np
import skvideo.io
from RPCA_ALM import IALM, ALM, RPCA_ALM
import argparse
from util import save_to_csv, write_video, resize

def main(args):
    video = skvideo.io.vread(fname=args.test_video)
    N, H, W, C = np.shape(video)
    video = video.flatten().reshape(N, H * W * C)
    rpca_alm = RPCA_ALM(lmbda=args.lmbda, maxIter=args.maxIter)

    # L, S = IALM(video, lmbda=0.01, maxIter=1000)
    # flatten_L = L.astype(np.uint8).flatten()
    # save_to_csv(flatten_L, 'result_task1.csv')

    L, S = IALM(video, lmbda=args.lmbda, maxIter=args.maxIter)
    flatten_L = L.astype(np.uint8).flatten()
    save_to_csv(flatten_L, os.path.join(args.save_dir + f'_lambda_{args.lmbda}_maxIter_{args.maxIter}', 'IALM_1_' + args.save_csv))

    L, S = rpca_alm.fit(video, method='IALM')
    flatten_L = L.astype(np.uint8).flatten()
    save_to_csv(flatten_L, os.path.join(args.save_dir + f'_lambda_{args.lmbda}_maxIter_{args.maxIter}', 'IALM_2_' + args.save_csv))


    L, S = ALM(video, lmbda=args.lmbda, maxIter=args.maxIter)
    flatten_L = L.astype(np.uint8).flatten()
    save_to_csv(flatten_L, os.path.join(args.save_dir + f'_lambda_{args.lmbda}_maxIter_{args.maxIter}', 'ALM_1_' + args.save_csv))


    L, S = rpca_alm.fit(video, method='ALM')
    flatten_L = L.astype(np.uint8).flatten()
    save_to_csv(flatten_L, os.path.join(args.save_dir + f'_lambda_{args.lmbda}_maxIter_{args.maxIter}', 'ALM_2_' + args.save_csv))


    if args.save_video:
        write_video(np.reshape(L, (N, H, W, C)), os.path.join(args.save_dir, args.save_video_clear))
        write_video(np.reshape(S, (N, H, W, C)), os.path.join(args.save_dir, args.save_video_noise))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # parse.add_argument('--lmbda', type=float, default=0.01)
    # parse.add_argument('--maxIter', type=int, default=1000)
    parse.add_argument('--lmbda', type=float, default=0.001)
    parse.add_argument('--maxIter', type=int, default=1000)
    parse.add_argument('--test_video', type=str, default='../dataset/Task1/test_noisy.mp4')
    parse.add_argument('--save_dir', type=str, default='../Task1_result')
    parse.add_argument('--save_csv', type=str, default='result.csv')
    parse.add_argument('--save_video', action='store_true')
    parse.add_argument('--save_video_clear', type=str, default='Task1_clear.mp4')
    parse.add_argument('--save_video_noise', type=str, default='Task1_noise.mp4')
    args = parse.parse_args()
    main(args)
    