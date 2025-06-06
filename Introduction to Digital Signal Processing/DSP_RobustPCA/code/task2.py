import os
import numpy as np
from RPCA_ALM import IALM, ALM, RPCA_ALM
import argparse
from util import save_to_csv

def get_poison_ids(test_data, L):
    residuals = test_data - L
    row_magnitudes = np.linalg.norm(residuals, axis=1)
    row_magnitudes_sort = np.sort(row_magnitudes)
    threshold = row_magnitudes_sort[-50]
    suspicious_indices = np.where(row_magnitudes > threshold)[0]
    return suspicious_indices

def get_result(test_data, L):
    results = np.zeros(test_data.shape[0], dtype=np.float64)
    suspicious_indices = get_poison_ids(test_data, L)
    results[suspicious_indices] = 1.0
    return results

def main(args):
    test_data = np.load(args.test_data)
    rpca_alm = RPCA_ALM(lmbda=args.lmbda, maxIter=args.maxIter)

    # L, S = ALM(test_data, lmbda=0.01, maxIter=100)
    # results = get_result(test_data, L)
    # save_to_csv(results, 'result_task2.csv')

    L, S = IALM(test_data, lmbda=args.lmbda, maxIter=args.maxIter)
    results = get_result(test_data, L)
    save_to_csv(results, os.path.join(args.save_dir + f'_lambda_{args.lmbda}_maxIter_{args.maxIter}', 'IALM_1_' + args.save_name))

    L, S = rpca_alm.fit(test_data, method='IALM')
    results = get_result(test_data, L)
    save_to_csv(results, os.path.join(args.save_dir + f'_lambda_{args.lmbda}_maxIter_{args.maxIter}', 'IALM_2_' + args.save_name))


    L, S = ALM(test_data, lmbda=args.lmbda, maxIter=args.maxIter)
    results = get_result(test_data, L)
    save_to_csv(results, os.path.join(args.save_dir + f'_lambda_{args.lmbda}_maxIter_{args.maxIter}', 'ALM_1_' + args.save_name))


    L, S = rpca_alm.fit(test_data, method='ALM')
    results = get_result(test_data, L)
    save_to_csv(results, os.path.join(args.save_dir + f'_lambda_{args.lmbda}_maxIter_{args.maxIter}', 'ALM_2_' + args.save_name))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # parse.add_argument('--lmbda', type=float, default=0.01)
    # parse.add_argument('--maxIter', type=int, default=100)
    parse.add_argument('--lmbda', type=float, default=0.1)
    parse.add_argument('--maxIter', type=int, default=200)
    parse.add_argument('--test_data', type=str, default='../dataset/Task2/test_data.npy')
    parse.add_argument('--save_dir', type=str, default='../Task2_result')
    parse.add_argument('--save_name', type=str, default='result.csv')
    args = parse.parse_args()
    main(args)
    