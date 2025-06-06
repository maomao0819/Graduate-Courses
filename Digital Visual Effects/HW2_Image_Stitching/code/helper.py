import cv2
import os
import numpy as np
import random
import glob
import re
from argparse import ArgumentParser
from annoy import AnnoyIndex


def get_focal_lengths(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()
                 and " " not in line and ".jpg" not in line and ".png" not in line]
    return [float(length) for length in lines]


def cylindrical_projection(img, f_len):
    h, w = img.shape[:2]
    y_o, x_o = h // 2, w // 2
    new_img = np.zeros(img.shape, dtype=int)
    y, x = np.mgrid[0:h, 0:w]
    x_c, y_c = x - x_o, y - y_o
    x_n = np.round(f_len * np.arctan(x_c / f_len)) + x_o
    y_n = np.round(f_len * y_c / np.hypot(x_c, f_len)) + y_o
    x_n, y_n = x_n.astype(int), y_n.astype(int)

    new_img[y_n, x_n] = img[y, x]
    return new_img.astype(np.uint8)


def match_keypoints(des1, des2, norm='L1'):
    # des1: (N, 128), des2: (M, 128)

    def build_annoy_index(features, dimensions, num_trees=10):
        if norm == 'L1':
            index = AnnoyIndex(dimensions, 'manhattan')
        else:
            index = AnnoyIndex(dimensions, 'euclidean')
        for i, feature in enumerate(features):
            index.add_item(i, feature)

        index.build(num_trees)
        return index

    def find_nearest_neighbors(annoy_index, query_feature, num_neighbors=1):
        # Perform feature matching
        return annoy_index.get_nns_by_vector(query_feature, num_neighbors)

    dimensions = 128

    annoy_index = build_annoy_index(des2, dimensions)
    matches = []

    for i in range(des1.shape[0]):
        nearest_neighbors = find_nearest_neighbors(annoy_index, des1[i])
        matches.append([i, nearest_neighbors[0]])

    return matches


def draw_custom_matches(img1, img2, point_pairs, thickness=1, color=(0, 255, 0), type=0):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create an empty image to accommodate both images
    result = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')

    # Place both images side by side
    result[:h1, :w1] = img1
    result[:h2, w1:] = img2

    # Get corresponding points
    if len(point_pairs) > 50:
        point_pairs = random.sample(point_pairs, 50)
    # Draw lines between corresponding points
    if type == 0:
        for (x1, y1), (x2, y2) in point_pairs:
            cv2.line(result, (int(x1), int(y1)), (int(x2 + w1), int(y2)), color, thickness)
    # Draw corresponding points
    else:
        for (x1, y1), (x2, y2) in point_pairs:
            cv2.circle(result, (int(x1), int(y1)), 5, color, -1)
            cv2.circle(result, (int(x2 + w1), int(y2)), 5, color, -1)
            
    return result
