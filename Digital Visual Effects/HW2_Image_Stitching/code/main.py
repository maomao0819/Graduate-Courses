import cv2
import os
import numpy as np
import random
import glob
import re
from argparse import ArgumentParser
from annoy import AnnoyIndex
import math

from helper import *
from sift_helper import sift_custom


def optimal_shift(match_results, threshold=3):
    shift_candidates = [(a[0] - b[0], a[1] - b[1]) for a, b in match_results]
    votes = []
    for candidate in shift_candidates:
        adjusted = [(s[0] - candidate[0], s[1] - candidate[1])
                    for s in shift_candidates]
        squared_dists = [x * x + y * y for x, y in adjusted]
        votes.append(np.count_nonzero(np.array(squared_dists) < threshold))

    return shift_candidates[np.argmax(votes)]


def remove_black_columns(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_col_sum = np.sum(img_gray, axis=0)
    all_black_col = np.where(img_col_sum == 0)
    return np.delete(img, all_black_col, axis=1)


def warp_and_blend(src, dst, H, height, width, direction='b', blend_strength=0.5):
    if direction == 'f':
        H = np.linalg.inv(H)

    ymax, ymin, xmax, xmin = height, 0, width, 0

    warped = cv2.warpPerspective(
        src, H, (width, height), flags=cv2.INTER_NEAREST)

    dst_roi = dst[ymin:ymax, xmin:xmax]

    assert dst[ymin:ymax, xmin:xmax].shape == dst.shape

    non_intersecting_mask = np.where(dst[ymin:ymax, xmin:xmax] == 0, 1, 0)

    blended = cv2.addWeighted(warped, blend_strength,
                              dst_roi, 1 - blend_strength, 0)
    # dst_combined = np.where((mask_warped > 0) & (non_intersecting_mask == 0), blended, dst_roi)
    dst_combined = np.where(warped > 0, blended, dst_roi)

    # Replace the roi destination with the combined image
    dst[ymin:ymax, xmin:xmax] = dst_combined

    # Combine the brightness-modified non-intersecting areas with the combined image
    dst[ymin:ymax, xmin:xmax] = np.where(
        non_intersecting_mask > 0, warped, dst_combined)

    return dst


# def sift_descriptor_fast(img):
#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(img, None)
#     return np.array([x.pt for x in kp1]), des1


def myDetectAndCompute(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # magnitude, I_x, I_y = compute_gradient(gray_img)
    # points = harris_corner(I_x, I_y)
    # return sift_descriptor(points, I_x, I_y)
    # points = harris_corner(img)
    # return sift_descriptor_harris(gray_img, points)
    return sift_custom(gray_img)
    # return sift_descriptor_fast(gray_img)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--images_dir', default='data/scene-our', type=str)
    parser.add_argument('--end2end_alignment', default=False, type=bool)

    args = parser.parse_args()

    img_paths = glob.glob(os.path.join(args.images_dir, "*.jpg"))
    img_paths = [x for x in img_paths if 'pano' not in x]
    img_paths = sorted(img_paths)
    imgs = [cv2.imread(path) for path in img_paths]
    if max(imgs[0].shape) > 1000:
        ratio = 400 / imgs[0].shape[0]
        imgs = [cv2.resize(img, dsize=(int(imgs[0].shape[1]*ratio), 400))
                for img in imgs]

    # Warp to cylindrical
    imgs = [cylindrical_projection(img, focal) for img, focal in zip(
        imgs, get_focal_lengths(os.path.join(args.images_dir, "pano.txt")))]

    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]

    # Compute pairwise alignments
    shift_x, shift_y = [], []
    M = np.eye(3)
    kp2 = None
    for idx in range(len(imgs) - 1):
        img1 = imgs[idx]
        img2 = imgs[idx + 1]

        kp1, des1 = (kp2, des2) if kp2 is not None else myDetectAndCompute(
            img1.copy())
        # kp1, des1 = myDetectAndCompute(img1.copy())
        kp2, des2 = myDetectAndCompute(img2.copy())
        print(f'Found {kp1.shape[0]} & {kp2.shape[0]} keypoints')

        matches = match_keypoints(des1, des2, norm='L1')
        print(f"Found {len(matches)} matches")

        points1 = np.array([kp1[m[0]] for m in matches])
        points2 = np.array([kp2[m[1]] for m in matches])

        point_pairs = list(zip(list(points1), list(points2)))
        result = draw_custom_matches(img1, img2, point_pairs)
        cv2.imwrite('data/matches.png', result)

        # keypoint matching + RANSAC
        dx, dy = optimal_shift(point_pairs)
        shift_x.append(dx)
        shift_y.append(dy)
        # H = find_homography(points1, points2, img1, img2)
        print(f'Finished {idx}th image')
        print('--------------------------')

        if args.end2end_alignment:
            continue

        M = M @ np.array([[1, 0, shift_x[idx]],
                         [0, 1, shift_y[idx]], [0, 0, 1]])
        dst = warp_and_blend(img2, dst, M, h_max, w_max, direction='b')

    if args.end2end_alignment:
        for idx in range(len(imgs) - 1):
            img2 = imgs[idx + 1]

            avg_dy = sum(shift_y) / len(shift_y)
            shift_y = [dy-avg_dy for dy in shift_y]
            M = M @ np.array([[1, 0, shift_x[idx]],
                             [0, 1, shift_y[idx]], [0, 0, 1]])
            dst = warp_and_blend(img2, dst, M, h_max, w_max, direction='b')
    # Refine Image with removing columns which are all black.
    dst = remove_black_columns(dst)
    cv2.imwrite("data/panorama.png", dst)

# Remark: Some code was completed with the help of ChatGPT.
