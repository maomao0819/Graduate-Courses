
import numpy as np
import cv2


class ImageAlignment:
    def __init__(self, depth=4, threshold=10):
        self.depth = depth
        self.threshold = threshold

    def compute_error(self, source, target, threshold):
        src_median = np.median(source)
        tgt_median = np.median(target)
        _, src_binary = cv2.threshold(source, src_median, 255, cv2.THRESH_BINARY)
        _, tgt_binary = cv2.threshold(target, tgt_median, 255, cv2.THRESH_BINARY)
        mask = cv2.inRange(source, src_median - threshold, src_median + threshold)
        return cv2.countNonZero(cv2.bitwise_and(cv2.bitwise_xor(src_binary, tgt_binary), cv2.bitwise_not(mask)))

    def find_translation(self, source, target, x, y, threshold):
        h, w = target.shape
        translations = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]

        def evaluate_translation(dx, dy):
            tgt_shifted = cv2.warpAffine(target, M=np.float32([[1, 0, x + dx], [0, 1, y + dy]]), dsize=(w, h))
            error = self.compute_error(source, tgt_shifted, threshold)
            return error, dx, dy

        min_error, best_dx, best_dy = min((evaluate_translation(dx, dy) for dx, dy in translations), key=lambda x: x[0])
        return x + best_dx, y + best_dy

    # Median Threshold Bitmap (MTB) algorithm
    def recursive_alignment(self, source, target, pyramid_layer, threshold):
        if pyramid_layer == 0:
            return self.find_translation(source, target, 0, 0, threshold)

        h, w = target.shape
        src_half = cv2.resize(source, (w // 2, h // 2))
        tgt_half = cv2.resize(target, (w // 2, h // 2))
        dx, dy = self.recursive_alignment(src_half, tgt_half, pyramid_layer - 1, threshold)

        return self.find_translation(source, target, dx * 2, dy * 2, threshold)

    def align(self, src_original, tgt_original):
        src_gray = cv2.cvtColor(src_original, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(tgt_original, cv2.COLOR_BGR2GRAY)
        h, w = tgt_gray.shape
        dx, dy = self.recursive_alignment(src_gray, tgt_gray, self.depth, self.threshold)
        translation_matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
        return cv2.warpAffine(src_original, translation_matrix, (w, h))
