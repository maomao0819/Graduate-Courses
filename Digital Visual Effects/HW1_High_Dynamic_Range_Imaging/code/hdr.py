
import cv2
import os
import numpy as np
import random
from argparse import ArgumentParser

from alignment import ImageAlignment
from methods import Debevec, Robertson, ToneMapping, weighting_function
from utils import *


class HDR():
    def __init__(self, args):
        self.args = args
        self.shutter_times = np.array([1/320, 1/200, 1/125, 1/80, 1/50, 1/30, 1/20, 1/13, 1/8, 1/5, 1/3])
        self.ln_shutter_times = np.log(self.shutter_times)

        self.alignment = ImageAlignment(self.args.depth, self.args.threshold)
        self.debevec = Debevec(self.shutter_times, lamda=10)
        self.robertson = Robertson(self.shutter_times)
        self.tone_mapping = ToneMapping()
        self.plotter = ImagePlotter()

    def sample_points(self, images, method, n_samples=256):
        H, W, C = images[0].shape

        if method == 'robertson':
            for i in range(len(images)):
                h, w, _ = images[i].shape
                images[i] = cv2.resize(images[i], (int(w/4), int(h/4)), interpolation=cv2.INTER_AREA)
                images[i] = np.array(images[i])
            images = np.stack(images, axis=0)
            return np.transpose(np.array([[images[p][:, :, c] for p in range(len(images))] for c in range(C)]), (1, 2, 3, 0))

        random.seed(0)
        indices = np.array(random.sample(range(H * W), n_samples))

        return np.array([[images[p][indices // W, indices % W, c] for p in range(len(images))] for c in range(C)])

    def mask_high_variance(self, images):

        def compute_variance(images):
            gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
            mean = np.mean(gray_images, axis=0)
            variance = np.mean((gray_images - mean) ** 2, axis=0) / (mean+1e-9) / (mean+1e-9)
            return variance

        def find_high_variance_regions(variance, threshold):
            mask = variance > threshold
            return mask

        def mask_out_high_variance_regions(images, mask):
            masked_images = []
            for img in images:
                img_copy = img.copy()
                img_copy[mask] = 0
                masked_images.append(img_copy)
            return masked_images

        variances = compute_variance(images)

        mask_threshold = 3
        high_variance_masks = find_high_variance_regions(variances, mask_threshold)
        cv2.imwrite(os.path.join(self.args.output_dir, 'mask.jpg'), (high_variance_masks).astype(np.uint8)*255)

        masked_images = mask_out_high_variance_regions(images, high_variance_masks)
        # cv2.imwrite('temp.jpg', (masked_images[-2]).astype(np.uint8))
        return masked_images

    def hdr(self, images):
        samples = self.sample_points(images, method=self.args.method)  # (channels , n_frames, n_points)
        if self.args.method == 'debevec':
            return [self.debevec.run(samples[c, :, :]) for c in range(3)]
        return self.robertson.run(samples)

    def run(self):
        images = read_images(images_dir=self.args.images_dir)
        if self.args.alignment:
            for i in range(1, len(images)):
                print(f'Aligning {i}th image')
                images[i] = self.alignment.align(images[i], images[i-1])

        if self.args.mask_high_variance_region:
            ln_G = self.hdr(self.mask_high_variance(images))
        else:
            ln_G = self.hdr(images)

        radiances = self.recovered_radiance(images, ln_G)
        self.run_tone_mapping(radiances)

        self.plotter.plot_response_curves(ln_G, self.args.output_dir)
        self.plotter.plot_radiance_map(radiances, self.args.output_dir)

    def run_tone_mapping(self, radiances, alpha=0.5):
        for t, f in [('global', self.tone_mapping.photographic_global), ('local', self.tone_mapping.photographic_local),
                     ('bilateral', self.tone_mapping.bilateral)]:
            ldr, filepath = f(radiances, a=alpha), os.path.join(self.args.output_dir, f"{t}.png")
            cv2.imwrite(filepath, ldr)
            if t == 'local':
                cv2.imwrite('result.png', ldr)

    def recovered_radiance(self, images, ln_G):
        H, W, C = images[0].shape
        ln_radiance = np.zeros_like(images[0]).astype(np.float32)

        for c in range(C):
            W_sum = np.zeros([H, W], dtype=np.float32) + 1e-6
            ln_radiance_sum = np.zeros([H, W], dtype=np.float32)

            for p in range(len(images)):
                image_1D = images[p][:, :, c].flatten()
                weights = weighting_function()[image_1D].reshape(H, W)
                ln_radiance_weighted = (ln_G[c][image_1D] - self.ln_shutter_times[p]).reshape(H, W) * weights
                ln_radiance_sum += ln_radiance_weighted
                W_sum += weights

            ln_radiance[:, :, c] = ln_radiance_sum / W_sum

        radiance_map = np.exp(ln_radiance).astype(np.float32)
        return radiance_map


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--method', default='debevec', choices=['debevec', 'robertson'], type=str)
    parser.add_argument('--images_dir', default='./data', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)

    parser.add_argument('--alignment', default=False, type=bool)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--threshold', default=10, type=int)

    parser.add_argument('--mask_high_variance_region', default=False, type=bool)

    hdr = HDR(parser.parse_args())
    hdr.run()

# Remark: Some code was completed with the help of ChatGPT.
