
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker


def read_images(images_dir):
    images = []
    for filename in np.sort(os.listdir(images_dir)):
        if filename.split('.')[-1] == 'jpg':
            imgPath = os.path.join(images_dir, filename)
            img = cv2.imread(imgPath)
            images.append(img)
    return images


class ImagePlotter:
    def plot_response_curves(self, ln_G, output_dir):
        num_channels = len(ln_G)
        channel_colors = ['Blue', 'Green', 'Red']

        fig, axes = plt.subplots(1, num_channels, figsize=(16, 6))

        for channel, color in zip(range(num_channels), channel_colors):
            axes[channel].plot(ln_G[channel], np.arange(256), c=color)
            axes[channel].set_title(color)
            axes[channel].set_xlabel('ShutterTime (log scale)')
            axes[channel].set_ylabel('Reading')
            axes[channel].grid(linestyle=':', linewidth=1)

        fig.savefig(os.path.join(output_dir, 'response_curves.png'))

    def plot_radiance_map(self, radiance, output_dir):
        def exp_formatter(x, pos): return '%.3f' % np.exp(x)

        _, _, num_channels = radiance.shape
        ln_radiance = np.log(radiance)
        channel_colors = ['Blue', 'Green', 'Red']

        plt.clf()
        fig, axes = plt.subplots(1, num_channels, figsize=(16, 6))

        for channel, color in zip(range(num_channels), channel_colors):
            img = axes[channel].imshow(ln_radiance[:, :, channel], cmap='jet')
            axes[channel].set_title(color)
            axes[channel].set_axis_off()
            cax = make_axes_locatable(axes[channel]).append_axes("right", size="10%", pad=0.1)
            fig.colorbar(img, cax=cax, format=ticker.FuncFormatter(exp_formatter))

        fig.savefig(os.path.join(output_dir, 'radiance_map.png'))
