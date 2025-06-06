import numpy as np
import cv2


def weighting_function(shape='triangle', c=0.5, d=64, eps=1e-2):
    assert shape in ['uniform', 'triangle', 'gaussian', 'trapezoid']

    if shape == 'uniform':
        return np.ones(256)
    elif shape == 'triangle':
        w = np.minimum(np.arange(256), 256 - np.arange(256))
    elif shape == 'gaussian':
        arr = np.linspace(-1, 1, 256)
        return np.exp(-arr**2 / c)
    elif shape == 'trapezoid':
        arr = np.linspace(0, 1, d)
        w = np.concatenate((arr, np.ones(256 - 2 * d), np.flip(arr)))
    w = np.float32(w)
    w[0] = w[-1] = eps
    return w


class Debevec():
    def __init__(self, shutter_times, lamda=10):
        self.ln_shutter_times = np.log(np.array(shutter_times))
        self.lamda = lamda

    def construct_matrix(self, Z_flat, vec_w):
        A = np.zeros([self.PN + 255, 256 + self.N])
        b = np.zeros(self.PN + 255)

        A[np.arange(self.PN), Z_flat] = vec_w

        for img_id in range(self.P):
            A[img_id * self.N:(img_id * self.N) + self.N, 256:] = -np.identity(self.N) * \
                vec_w[img_id * self.N:(img_id * self.N) + self.N]
            b[img_id * self.N:(img_id * self.N) + self.N] = self.ln_shutter_times[img_id]

        constraint = np.floor(0.5 * (np.max(Z_flat) + np.min(Z_flat))).astype(int)
        for i in range(254):
            A[self.PN + i, i:i + 3] = np.array([1, -2, 1]) * np.abs(i - constraint) * self.lamda

        A[-1, constraint] = 1
        b[:self.PN] = vec_w * b[:self.PN]

        return A, b

    def run(self, Z):
        self.Z = np.array(Z)
        self.P = self.Z.shape[0]
        self.N = self.Z.shape[1]
        self.PN = self.P * self.N

        vec_w = weighting_function()[self.Z.flatten()]

        A, b = self.construct_matrix(self.Z.flatten(), vec_w)

        A_inv = np.linalg.pinv(A.astype(np.float64))
        vec_x = A_inv @ b
        ln_G = vec_x[:256]

        return ln_G


class Robertson():
    def __init__(self, shutter_times, weight_shape='triangle', iteration=3):
        self.shutter_times = shutter_times
        self.weight_shape = weight_shape
        self.iteration = iteration

    def run(self, images):
        N, H, W, C = images.shape

        g_init = np.arange(256) / 127
        g = np.repeat(g_init.reshape(1, 256), repeats=C, axis=0)
        E = np.zeros((C, H, W))
        weight = weighting_function(shape=self.weight_shape)

        images = images.transpose(3, 0, 1, 2)

        Wij = cv2.LUT(images, weight)
        numerator = Wij * self.shutter_times[:, None, None]
        denominator = np.sum(Wij * np.square(self.shutter_times[:, None, None]), axis=1)

        for c in range(C):
            for i in range(self.iteration):
                Gij = cv2.LUT(images[c], g[c])
                E[c] = np.sum(numerator[c] * Gij, axis=0) / denominator[c]

                for pixel_value in range(256):
                    g_sum = 0
                    for n in range(N):
                        idx = images[c, n] == pixel_value
                        g_sum += np.sum(E[c][idx] * self.shutter_times[n])
                    pixel_count = np.count_nonzero(images[c] == pixel_value)
                    g[c, pixel_value] = g_sum / pixel_count if pixel_count != 0 else 0

                g[c] /= g[c, 127]

        return g


class ToneMapping():

    def __init__(self):
        pass

    def photographic_global(self, hdr, d=1e-6, a=0.5):
        Lw, Lw_ave, Lm = hdr, np.exp(np.mean(np.log(d + hdr))), (a / np.exp(np.mean(np.log(d + hdr)))) * hdr
        ldr = np.clip(np.array((Lm * (1 + (Lm / (np.max(Lm) ** 2)))) / (1 + Lm) * 255), 0, 255)
        return ldr.astype(np.uint8)

    def gaussian_blurs(self, im, smax=25, a=0.5, fi=8.0, epsilon=0.01):
        cols, rows, num_s = im.shape[0], im.shape[1], int((smax + 1) / 2)
        blur_list, Vs_list = np.zeros(im.shape + (num_s,)), np.zeros(im.shape + (num_s,))
        for i, s in enumerate(range(1, smax + 1, 2)):
            blur, Vs = cv2.GaussianBlur(im, (s, s), 0), np.abs(
                (cv2.GaussianBlur(im, (s, s), 0) - im) / (2 ** fi * a / s ** 2 + im))
            blur_list[:, :, i], Vs_list[:, :, i] = blur, Vs
        smax = np.argmax(Vs_list > epsilon, axis=2)
        smax[np.where(smax == 0)] = num_s
        smax -= 1
        I, J = np.ogrid[:cols, :rows]
        return blur_list[I, J, smax]

    def photographic_local(self, hdr, d=1e-6, a=0.25):
        ldr, Lw_ave = np.zeros_like(hdr, dtype=np.float32), np.exp(np.mean(np.log(d + hdr)))
        for c in range(3):
            Lw, Lm, Ls = hdr[:, :, c], (a / Lw_ave) * hdr[:, :, c], self.gaussian_blurs((a / Lw_ave) * hdr[:, :, c])
            ldr[:, :, c] = np.clip(np.array((Lm / (1 + Ls)) * 255), 0, 255)
        return ldr.astype(np.uint8)

    def bilateral(self, hdr, a=0.5):
        r, g, b = 0.2126, 0.7152, 0.0722
        Lw = np.dot(hdr, np.array([b, g, r]) / (b + g + r))
        log_Lw = np.log10(Lw).astype(np.float32)
        log_base = cv2.bilateralFilter(log_Lw, 5, 15, 15)
        log_Ld = 2 * (log_base - log_base.max()) / (log_base.max() - log_base.min()) + (log_Lw - log_base)
        Ld = np.power(10, log_Ld)

        ldr = np.clip(((hdr * Ld[..., np.newaxis] / Lw[..., np.newaxis]) ** 0.3) * 255, 0, 255)

        return ldr.astype(np.uint8)
