import numpy as np
import cv2
from scipy.ndimage import filters, interpolation
import math


def generate_gaussian_pyramid(image, sigma, num_intervals, num_octaves):
    gaussian_pyramid = []
    k = 2 ** (1.0 / num_intervals)
    for octave in range(num_octaves):
        octave_images = []
        for interval in range(num_intervals + 3):
            if octave == 0 and interval == 0:
                next_image = image.copy()
            elif interval == 0:
                next_image = cv2.resize(gaussian_pyramid[octave - 1][-3], None,
                                        fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            else:
                next_image = cv2.GaussianBlur(next_image, (0, 0), sigmaX=sigma * (k ** (interval - 1)))
            octave_images.append(next_image)
        gaussian_pyramid.append(octave_images)
    return gaussian_pyramid


def generate_DoG_pyramid(gaussian_pyramid):
    dog_pyramid = []
    for octave_images in gaussian_pyramid:
        octave_dog = []
        for i in range(1, len(octave_images)):
            octave_dog.append(octave_images[i] - octave_images[i - 1])
        dog_pyramid.append(octave_dog)
    return dog_pyramid


def find_local_extrema(dog_pyramid, num_intervals, threshold):
    extrema_points = []
    for octave_idx, octave_dog in enumerate(dog_pyramid):
        for interval_idx in range(1, num_intervals + 1):
            curr_dog = octave_dog[interval_idx]
            for y in range(1, curr_dog.shape[0] - 1):
                for x in range(1, curr_dog.shape[1] - 1):
                    value = curr_dog[y, x]
                    if abs(value) > threshold:
                        neighbors = [
                            octave_dog[interval_idx - 1][y - 1:y + 2, x - 1:x + 2],
                            octave_dog[interval_idx][y - 1:y + 2, x - 1:x + 2],
                            octave_dog[interval_idx + 1][y - 1:y + 2, x - 1:x + 2]
                        ]
                        if (value > 0 and value == np.max(neighbors)) or (value < 0 and value == np.min(neighbors)):
                            extrema_points.append((octave_idx, interval_idx, y, x))
    return extrema_points


def scale_space_extrema(image, sigma, num_intervals, num_octaves, threshold):
    # Step 1: Generate Gaussian Pyramid
    gaussian_pyramid = generate_gaussian_pyramid(image, sigma, num_intervals, num_octaves)

    # Step 2: Generate DoG Pyramid
    dog_pyramid = generate_DoG_pyramid(gaussian_pyramid)

    # Step 3: Find local extrema
    extrema_points = find_local_extrema(dog_pyramid, num_intervals, threshold)

    return extrema_points


def localize_keypoints(extrema_points, image, sigma, num_intervals, num_octaves):
    def hessian3D_at(point, img, r):
        x, y = point
        dxx = img[y, x + r] + img[y, x - r] - 2 * img[y, x]
        dyy = img[y + r, x] + img[y - r, x] - 2 * img[y, x]
        dxy = (img[y + r, x + r] + img[y - r, x - r] - img[y - r, x + r] - img[y + r, x - r]) / 4
        return np.array([[dxx, dxy], [dxy, dyy]])

    def gradient_at(point, img, r):
        x, y = point
        dx = (img[y, x + r] - img[y, x - r]) / 2
        dy = (img[y + r, x] - img[y - r, x]) / 2
        return np.array([dx, dy])

    keypoints = []
    edge_threshold = 10.0

    gaussian_pyramid = generate_gaussian_pyramid(image, sigma, num_intervals, num_octaves)
    for octave_idx, interval_idx, y, x in extrema_points:
        img = gaussian_pyramid[octave_idx][interval_idx]
        point = np.array([x, y])

        # Refine the keypoint location
        localization_attempts = 5
        r = 1
        for _ in range(localization_attempts):
            gradient = gradient_at(point, img, r)
            hessian = hessian3D_at(point, img, r)
            offset = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if np.all(np.abs(offset) < 0.5):
                break
            point += np.round(offset).astype(int)

            if not (r < point[0] < img.shape[1] - r and r < point[1] < img.shape[0] - r):
                break
        else:
            continue

        # Compute the principal curvature ratio
        hessian_det = np.linalg.det(hessian)
        hessian_trace = np.trace(hessian)
        curvature_ratio = (hessian_trace ** 2) / hessian_det

        # Check if the point is edge-like
        if hessian_det > 0 and curvature_ratio < ((edge_threshold + 1) ** 2) / edge_threshold:
            keypoint = {
                'x': point[0] * (2 ** octave_idx),
                'y': point[1] * (2 ** octave_idx),
                'octave': octave_idx,
                'interval': interval_idx,
                'size': 2 * (2 ** octave_idx) * sigma * (2 ** (interval_idx / num_intervals)),
            }
            keypoints.append(keypoint)

    return keypoints


def orientation_assignment(keypoints, image, sigma, num_intervals, num_octaves):
    def compute_magnitude_and_orientation(img, sigma):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 360
        return magnitude, orientation

    def create_histogram(magnitude, orientation, x, y, radius, num_bins):
        hist = np.zeros(num_bins, dtype=np.float32)
        max_radius = radius ** 2
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i ** 2 + j ** 2 <= max_radius:
                    bin_idx = int(orientation[y + i, x + j] / 360 * num_bins)
                    hist[bin_idx] += magnitude[y + i, x + j]
        return hist

    def smooth_histogram(hist, num_bins):
        for i in range(num_bins):
            hist[i] = (hist[i - 1] + hist[i] + hist[(i + 1) % num_bins]) / 3
        return hist

    def find_peaks(hist, peak_ratio):
        max_value = np.max(hist)
        peak_threshold = peak_ratio * max_value
        peaks = np.where(hist > peak_threshold)[0]
        return peaks

    num_bins = 8
    peak_ratio = 0.2
    radius_factor = 3

    gaussian_pyramid = generate_gaussian_pyramid(image, sigma, num_intervals, num_octaves)
    keypoints_with_orientations = []

    for keypoint in keypoints:
        octave = keypoint['octave']
        interval = keypoint['interval']
        x, y = int(keypoint['x'] / (2 ** octave)), int(keypoint['y'] / (2 ** octave))

        img = gaussian_pyramid[octave][interval]
        magnitude, orientation = compute_magnitude_and_orientation(img, sigma)
        radius = int(radius_factor * keypoint['size'] / (2 ** octave))

        if y - radius >= 0 and y + radius < img.shape[0] and x - radius >= 0 and x + radius < img.shape[1]:
            hist = create_histogram(magnitude, orientation, x, y, radius, num_bins)
            hist = smooth_histogram(hist, num_bins)
            peaks = find_peaks(hist, peak_ratio)

            for peak in peaks:
                angle = (peak * 360 / num_bins) % 360
                new_keypoint = keypoint.copy()
                new_keypoint['orientation'] = angle
                keypoints_with_orientations.append(new_keypoint)

    return keypoints_with_orientations


def descriptor_generation(keypoints, image, sigma):
    def gradient_magnitude_and_orientation(image):
        dx = filters.sobel(image, axis=1)
        dy = filters.sobel(image, axis=0)
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        orientation = np.arctan2(dy, dx) * 180 / np.pi % 360
        return magnitude, orientation

    def create_descriptor(magnitude, orientation, x, y, angle, num_histograms, num_bins):
        descriptor = np.zeros((num_histograms, num_histograms, num_bins))
        cell_size = 4
        total_size = num_histograms * cell_size
        scale = 1.5 * sigma
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))

        for i in range(-total_size // 2, total_size // 2):
            for j in range(-total_size // 2, total_size // 2):
                curr_y = y + i
                curr_x = x + j
                if 0 <= curr_y < magnitude.shape[0] and 0 <= curr_x < magnitude.shape[1]:
                    rotated_i = i * cos_angle + j * sin_angle
                    rotated_j = -i * sin_angle + j * cos_angle
                    row_idx = int((rotated_i + total_size / 2) // cell_size)
                    col_idx = int((rotated_j + total_size / 2) // cell_size)
                    if 0 <= row_idx < num_histograms and 0 <= col_idx < num_histograms:
                        bin_idx = int(orientation[curr_y, curr_x] // (360 / num_bins)) % num_bins
                        weight = magnitude[curr_y, curr_x] * \
                            np.exp(-0.5 * (rotated_i ** 2 + rotated_j ** 2) / (scale ** 2))
                        descriptor[row_idx, col_idx, bin_idx] += weight
        return descriptor.flatten()

    num_histograms = 4
    num_bins = 8
    magnitude, orientation = gradient_magnitude_and_orientation(image)

    descriptors = []
    for keypoint in keypoints:
        x, y = int(keypoint['x']), int(keypoint['y'])
        angle = keypoint['orientation']
        descriptor = create_descriptor(magnitude, orientation, x, y, angle, num_histograms, num_bins)
        descriptor /= max(1e-7, np.linalg.norm(descriptor))
        descriptor[descriptor > 0.2] = 0.2
        descriptor /= max(1e-7, np.linalg.norm(descriptor))
        descriptors.append(descriptor)

    return np.array(descriptors, dtype=np.float32)


def sift_custom(image, sigma=1.6, num_intervals=1, num_octaves=8, contrast_threshold=0.01):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Normalize image to 0-1 range
    gray_image = gray_image.astype(np.float32) / 255.0

    # Step 1: Scale-space extrema detection
    extrema_points = scale_space_extrema(gray_image, sigma, num_intervals, num_octaves, contrast_threshold)

    # Step 2: Keypoint localization
    keypoints = localize_keypoints(extrema_points, gray_image, sigma, num_intervals, num_octaves)

    # Step 3: Orientation assignment
    keypoints = orientation_assignment(keypoints, gray_image, sigma, num_intervals, num_octaves)

    # Step 4: Descriptor generation
    descriptors = descriptor_generation(keypoints, gray_image, sigma)
    keypoints = np.array([(pt['x'], pt['y']) for pt in keypoints])

    return keypoints, descriptors
