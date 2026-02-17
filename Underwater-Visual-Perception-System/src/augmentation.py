import cv2
import numpy as np
import random

def add_underwater_cast(image, blue_boost=30, green_boost=10, red_drop=30):
    """
    Simulate underwater color cast by boosting blue/green and reducing red.
    """
    b, g, r = cv2.split(image)
    b = cv2.add(b, blue_boost)
    g = cv2.add(g, green_boost)
    r = cv2.subtract(r, red_drop)
    return cv2.merge((b, g, r))

def add_turbidity(image, ksize=(5, 5)):
    """
    Simulate turbidity using Gaussian blur.
    """
    return cv2.GaussianBlur(image, ksize, 0)

def add_noise(image, noise_type="gaussian"):
    """
    Add random noise to simulate scattering/particulates.
    """
    if noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss * 20
        return np.clip(noisy, 0, 255).astype(np.uint8)
    return image

def augment_image(image):
    """
    Apply a random combination of underwater effects.
    """
    aug = image.copy()
    if random.random() > 0.3:
        aug = add_underwater_cast(aug)
    if random.random() > 0.3:
        aug = add_turbidity(aug)
    if random.random() > 0.5:
        aug = add_noise(aug)
    return aug

if __name__ == "__main__":
    # Test block
    print("Augmentation module loaded.")
