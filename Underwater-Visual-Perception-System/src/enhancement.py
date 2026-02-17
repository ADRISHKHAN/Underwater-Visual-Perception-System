import cv2
import numpy as np

class UnderwaterEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Initialize the enhancers.
        :param clip_limit: Threshold for contrast limiting in CLAHE.
        :param tile_grid_size: Size of grid for histogram equalization.
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply_clahe(self, image):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the L channel of LAB image.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def white_balance(self, image):
        """
        Apply Gray World White Balance algorithm.
        """
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def enhance(self, image):
        """
        Pipeline: White Balance -> CLAHE
        """
        if image is None:
            return None
            
        wb = self.white_balance(image)
        enhanced = self.apply_clahe(wb)
        return enhanced
