import cv2
import numpy as np

class CameraCalibrator:
    """
    Handles camera calibration and distortion correction.
    """
    def __init__(self, matrix=None, dist_coeffs=None):
        # Default placeholder matrix (Identity)
        self.camera_matrix = matrix if matrix is not None else np.eye(3)
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)

    def undistort(self, frame):
        """
        Apply lens undistortion to the frame.
        """
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        
        # Crop the image based on ROI if needed
        # x, y, w, h = roi
        # undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted

    @staticmethod
    def load_calibration(yaml_path):
        """
        Load calibration data from a file (e.g., from OpenCV calibration tool).
        """
        # Placeholder for loading logic
        return CameraCalibrator()
