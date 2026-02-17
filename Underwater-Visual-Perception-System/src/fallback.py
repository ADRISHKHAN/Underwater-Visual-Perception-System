import cv2
import numpy as np

class FallbackDetector:
    """
    Lightweight, non-AI detector using HSV color thresholding.
    Useful as a failsafe if the YOLO model fails.
    """
    def __init__(self):
        # Default: Look for bright/distinct colors (e.g., orange/red plastic)
        self.lower_bound = np.array([0, 120, 70])
        self.upper_bound = np.array([10, 255, 255])

    def detect_trash(self, frame):
        """
        Simple blob detection based on color thresholding.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500: # Minimum size threshold
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append({
                    'Class': 'Potential Debris (Fallback)',
                    'BBox': [x, y, w, h],
                    'Area': area
                })
                
        return detections, mask
