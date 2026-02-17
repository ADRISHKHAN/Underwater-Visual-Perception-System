import cv2
import numpy as np
import sys
import os

# Add relevant paths
sys.path.append(os.getcwd())

from src.enhancement import UnderwaterEnhancer
from src.detector import TrashDetector
from src.augmentation import augment_image

def test_pipeline():
    print("Starting verification...")
    
    # 1. Create dummy image (640x480)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some color to test channel ops
    dummy_img[:] = (100, 50, 50) 
    print("Dummy image created.")

    # 2. Test Augmentation
    try:
        aug_img = augment_image(dummy_img)
        assert aug_img is not None
        assert aug_img.shape == dummy_img.shape
        print("[PASS] Augmentation module.")
    except Exception as e:
        print(f"[FAIL] Augmentation module: {e}")

    # 3. Test Enhancement
    try:
        enhancer = UnderwaterEnhancer()
        enhanced_img = enhancer.enhance(dummy_img)
        assert enhanced_img is not None
        assert enhanced_img.shape == dummy_img.shape
        print("[PASS] Enhancement module.")
    except Exception as e:
        print(f"[FAIL] Enhancement module: {e}")

    # 4. Test Detector
    # Note: This requires yolov8n.pt to be downloaded. Ultralytics usually handles this.
    try:
        detector = TrashDetector(model_path='models/yolov8n.pt') # wrapper handles loading
        # We need to mock the result or just check it runs
        result = detector.detect(dummy_img)
        assert result is not None
        print("[PASS] Detection module.")
    except Exception as e:
        print(f"[FAIL] Detection module: {e}")
        print("Note: If network is unavailable, model download might have failed.")

    print("\nVerification Complete.")

if __name__ == "__main__":
    test_pipeline()
