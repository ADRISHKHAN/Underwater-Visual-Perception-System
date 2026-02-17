import os
from ultralytics import YOLO

class TrashDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLOv8 detector.
        :param model_path: Path to the YOLOv8 model file.
        """
        # Resolve model path robustly for local/dev/hosted environments.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        candidates = []

        if os.path.isabs(model_path):
            candidates.append(model_path)
        else:
            candidates.append(os.path.abspath(model_path))
            candidates.append(os.path.abspath(os.path.join(project_root, model_path)))
            candidates.append(os.path.abspath(os.path.join(project_root, 'models', model_path)))

        model_path = next((p for p in candidates if os.path.exists(p)), None) or os.path.abspath(
            os.path.join(project_root, 'models', os.path.basename(model_path))
        )
        self.model = YOLO(model_path)
        # Calibration constants for distance estimation (Heuristic)
        # Assumes a 640x480 resolution and standard focal length
        self.reference_widths = {
            'plastic_bag': 0.3,  # meters
            'bottle': 0.25,
            'can': 0.15,
            'rope': 0.5,
            'fish_net': 1.0
        }
        self.focal_length = 500  # Pixels (Heuristic for common webcams)

    def detect(self, image, conf=0.25, iou=0.45):
        """
        Run inference on the image with enhanced small object detection.
        :param image: Input image (numpy array).
        :param conf: Confidence threshold.
        :param iou: IoU threshold for NMS.
        :return: Results object from YOLOv8.
        """
        # Run inference with enhanced parameters for small objects
        results = self.model(
            image, 
            conf=conf, 
            iou=iou,
            verbose=False,
            # Enhanced parameters for small object detection
            max_det=1000,  # Allow more detections
            agnostic_nms=False
        )
        return results[0]  # Return the first result

    def track(self, image, conf=0.25, iou=0.45, tracker="bytetrack.yaml"):
        """
        Run inference with persistent tracking.
        """
        results = self.model.track(
            image,
            conf=conf,
            iou=iou,
            persist=True,
            tracker=tracker,
            verbose=False,
            max_det=1000
        )
        return results[0]

    def estimate_distance(self, box, class_name):
        """
        Estimate distance to object using Triangle Similarity.
        Distance (D) = (Actual Width * Focal Length) / Pixel Width
        """
        actual_width = self.reference_widths.get(class_name, 0.3)
        pixel_width = box.xywh[0][2] # width is the 3rd element in xywh
        
        if pixel_width == 0:
            return 0.0
            
        distance = (actual_width * self.focal_length) / pixel_width
        return float(distance)
