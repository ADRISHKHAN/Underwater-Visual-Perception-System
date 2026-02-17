import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
from src.enhancement import UnderwaterEnhancer
from src.detector import TrashDetector
from src.calibration import CameraCalibrator
from src.fallback import FallbackDetector
from src.communication import MissionLogger, DetectionTransmitter
from src.ros2_bridge import DiyaROS2Bridge
import time
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Underwater Trash Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("🌊 Underwater Visual Perception System")
st.markdown("Real-time Trash Detection & Image Enhancement for AUVs/ROVs")

# Sidebar Configuration
st.sidebar.header("Configuration")

# 1. Source Selection
st.sidebar.subheader("Input Source")
source_option = st.sidebar.selectbox(
    "Select Input Source",
    ("Webcam", "Video File"),
    key="source_selectbox"
)

# 2. Enhancement Settings
st.sidebar.subheader("Image Enhancement")
enable_enhancement = st.sidebar.checkbox("Enable Enhancement (CLAHE + WB)", value=True, key="enhancement_checkbox")
clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.5, key="clip_limit_slider")

# 3. Detection Settings
st.sidebar.subheader("Object Detection")
enable_detection = st.sidebar.checkbox("Enable Detection", value=True, key="detection_checkbox")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, key="conf_slider")
model_options = [
    ("yolov8n.pt (Fastest)", "yolov8n.pt"),
    ("yolov8s.pt (Balanced)", "yolov8s.pt"),
    ("yolov8m.pt (Accurate)", "yolov8m.pt"),
    ("yolov8l.pt (Most Accurate)", "yolov8l.pt"),
]
available_models = []
for label, filename in model_options:
    if os.path.exists(os.path.join("models", filename)):
        available_models.append((label, filename))
if not available_models:
    available_models = [("yolov8n.pt (Fastest)", "yolov8n.pt")]

model_label = st.sidebar.selectbox(
    "Select Model (Accuracy vs Speed)",
    [label for label, _ in available_models],
    key="model_selectbox"
)
model_type = dict(available_models)[model_label]

# 4. Mission Controls
st.sidebar.subheader("Mission Controls")
enable_tracking = st.sidebar.checkbox("Enable Tracking (Item 8)", value=True, key="tracking_checkbox")
enable_calibration = st.sidebar.checkbox("Enable Calibration (Item 11)", value=False, key="calibration_checkbox")
enable_fallback = st.sidebar.checkbox("Enable Non-AI Fallback (Item 12)", value=False, key="fallback_checkbox")
enable_logging = st.sidebar.checkbox("Enable Mission Logging (Item 1)", value=True, key="logging_checkbox")
enable_ros2 = st.sidebar.checkbox("Enable ROS2 Bridge (Item 10)", value=False, key="ros2_checkbox")

# Initialize Modules (Cached)
# Note: We include model_path in the cache key so it reloads if changed
@st.cache_resource
def load_modules(model_name):
    enhancer = UnderwaterEnhancer()
    calibrator = CameraCalibrator()
    fallback = FallbackDetector()
    logger = MissionLogger()
    transmitter = DetectionTransmitter()
    ros_bridge = DiyaROS2Bridge()
    detector = TrashDetector(model_path=model_name)
    return enhancer, detector, calibrator, fallback, logger, transmitter, ros_bridge

enhancer, detector, calibrator, fallback, logger, transmitter, ros_bridge = load_modules(model_type)

# Main Logic
def process_frame(frame, enhancer, detector, calibrator, fallback, settings):
    # 0. Calibration (De-warping)
    if settings['enable_calibration']:
        frame = calibrator.undistort(frame)

    # 1. Enhancement
    if settings['enable_enhancement']:
        # Update clip limit if changed (re-initializing is expensive, so we modify the object)
        enhancer.clahe.setClipLimit(settings['clip_limit'])
        processed_frame = enhancer.enhance(frame)
    else:
        processed_frame = frame.copy()
    annotated_frame = processed_frame.copy()
    detections = []

    # 2. Detection & Tracking
    if settings['enable_detection']:
        # Use track() instead of detect() for persistent IDs
        results = detector.track(processed_frame, conf=settings['conf_threshold'])

        if hasattr(results, 'boxes') and results.boxes is not None:
            for box in results.boxes:
                # Basic Box Info
                cls_id = int(box.cls[0])
                name = results.names[cls_id]
                conf = float(box.conf[0])
                
                # Tracking ID
                track_id = int(box.id[0]) if box.id is not None else "N/A"
                
                # Distance Estimation
                distance = detector.estimate_distance(box, name)
                
                # Draw on frame
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"ID:{track_id} {name} {conf:.2f} | {distance:.1f}m"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detections.append({
                    'ID': track_id,
                    'Class': name, 
                    'Confidence': f"{conf:.2f}",
                    'Distance (m)': f"{distance:.1f}"
                })
    
    # 3. Fallback Detection (Colors)
    if settings['enable_fallback']:
        fb_detections, fb_mask = fallback.detect_trash(frame)
        for d in fb_detections:
            x, y, w, h = d['BBox']
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(annotated_frame, d['Class'], (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            detections.append({
                'ID': "FB",
                'Class': d['Class'],
                'Confidence': "1.00",
                'Distance (m)': "N/A"
            })

    return processed_frame, annotated_frame, detections

# Layout for Video
col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw Input")
    raw_placeholder = st.empty()
with col2:
    st.subheader("Processed Output")
    processed_placeholder = st.empty()

# Detections Sidebar/Bottom
st.sidebar.subheader("Live Detections")
stats_placeholder = st.sidebar.empty()
snapshot_placeholder = st.sidebar.empty()

# Run Loop
start_button = st.sidebar.button("Start Processing")
stop_button = st.sidebar.button("Stop")

if 'run' not in st.session_state:
    st.session_state['run'] = False

if start_button:
    st.session_state['run'] = True
if stop_button:
    st.session_state['run'] = False

if st.session_state['run']:
    cap = None
    if source_option == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        video_file = st.sidebar.file_uploader("Upload Underwater Video", type=['mp4', 'mov', 'avi'])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
    
    if cap and cap.isOpened():
        while st.session_state['run']:
            ret, frame = cap.read()
            if not ret:
                st.write("End of video stream.")
                break
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Build settings dict
            settings = {
                'enable_enhancement': enable_enhancement,
                'clip_limit': clip_limit,
                'enable_detection': enable_detection,
                'conf_threshold': conf_threshold,
                'enable_tracking': enable_tracking,
                'enable_calibration': enable_calibration,
                'enable_fallback': enable_fallback
            }

            # Process
            enhanced, result, detections = process_frame(
                frame_rgb, enhancer, detector, calibrator, fallback, settings
            )

            # Robotics & Logging Integrations
            if enable_logging and detections:
                logger.log_detections(detections)
            
            if enable_ros2 and detections:
                ros_bridge.publish_detections(detections)
                
            if detections:
                payload = transmitter.package_payload(detections)
                transmitter.transmit(payload)

            # Display
            raw_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            processed_placeholder.image(result, channels="RGB", use_container_width=True)

            # Update Stats
            if detections:
                df = pd.DataFrame(detections)
                stats_placeholder.table(df['Class'].value_counts())
            else:
                stats_placeholder.write("No objects detected.")

            # Snapshot Feature
            if snapshot_placeholder.button("📸 Take Snapshot", key=f"snap_{time.time()}"):
                snapshot_dir = os.getenv("DIYA_SNAPSHOT_DIR", tempfile.gettempdir())
                os.makedirs(snapshot_dir, exist_ok=True)
                snap_path = os.path.join(snapshot_dir, f"snapshot_{int(time.time())}.png")
                Image.fromarray(result).save(snap_path)
                st.sidebar.success(f"Saved to {snap_path}")

        cap.release()
    else:
        if source_option == "Video File":
            st.warning("Please upload a video file to start.")
        else:
            st.error("Could not access webcam.")
else:
    st.info("Click 'Start Processing' in the sidebar.")
