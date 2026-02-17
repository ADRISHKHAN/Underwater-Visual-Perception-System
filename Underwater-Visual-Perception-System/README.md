# 🌊 Diya: Underwater Trash Detection System

Diya is a real-time visual perception system designed for AUVs (Autonomous Underwater Vehicles) and ROVs (Remotely Operated Vehicles). It combines advanced image enhancement with state-of-the-art object detection to identify marine debris in challenging underwater environments.

## 🚀 Features

- **Real-time Image Enhancement**: Uses CLAHE and White Balance to correct underwater color casts and improve contrast.
- **YOLOv8 Detection**: Optimized for small object detection in turbid water.
- **Streamlit Interface**: User-friendly dashboard for live video processing and monitoring.
- **Data Augmentation**: Synthetic underwater noise and color cast simulation for training robust models.

## 📂 Project Structure

- `app.py`: Main Streamlit application.
- `src/`: Core logic modules (enhancement, detection, augmentation).
- `models/`: Trained YOLOv8 model weights (.pt).
- `data/`: Dataset storage (images/train, images/val).
- `notebooks/`: Training and demo notebooks for Google Colab.

## 🛠️ Setup Instructions

### 1. Requirements

Ensure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

### 2. Running the App

```bash
streamlit run app.py
```

## 📄 License

This project is open-source and available under the MIT License.
