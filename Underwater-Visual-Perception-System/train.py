from ultralytics import YOLO

def train_model(data_config='data.yaml', epochs=100, img_size=640):
    """
    Train YOLOv8 model on custom data.
    """
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=data_config, epochs=epochs, imgsz=img_size, plots=True)
    
    print("Training Completed.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    # Example usage
    # Ensure data.yaml exists and points to valid data before running
    # train_model()
    pass
