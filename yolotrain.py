from ultralytics import YOLO

def train_yolov8(
    model_config="yolov8m.pt",   # Bisa juga langsung pakai .pt misal 'yolov8n.pt'
    data_config="trainstepv3\\data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,                     # 0 untuk GPU, 'cpu' untuk CPU
    patience=100
):
    # Load model config (bisa custom model atau pretrained)
    model = YOLO(model_config)

    # Train model
    model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device
    )

if __name__ == "__main__":
    train_yolov8()
