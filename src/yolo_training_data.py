from ultralytics import YOLO

model = YOLO('yolov9c.pt')
results = model.train(
    data='/content/training_data/data.yaml',
    epochs=100,
    imgsz=640,
    batch=7,
    save=True,
)
