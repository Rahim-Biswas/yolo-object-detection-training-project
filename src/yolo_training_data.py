from ultralytics import YOLO
import wandb

wandb.init(project="gbc_defect_detection_v2", job_type="model_training")

model = YOLO('yolov9c.pt')
results = model.train(
    data='/home/ubuntu/yolo_model_training/yolo-object-detection-training-project/data/data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    save=True,
)

wandb.finish()