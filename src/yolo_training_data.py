# from ultralytics import YOLO
# import wandb

# wandb.init(project="gbc_defect_detection_v2", job_type="model_training")

# model = YOLO('yolov9c.pt')
# results = model.train(
#     data='/home/ubuntu/yolo_model_training/yolo-object-detection-training-project/data/data.yaml',
#     epochs=100,
#     imgsz=640,
#     batch=8,
#     save=True,
# )

# wandb.finish()



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

    # --- Early stopping ---
    patience=15,          # stop if val fitness doesn't improve for 15 epochs
                          # (best.pt is still saved automatically throughout)

    # --- Regularization ---
    weight_decay=0.001,   # up from YOLO's default 0.0005 - fights overfitting
    label_smoothing=0.0,  # try 0.05-0.1 if the model stays overconfident

    # --- Augmentation (stronger, to force more robust features) ---
    degrees=10.0,         # random rotation (+/- deg)
    translate=0.1,        # random translation
    scale=0.5,             # random scale jitter
    shear=2.0,             # random shear
    perspective=0.0005,    # slight random perspective warp
    flipud=0.2,            # vertical flip probability
    fliplr=0.5,            # horizontal flip probability
    mosaic=1.0,             # keep mosaic on
    mixup=0.15,             # blend pairs of images/labels
    copy_paste=0.1,         # paste objects between images
    close_mosaic=10,        # disable mosaic for the last 10 epochs
                            # so the model fine-tunes on realistic images
)

wandb.finish()