from ultralytics import YOLO

# Load a model

model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

#  the model
results = model.export(format='engine',imgsz=(736,1280),device="cuda:0", half = True, simplify = True, workspace = 8) # export the m
