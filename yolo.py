# For model
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# LABELS = ["alaxan", "bactidol", "bioflu", "biogesic", "dayzinc", "decolgen", "fish oil", "kremil s", "medicol", "neozep"]

LABELS = ['neozep', 'biogesic', 'fish oil', 'medicol', 'bactidol', 'bioflu', 'kremil s', 'alaxan', 'decolgen', 'dayzinc']

def load_yolo_model(path):
    model = YOLO(path, task='detect')
    return model

def _predict(data):
    # Load model
    model = load_yolo_model("assets/meds_yolov8.torchscript")

    # Inference
    image = Image.open(data)
    image = np.asarray(image)
    results = model.predict(image, conf=0.5)

    return image, results

def get_predicted_image(data):
    image, results = _predict(data)

    for result in results:        
        pred = result.boxes.data[0]                           # extracts class label
        boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
        for box in boxes:                                          # iterate boxes
            r = box.xyxy[0].astype(int)                            # get corner points as int                                           # print boxes
            cv2.rectangle(image, r[:2], r[2:], (0, 255, 0), 2) # draw boxes on img
            cv2.putText(image, LABELS[int(pred[5])], (r[0] + 3, r[3] + 14), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

    return image
