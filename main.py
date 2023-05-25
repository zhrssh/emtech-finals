# For deployment
import streamlit as st

# For model
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

LABELS = ['neozep', 'biogesic', 'fish oil', 'medicol', 'bactidol', 'bioflu', 'kremil s', 'alaxan', 'decolgen', 'dayzinc']

def load_yolo_model(path):
    model = YOLO(path)
    return model

def display_image_with_bounding_boxes(image, boxes, labels):
    for box, label in zip(boxes, labels):
        x, y, w, h, score, pred = box
        xmin = int((x - w/2) * image.shape[1])
        ymin = int((y - h/2) * image.shape[0])
        xmax = int((x + w/2) * image.shape[1])
        ymax = int((y + h/2) * image.shape[0])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def predict(data :st.runtime.uploaded_file_manager.UploadedFile):
    # Load model
    model = load_yolo_model("assets/best.pt")

    # Inference
    image = Image.open(data)
    image = np.asarray(image)
    results = model.predict(image)

    return image, results

def start():
    # Default
    clicked = False

    # Main app
    st.markdown("# Group Ginger 🫚\n## Medicine Detection and Classification")
    st.write("The images uploaded were identified and classified by YOLO model.")
    
    # Upload an image
    st.markdown("### Test the model")
    data = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    clicked = st.button("Upload")

    # Predict image
    st.markdown("### Results")
    con = st.container()

    if clicked == True:
        try: 
            with st.spinner("Loading results..."):
                image, results = predict(data)

                for result in results:        
                    pred = result.boxes.data[0]                           # extracts class label
                    boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
                    for box in boxes:                                          # iterate boxes
                        r = box.xyxy[0].astype(int)                            # get corner points as int                                           # print boxes
                        cv2.rectangle(image, r[:2], r[2:], (0, 255, 0), 2) # draw boxes on img
                        cv2.putText(image, LABELS[int(pred[5])], (r[0] + 3, r[3] + 14), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 1)
                
                con.image(image, channels="RGB")
        except AttributeError:
            st.error("Please upload an image.")
        except:
            st.error("There are no detections found in the image.")
    
    # Show result
    st.write("This is where the results will be shown.")

    # Member list
    st.markdown("""
    ## Ginger Members 🫚\n
    ### Split 1
    1. Ausan, Reizen Kim
    2. Layson, Sebastian Carlo Enrique Nicolas
    ### Split 2
    1. Mayordo, Zherish Galvin
    2. Villasan, John Michael
    ### Split 3
    1. Bayoneta, Allen Jethro
    2. Valeroso, Sarah Mae
    """)


if __name__ == "__main__":
    start()