import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

CNN_IMAGE_SIZE = 203     # must match CNN input size
CNN_MODEL_PATH = "best_cnn_liveness.h5" #file name

@st.cache_resource
def load_cnn_model():
    model = load_model(CNN_MODEL_PATH)
    return model

cnn_model = load_cnn_model()

st.title("Face Liveness Demo: CNN (Real vs Spoof)")

st.markdown("""
This app uses a **CNN liveness model** to classify a face as:

- **REAL (LIVE)**  
- **SPOOF (FAKE)** (e.g., printed photo / screen)  

Take a picture below and see the result.
""")

img_file = st.camera_input("Take a picture with your webcam")

if img_file is not None:
    # take picture
  
    image = Image.open(img_file)
    st.image(image, caption="Captured frame", use_column_width=True)

    # Convert PIL (RGB) to NumPy (BGR) for OpenCV
    img_rgb = np.array(image)  # RGB
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Resize to CNN input size
    cnn_img = cv2.resize(img_bgr, (CNN_IMAGE_SIZE, CNN_IMAGE_SIZE))

    cnn_img = cnn_img.astype("float32") / 255.0
    cnn_img = np.expand_dims(cnn_img, axis=0)  # shape (1, H, W, 3)

    # Predict
    cnn_pred = cnn_model.predict(cnn_img)

    if cnn_pred.shape[-1] == 1:
        prob_real = float(cnn_pred[0][0])
    else:
        prob_real = float(cnn_pred[0][1])

    live_label = "REAL (LIVE)" if prob_real >= 0.5 else "SPOOF (FAKE)"

    st.subheader("Liveness Prediction -  CNN")
    st.write(f"**Prediction:** {live_label}")
    st.write(f"Probability real: `{prob_real:.3f}`")

    st.info("This model is performing **liveness detection**: "
            "it predicts whether the captured face looks like a real live person "
            "or a spoof (e.g., printed photo or screen).")
else:
    st.info("Please take a picture with your webcam to run the liveness model.")
