import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import csv

import streamlit as st
from tensorflow.keras.models import load_model
import faiss

from scripts.liveness import LiveFacePipelineFull


#==============================
# CONFIGURATION
#==============================

BASE_DIR = Path(__file__).resolve().parent

CNN_IMAGE_SIZE = 203     # must match CNN input size
CNN_MODEL_PATH = "best_cnn_liveness.h5" #file name

ENROLL_DIR = BASE_DIR / "data_processed" / "vggface2" / "enrolled_users"
ENROLL_INDEX_PATH = ENROLL_DIR / "templates_all_enroll_hq.index"
ENROLL_MAP_PATH   = ENROLL_DIR / "templates_all_enroll_hq_map.csv"

THRESHOLD = 0.7

#==============================
# LOAD MODELS
#==============================

@st.cache_resource
def load_cnn_model():
    model = load_model(CNN_MODEL_PATH)
    return model

@st.cache_resource
def load_live_pipeline():
    pipeline = LiveFacePipelineFull(
        device=None,
        min_face_side=80,
        min_conf=0.9,
        image_size=160,
    )
    return pipeline

cnn_model = load_cnn_model()
live_pipeline = load_live_pipeline()

#=============================
# FAISS-BASED ENROLLMENT
#=============================

def load_faiss():
    """
    Load FAISS index and enrollment map.
    """
    if ENROLL_INDEX_PATH.exists() and ENROLL_MAP_PATH.exists():
        index = faiss.read_index(str(ENROLL_INDEX_PATH))

        user_ids = []
        with ENROLL_MAP_PATH.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # non-empty line
                    user_ids.append(row[0])

        return index, user_ids
    else:
        return None, []

def save_faiss(index, user_ids):
    """
    Save FAISS index and enrollment map.
    """
     # Make sure the directory exists
    ENROLL_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(ENROLL_INDEX_PATH))

    with ENROLL_MAP_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        for uid in user_ids:
            writer.writerow([uid])

def add_enrollment_faiss(user_id, embedding):
    """
    Enroll new user to FAISS index.
    """
    emb = np.asarray(embedding, dtype=np.float32)
    # Normalize for cosine similarity
    emb = emb / (np.linalg.norm(emb) + 1e-12)

    index, user_ids = load_faiss()
    print(f"Loaded FAISS index with {len(user_ids)} users.")

    if index is None:
        # Create new index
        dim = emb.shape[0]
        index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity

    index.add(emb.reshape(1, -1))
    user_ids.append(user_id)

    print(f"Saving {user_id} to FAISS index.")
    save_faiss(index, user_ids)

def match_identity_faiss(embedding, threshold=THRESHOLD):
    """
    Match embedding against enrolled users using FAISS.
    Returns user_id if match found, else None.
    """
    index, user_ids = load_faiss()
    if index is None or len(user_ids) == 0:
        return None, 0.0

    emb = np.asarray(embedding, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-12)

    D, I = index.search(emb.reshape(1, -1), k=1)  # Search for the nearest neighbor
    similarity = float(D[0][0])
    idx = int(I[0][0])

    if idx < 0:
        return None, similarity

    best_user = user_ids[idx]

    if similarity >= threshold:
        return best_user, similarity
    else:
        return None, similarity

#==============================
# Helpers
#==============================

def pil_to_bgr(pil_img):
    """Convert PIL Image to OpenCV BGR format."""
    img_rgb = np.array(pil_img)  # RGB
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr

def run_liveness_cnn(img_bgr):
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
    is_live = prob_real >= 0.5
    return prob_real, live_label, is_live

def run_full_pil_pipeline(pil_img):
    """
    PIL image to BGR frame to LiveFacePipelineFull.process_frame().
    """
    frame_bgr = pil_to_bgr(pil_img)
    results = live_pipeline.process_frame(frame_bgr)
    return frame_bgr, results


#==============================
# STREAMLIT APP
#==============================

st.title("Face Liveness Demo: ")
st.subheader("Identity Match + CNN (Real vs Spoof)")
st.markdown("""
This app combines:
1. **Face liveness detection** (MTCNN + InceptionResnetV1 + heuristic liveness + pose)
2. **CNN-based liveness classification** (Real vs Spoof)
3. **FAISS-based and identity matching** with incremental enrollment 

Enrolls a "New User" with a captured live face embedding

Uses a **CNN liveness model** to classify a face as:

- **REAL (LIVE)**  
- **SPOOF (FAKE)** (e.g., printed photo / screen)  

Take a picture below and see the result.
""")

tab_enroll, tab_verify = st.tabs(["Enroll New User", "Verify / Login"])

#=============================
# TAB: ENROLL NEW USER
#=============================
with tab_enroll:
    st.header("Enroll New User")

    user_id = st.text_input("Enter User ID / name to enroll (e.g., user123)")

    img_file = st.camera_input("Take a picture with your webcam for enrollment")
    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption="Captured frame for enrollment", use_container_width=True)

        # Run full pipeline
        frame_bgr, results = run_full_pil_pipeline(image)

        emb = results.get("embedding", None)
        heuristic_live = results.get("liveness_score", 0.0)
        yaw = results.get("yaw", np.nan)
        pitch = results.get("pitch", np.nan)
        roll = results.get("roll", np.nan)

        st.subheader("Liveness Heuristic Results")
        st.write(f"- Heuristic liveness score: `{heuristic_live:.3f}`(0-1)")
        st.write(f"- Head pose (yaw, pitch, roll): `{yaw:.1f}`, `{pitch:.1f}`, `{roll:.1f}`")
        
        if emb is None:
            st.error("No face / embedding detected. Please try again with a clear frontal face image.")
        else:
            st.success("Face embedding extracted successfully.")

            # CNN liveness
            prob_real, live_label, is_live = run_liveness_cnn(frame_bgr)
            st.subheader("Liveness Prediction -  CNN")
            if is_live:
                st.success(f"**Prediction:** {live_label}")
            else:
                st.error(f"**Prediction:** {live_label}")

            st.write(f"Probability real: `{prob_real:.3f}`")
            if not is_live:
                st.error("**SPOOF**: Liveness check failed. Cannot enroll spoofed face.")
            else:
                if user_id.strip() =="":
                    st.warning("Please enter a valid User ID / name to enroll.")
                else:
                    if st.button("Enroll User"):
                        add_enrollment_faiss(user_id.strip(), emb)
                        st.success(f"User '{user_id}' enrolled successfully!")

#=============================
# TAB: VERIFY / LOGIN
#=============================
with tab_verify:
    st.header("Verify / Login")

    img_file2 = st.camera_input("Take a picture with your webcam for verification")

    if img_file2 is not None:
        image2 = Image.open(img_file2)
        st.image(image2, caption="Captured frame for verification", use_container_width=True)

        # Run full pipeline
        frame_bgr2, results2 = run_full_pil_pipeline(image2)

        emb2 = results2.get("embedding", None)
        heuristic_live2 = results2.get("liveness_score", 0.0)
        yaw2 = results2.get("yaw", np.nan)
        pitch2 = results2.get("pitch", np.nan)
        roll2 = results2.get("roll", np.nan)

        st.subheader("Liveness Heuristic Results")
        st.write(f"- Heuristic liveness score: `{heuristic_live2:.3f}`(0-1)")
        st.write(f"- Head pose (yaw, pitch, roll): `{yaw2:.1f}`, `{pitch2:.1f}`, `{roll2:.1f}`")

        # CNN liveness
        prob_real2, live_label2, is_live2 = run_liveness_cnn(frame_bgr2)
        st.subheader("Liveness Prediction -  CNN")
        st.write(f"**Prediction:** {live_label2}")
        st.write(f"Probability real: `{prob_real2:.3f}`")

        if not is_live2:
            st.error("**SPOOF**: Liveness check failed. Cannot verify spoofed face.")
        else:
            st.success("Liveness passed: continuing to FAISS identity match...")

            if emb2 is  None:
                st.error("No face / embedding detected. Please try again with a clear frontal face image.")
            else:
                user_id_pred, score = match_identity_faiss(emb2, threshold=THRESHOLD)

                st.subheader("FAISS Identity Match Results")
                if user_id_pred is None:
                    st.error("NOT AUTHORIZED: No matching user found.")
                    st.write(f"Best similarity score: `{score:.3f}` (threshold: `{THRESHOLD}`)")
                else:
                    st.success(f"AUTHORIZED: Welcome back, **{user_id_pred}**!")
                    st.write(f"Similarity score: `{score:.3f}` (threshold: `{THRESHOLD}`)")
    else:
        st.info("Please take a picture with your webcam to run the identity + verification process.")
