from scripts.pad import heuristic_liveness_score
from scripts.pose import landmarks_to_pose
from scripts.embeddings import preprocess_pil

import numpy as np
import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from pathlib import Path

class LiveFacePipelineFull:
    def __init__(self,
                 device=None,
                 min_face_side=80,
                 min_conf=0.9,
                 image_size=160):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.min_face_side = min_face_side
        self.min_conf = min_conf
        self.image_size = image_size

        # MTCNN face detector (detects 5 landmarks too)
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        # InceptionResnetV1 embedder
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        for p in self.embedder.parameters():
            p.requires_grad = False

    def _run_mtcnn(self, img_pil):
        boxes, probs, landmarks = self.mtcnn.detect(img_pil, landmarks=True)
        if boxes is None or probs is None or landmarks is None:
            return None, None, None, None

        best = int(np.argmax(probs))
        return boxes[best], float(probs[best]), landmarks[best], True

    def _compute_embedding(self, face_pil):
        arr = preprocess_pil(face_pil, size=self.image_size)
        t = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.embedder(t).cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb.astype(np.float32)

    def process_image(self, path):
        path = Path(path)
        if not path.exists():
            return self._empty_return()

        img_pil = Image.open(path).convert("RGB")
        frame_rgb = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        return self._process(frame_bgr, img_pil)

    def process_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        return self._process(frame_bgr, img_pil)

    def _process(self, frame_bgr, img_pil):

        # --- Detection ---
        box, prob, landmarks, ok = self._run_mtcnn(img_pil)
        if not ok:
            return self._empty_return()

        x1, y1, x2, y2 = [int(x) for x in box]
        bw, bh = x2 - x1, y2 - y1
        box_area = float(bw * bh)
        min_side = min(bw, bh)
        aligned = (min_side >= self.min_face_side) and (prob >= self.min_conf)

        # --- Pose ---
        w, h = img_pil.size
        yaw, pitch, roll = landmarks_to_pose(landmarks, (w, h))

        # --- Face crop for embedding ---
        face_rgb = frame_bgr[y1:y2, x1:x2]
        if face_rgb.size == 0:
            emb = None
        else:
            face_pil = Image.fromarray(face_rgb[..., ::-1])  # BGR->RGB
            emb = self._compute_embedding(face_pil)

        # --- Liveness ---
        live = float(heuristic_liveness_score(frame_bgr))

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "landmarks": landmarks.tolist(),
            "liveness_score": live,
            "face_aligned": bool(aligned),
            "face_box_area": box_area,
            "embedding": emb,
        }

    def _empty_return(self):
        return {
            "yaw": np.nan,
            "pitch": np.nan,
            "roll": np.nan,
            "landmarks": None,
            "liveness_score": 0.0,
            "face_aligned": False,
            "face_box_area": 0.0,
            "embedding": None,
        }
