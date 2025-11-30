"""
pose.py

Usage:
  - Convert 5-point landmarks to yaw, pitch, roll:
  - 

"""
from pathlib import Path
import numpy as np
import cv2

# 3D model points chosen as approximate coordinates for nose, left eye, right eye, left mouth, right mouth
_MODEL_POINTS_5 = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (-30.0, 30.0, -20.0),   # left eye (approx)
    (30.0, 30.0, -20.0),    # right eye (approx)
    (-25.0, -25.0, -20.0),  # left mouth corner (approx)
    (25.0, -25.0, -20.0)    # right mouth corner (approx)
], dtype=np.float64)

def landmarks_to_pose_5(landmarks, image_size):
    """
    landmarks: (5,2) in order [left_eye, right_eye, nose, mouth_left, mouth_right]
    image_size: (width, height)
    returns: yaw, pitch, roll (degrees) or (nan,nan,nan)
    """
    try:
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.shape[0] != 5:
            return np.nan, np.nan, np.nan
        # Build image_points to match _MODEL_POINTS_5 ordering:
        # [nose, left_eye, right_eye, left_mouth, right_mouth] -> reorder accordingly
        image_points = np.array([
            lm[2],  # nose
            lm[0],  # left eye
            lm[1],  # right eye
            lm[3],  # left mouth
            lm[4]   # right mouth
        ], dtype=np.float64)

        w, h = image_size
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        # Need at least 4 points. We have 5 -> ok.
        ok, rvec, tvec = cv2.solvePnP(_MODEL_POINTS_5, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return np.nan, np.nan, np.nan
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
        if sy < 1e-6:
            x = np.arctan2(-rmat[1,2], rmat[1,1])
            y = np.arctan2(-rmat[2,0], sy)
            z = 0
        else:
            x = np.arctan2(rmat[2,1], rmat[2,2])
            y = np.arctan2(-rmat[2,0], sy)
            z = np.arctan2(rmat[1,0], rmat[0,0])
        return float(np.degrees(y)), float(np.degrees(x)), float(np.degrees(z))
    except Exception:
        return np.nan, np.nan, np.nan