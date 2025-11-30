"""
pad.py

Usage:
  - Compute heuristic liveness score for images 
  - Based on blur, saturation, specularity, color banding

"""

from pathlib import Path
import numpy as np
import cv2
import math

# laplacian variance as blur measure (higherer variance = sharper image)
def laplacian_var(gray):
    """ Laplacian variance as blur measure"""
    # gray: single-channel grayscale image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# median saturation in [0,1]
def saturation_score(bgr):
    """Return median saturation normalized to [0,1]"""
    # hsv: color space (hue, saturation, value)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[...,1].astype(np.float32)
    med = float(np.median(s))
    return med / 255.0

# fraction of very bright pixels in V channel
def specularity_score(bgr):
    """Heuristic for specular hotspots: fraction of very bright pixels in V channel"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[...,2]
    thr = 240   # threshold for "very bright" pixel
    # counts number of pixels that are very bright
    frac = float((v >= thr).sum()) / (v.size + 1e-12)
    return frac

# normalized variance of mean values in B,G,R channels
def color_band_score(bgr):
    """Heuristic: strong channel imbalance may indicate printing or screen artifact"""
    chans = [bgr[...,i].astype(np.float32) for i in range(3)]
    means = [c.mean() for c in chans]
    # normalized variance of channel means
    mv = float(np.var(means) / (np.mean(means)+1e-6))
    mv = min(1.0, mv*2.0)
    return mv

# computes heuristic liveness score in [0,1]
def heuristic_liveness_score(img_bgr):
    """
    Compute a heuristic liveness score in [0,1]. Higher = more likely live.
    Uses blur (Laplacian), saturation, specularity, color banding heuristics.
    """
    try:
        h, w = img_bgr.shape[:2]    # height, width
        # resize large images for speed
        if max(h,w) > 800:
            img_bgr = cv2.resize(img_bgr, (800 * w // max(h,w), 800 * h // max(h,w)))
        # convert to grayscale for blur measure
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = laplacian_var(gray)
        # color measures
        sat = saturation_score(img_bgr)
        spec = specularity_score(img_bgr)
        color_band = color_band_score(img_bgr)
        # heuristics:
        # blur: high var = sharp = more live
        b_norm = 1.0 - math.exp(-blur / 100.0) 
        # combine: prefer high saturation, low specularity, low color_band
        score = 0.5 * b_norm + 0.3 * sat + 0.2 * (1.0 - spec) - 0.2 * color_band
        score = float(max(0.0, min(1.0, score)))
        return score
    except Exception:
        return 0.5

# compute liveness scores for a list of image paths
def compute_liveness_for_paths(image_paths, heuristic_only=True, verbose=False):
    """
    image_paths: iterable of file paths
    Returns list of dicts: {'image_path':..., 'liveness_score':...}
    """
    results = []
    for p in image_paths:
        pstr = str(p)
        try:
            img = cv2.imread(pstr)
            if img is None:
                results.append({'image_path': pstr, 'liveness_score': float('nan')})
                continue
            score = heuristic_liveness_score(img)
            results.append({'image_path': pstr, 'liveness_score': float(score)})
        except Exception:
            results.append({'image_path': pstr, 'liveness_score': float('nan')})
    return results