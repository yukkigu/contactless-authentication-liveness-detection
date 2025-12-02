"""
build_templates.py

Usage:
  - Build HQ per-person templates (pooled + pose-aware) for a given split and save them along with a FAISS index.

  - Outputs are written to: <data_root>/<split>/enrolled_users/templates/
      - templates_pose_hq.npy          (L2-normalized templates)
      - templates_map_pose_hq.csv      (metadata; includes 'type' column)
      - templates_pose_hq.index        (FAISS IndexFlatIP over templates)
"""

# =========================
# Imports & global configs
# =========================

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import faiss

# ID column name, same as in your CSVs
ID_COL = "person_id"

# HQ filter knobs
MIN_FACE_SIDE = 80                  # minimum face side (in pixels) to consider HQ
MIN_LIVENESS = 0.0                  # minimum liveness_score to consider HQ
HQ_MIN_IMAGES_PER_TEMPLATE = 2      # require >= this many HQ images for a per-pose template
HQ_MIN_QUALITY = 0.4                # min quality_score; adjust to your taste

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_templates")


# =========================
# Helper: pose binning from yaw
# =========================

def yaw_to_bin(yaw_deg: float) -> int:
    """
    Map yaw (deg) into 5 coarse bins:
      0: far-left
      1: left
      2: frontal
      3: right
      4: far-right

    This is used if your CSV does NOT already have a 'pose_cluster_k5'
    column. If it DOES, we will use that instead.
    """
    if yaw_deg <= -60:
        return 0
    elif yaw_deg <= -20:
        return 1
    elif yaw_deg < 20:
        return 2
    elif yaw_deg < 60:
        return 3
    else:
        return 4


# =========================
# For pose-aware HQ templates: quality_score
# (from enrollment.ipynb, Cell 5: quality_score logic – reconstructed)
# =========================

def compute_quality_score(row,
                          min_face_side: float = MIN_FACE_SIDE,
                          max_face_side: float = 200.0) -> float:
    """
    Computes quality_score for a single row, based on:
    - Combine liveness_score and face size (min_side_est) into a single scalar.
    - Optionally incorporate detection confidence if column is present.

    Assumed structure:

        side_norm = (min_side_est - min_face_side) / (max_face_side - min_face_side)
        side_norm = clipped to [0, 1]

        quality = 0.6 * liveness_score + 0.3 * side_norm
        + 0.1 * det_conf  (if face_det_conf column exists)

    Returns:
        quality_score in [0, 1] (clipped).
    """
    liveness = float(row.get("liveness_score", 0.0))

    # Approximate min_side_est = sqrt(area)
    box_area = float(row.get("face_box_area", 0.0))
    min_side_est = max(0.0, box_area ** 0.5)

    # Normalize face side into [0,1] with respect to MIN_FACE_SIDE and max_face_side
    if max_face_side <= min_face_side:
        side_norm = 0.0
    else:
        side_norm = (min_side_est - min_face_side) / (max_face_side - min_face_side)
        side_norm = max(0.0, min(1.0, side_norm))

    # Optional detection confidence (if you had something like face_det_conf)
    det_conf = 0.0
    if "face_det_conf" in row:
        try:
            det_conf = float(row["face_det_conf"])
        except Exception:
            det_conf = 0.0

    # Weighting following your comments: liveness dominates, then size, then det_conf
    quality = 0.6 * liveness + 0.3 * side_norm + 0.1 * det_conf
    quality = max(0.0, min(1.0, quality))
    return quality


# =========================
# For pose-aware HQ templates: HQ mask per row
# =========================

def row_is_hq(row,
              min_face_side: float = MIN_FACE_SIDE,
              min_liveness: float = MIN_LIVENESS,
              min_quality: float = HQ_MIN_QUALITY) -> bool:
    """
    Decide whether a single row is "HQ" based on:

    - face_aligned flag
    - min_side_est >= MIN_FACE_SIDE
    - liveness_score >= MIN_LIVENESS
    - quality_score >= HQ_MIN_QUALITY
    - yaw not NaN
    Returns:
        True if row is HQ, else False.
    """

    # alignment
    if not bool(row.get("face_aligned", False)):
        return False

    # size
    box_area = float(row.get("face_box_area", 0.0))
    min_side_est = max(0.0, box_area ** 0.5)
    if min_side_est < min_face_side:
        return False

    # liveness
    live = float(row.get("liveness_score", 0.0))
    if live < min_liveness:
        return False

    # pose validity
    if pd.isna(row.get("yaw", np.nan)):
        return False

    # quality_score
    q = compute_quality_score(row, min_face_side=min_face_side)
    if q < min_quality:
        return False

    return True

# load existing HQ gallery templates + map + index to add new users
def load_gallery_hq(
    split: str = "enroll",
    data_root: str | Path = "../data_processed/vggface2",
    out_name_prefix: str = "templates_all_enroll_hq",
):
    """
    Load HQ gallery templates, map, and FAISS index from:
        <data_root>//enrolled_users/templates/
    """
    data_root = Path(data_root)
    
    # If caller passed the embeddings dir (e.g. .../vggface2/embeddings),
    # use the parent (vggface2) so enrolled_users lives at .../vggface2/enrolled_users
    p = data_root
    if p.name == "embeddings":
        base_dir = p.parent
    else:
        # if 'embeddings' appears anywhere in the path, take everything before it
        try:
            idx = list(p.parts).index("embeddings")
            base_dir = Path(*p.parts[:idx])
        except ValueError:
            base_dir = p

    out_dir = base_dir / "enrolled_users"

    tpl_path = out_dir / f"{out_name_prefix}.npy"
    map_path = out_dir / f"{out_name_prefix}_map.csv"
    idx_path = out_dir / f"{out_name_prefix}.index"

    if not tpl_path.exists() or not map_path.exists() or not idx_path.exists():
        raise FileNotFoundError(f"Gallery files not found at {out_dir}. Did you run build_gallery_hq?")

    templates = np.load(tpl_path).astype(np.float32)
    templates_map = pd.read_csv(map_path)
    index = faiss.read_index(str(idx_path))

    return templates, templates_map, index, out_dir

# enroll a new user from camera with HQ template 
def enroll_new_user_hq(
    new_person_id: str,
    frame_embs: np.ndarray,
    frame_liveness: np.ndarray,
    frame_face_box_area: np.ndarray,
    frame_face_aligned: np.ndarray,
    split: str = "enroll",
    data_root: str | Path = "../data_processed/vggface2/embeddings",
    out_name_prefix: str = "templates_enroll_hq",
):
    """
    Online enrollment:
      - Build ONE HQ template for a NEW user from webcam frames
      - Append to existing HQ gallery in:
          <data_root>/<split>/enrolled_users/templates/

    frame_embs            : (N, D)
    frame_liveness        : (N,)
    frame_face_box_area   : (N,)
    frame_face_aligned    : (N,)
    """
    # Load existing gallery
    templates, templates_map, index, out_dir = load_gallery_hq(
        split=split,
        data_root=data_root,
        out_name_prefix=out_name_prefix,
    )

    # Build df_person for this new user
    N, D = frame_embs.shape
    df_person = pd.DataFrame({
        ID_COL: [new_person_id] * N,
        "embedding_index": np.arange(N),
        "liveness_score": frame_liveness,
        "face_box_area": frame_face_box_area,
        "face_aligned": frame_face_aligned,
    })

    # Use the SAME HQ logic as offline
    tpl_new = build_user_template_hq(df_person, frame_embs)  # (D,)

    # L2-normalize new template
    tpl_new = tpl_new.astype(np.float32)
    norm = np.linalg.norm(tpl_new) + 1e-12
    tpl_new_norm = tpl_new / norm

    # --- Check if this ID already exists ---
    existing_mask = (templates_map[ID_COL] == new_person_id)
    if existing_mask.any():
        # Update existing template in-place
        existing_row = templates_map[existing_mask].iloc[0]
        t_idx = int(existing_row["template_index"])
        templates[t_idx] = tpl_new_norm
        logger.info(f"[enroll_new_user_hq] Updated existing template for {new_person_id} at index {t_idx}")
    else:
        # Append as a new template
        t_idx = templates.shape[0]
        templates = np.vstack([templates, tpl_new_norm[None, :]])

        new_row = {
            ID_COL: new_person_id,
            "template_index": t_idx,
        }
        templates_map = pd.concat(
            [templates_map, pd.DataFrame([new_row])],
            ignore_index=True,
        )
        logger.info(f"[enroll_new_user_hq] Added new template for {new_person_id} at index {t_idx}")


    # # Append to gallery arrays
    # old_count = templates.shape[0]
    # templates_updated = np.vstack([templates, tpl_new_norm[None, :]])

    # new_row = {
    #     ID_COL: new_person_id,
    #     "template_index": old_count,
    # }
    # templates_map_updated = pd.concat(
    #     [templates_map, pd.DataFrame([new_row])],
    #     ignore_index=True,
    # )

    # --- Rebuild FAISS index from scratch (simpler & correct) ---
    dim = templates.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(templates.astype(np.float32))

    # Save updated gallery
    tpl_path = out_dir / f"{out_name_prefix}.npy"
    map_path = out_dir / f"{out_name_prefix}_map.csv"
    idx_path = out_dir / f"{out_name_prefix}.index"

    np.save(tpl_path, templates.astype(np.float32))
    templates_map.to_csv(map_path, index=False)
    faiss.write_index(index, str(idx_path))

    print(f"[enroll_new_user_hq] Enrolled/updated {new_person_id} at template_index={t_idx}")
    print(f"[enroll_new_user_hq] Updated templates: {tpl_path}  shape={templates.shape}")
    print(f"[enroll_new_user_hq] Updated map:       {map_path}  rows={len(templates_map)})")

    return templates, templates_map, index


# =========================
# Core reusable function that builds templates for users
# build_user_template_hq 
# =========================

def build_user_template_hq(
    df_person: pd.DataFrame,
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Build ONE HQ-pooled template for ONE user.

    Steps:
      1) Filter this person's rows with HQ rules (row_is_hq).
      2) If there are HQ rows:
           - use ONLY their embeddings to build the template.
         Else:
           - fall back to ALL embeddings for this user.
      3) Return a single centroid vector (D,).

    """
    df_person = df_person.copy()
    if "quality_score" not in df_person.columns:
        df_person["quality_score"] = df_person.apply(compute_quality_score, axis=1)
    if "is_hq" not in df_person.columns:
        df_person["is_hq"] = df_person.apply(row_is_hq, axis=1)

    df_hq = df_person[df_person["is_hq"]]

    if len(df_hq) > 0:
        # use HQ frames only
        idxs = df_hq["embedding_index"].to_numpy()
    else:
        # fallback: use all frames
        idxs = df_person["embedding_index"].to_numpy()

    if len(idxs) == 0:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    embs = embeddings[idxs]
    centroid = embs.mean(axis=0).astype(np.float32)
    return centroid


# =========================
# Top-level: build hq template gallery for a split (offline)
#
# - load enroll CSV + embeddings
# - loop over all person_id
# - collect templates + rows into big arrays
# - L2 normalize, save npy/csv, build FAISS index
# =========================

def build_template_gallery_hq(
    split: str = "enroll",
    data_root: str | Path = "../data_processed/vggface2",
    out_name_prefix: str = "templates_all_enroll_hq",
):
    """
    Offline helper:
      - load <split>/embeddings_map_with_pose_k5.csv + embeddings.npy
      - for each user: build ONE HQ-pooled template
      - L2-normalize, save npy/csv/index.

    Idea: "enroll" every user using HQ frames.
    """
    data_root = Path(data_root)
    emb_dir = data_root / "embeddings" / split

    map_csv = emb_dir / "embeddings_map_with_pose_k5.csv"
    emb_npy = emb_dir / "embeddings.npy"

    df = pd.read_csv(map_csv)
    embeddings = np.load(emb_npy).astype(np.float32)

    assert df["embedding_index"].max() < embeddings.shape[0], \
        "embedding_index out of range for embeddings.npy"
    
    # output dir
    out_dir = data_root / "enrolled_users" 
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = []
    rows = []

    for person_id, df_person in df.groupby(ID_COL):
        tpl = build_user_template_hq(df_person, embeddings)
        templates.append(tpl)
        rows.append(
            {
                ID_COL: person_id,
                "template_index": len(rows),
            }
        )

    templates = np.vstack(templates).astype(np.float32)
    templates_map = pd.DataFrame(rows)
    templates_map["template_index"] = np.arange(len(templates_map))

    # L2-normalize
    norms = np.linalg.norm(templates, axis=1, keepdims=True).clip(min=1e-12)
    templates_norm = templates / norms

    # save outputs
    tpl_path = out_dir / f"{out_name_prefix}.npy"
    map_path = out_dir / f"{out_name_prefix}_map.csv"
    idx_path = out_dir / f"{out_name_prefix}.index"

    np.save(tpl_path, templates_norm)
    templates_map.to_csv(map_path, index=False)

    dim = templates_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(templates_norm)
    faiss.write_index(index, str(idx_path))

    print(f"Saved HQ templates: {tpl_path} (shape={templates_norm.shape})")
    print(f"Saved HQ map:       {map_path} (rows={len(templates_map)})")
    print(f"Saved HQ index:     {idx_path}")

    return templates_norm, templates_map, index

# For pose-aware HQ templates, use:
def build_template_gallery_pose_hq(
    split: str = "enroll",
    data_root: str | Path = "../data_processed/vggface2",
    out_name_prefix: str = "templates_pose_hq",
    pose_col_candidates=("pose_cluster","pose_cluster_k5"),
    min_images_per_template: int = 1,
):
    """
    Build pose-aware HQ templates: one template per (person_id, pose_bin).
    Requires embeddings_map with a pose column (falls back to yaw->bin).
    """
    data_root = Path(data_root)
    emb_dir = data_root / "embeddings" / split
    map_csv = emb_dir / "embeddings_map_with_pose_k5.csv"
    emb_npy = emb_dir / "embeddings.npy"

    df = pd.read_csv(map_csv)
    embs = np.load(emb_npy).astype(np.float32)

    # pick pose column or fallback to coarse yaw bins
    pose_col = None
    for c in pose_col_candidates:
        if c in df.columns:
            pose_col = c
            break
    if pose_col is None:
        def yaw_to_bin(y):
            if pd.isna(y): return -1
            if y < -15: return 0
            if y > 15: return 2
            return 1
        df["pose_bin"] = df["yaw"].apply(yaw_to_bin)
        pose_col = "pose_bin"

    out_dir = data_root / "enrolled_users"
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = []
    rows = []

    for person_id, g_person in df.groupby("person_id"):
        for pose_label, g in g_person.groupby(pose_col):
            if pd.isna(pose_label) or int(pose_label) < 0:
                continue
            idxs = g["embedding_index"].to_numpy()
            if len(idxs) < min_images_per_template:
                continue
            em = embs[idxs]
            centroid = em.mean(axis=0).astype(np.float32)
            rows.append({
                "person_id": person_id,
                "pose_label": int(pose_label),
                "template_index": len(rows),
                "type": "pose_hq",
                "n_images": len(idxs),
            })
            templates.append(centroid)

    if len(templates) == 0:
        templates_arr = np.zeros((0, embs.shape[1]), dtype=np.float32)
    else:
        templates_arr = np.vstack(templates).astype(np.float32)

    # L2-normalize
    if templates_arr.shape[0] > 0:
        norms = np.linalg.norm(templates_arr, axis=1, keepdims=True).clip(min=1e-12)
        templates_norm = templates_arr / norms
    else:
        templates_norm = templates_arr

    templates_map = pd.DataFrame(rows)
    templates_map["template_index"] = np.arange(len(templates_map))

    tpl_path = out_dir / f"{out_name_prefix}.npy"
    map_path = out_dir / f"{out_name_prefix}_map.csv"
    idx_path = out_dir / f"{out_name_prefix}.index"

    np.save(tpl_path, templates_norm)
    templates_map.to_csv(map_path, index=False)

    dim = templates_norm.shape[1] if templates_norm.size else embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    if templates_norm.shape[0] > 0:
        index.add(templates_norm)
    faiss.write_index(index, str(idx_path))

    print(f"Saved pose HQ templates: {tpl_path} (shape={templates_norm.shape})")
    print(f"Saved pose HQ map:       {map_path} (rows={len(templates_map)})")
    print(f"Saved pose HQ index:     {idx_path}")

    return templates_norm, templates_map, index


# =========================
# CLI entry point
# run: python build_templates.py --split enroll
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="enroll",
        help="Which split directory under embeddings/ to process (e.g., enroll, val)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../data_processed/vggface2/embeddings",
        help="Root directory that contains <split> subfolders.",
    )
    args = parser.parse_args()

    build_template_gallery_hq(split=args.split, data_root=args.data_root)
