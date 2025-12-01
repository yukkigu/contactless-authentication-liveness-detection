"""
embeddings.py

Usage:
  - Create shards and compute embeddings:
      python scripts/embeddings.py run --manifest ../data_processed/vggface2/manifests/manifest_basic.csv --out data_processed/vggface2/embeddings --shard-size 5000 --batch-size 128

  - Merge shard files into final outputs:
      python scripts/embeddings.py merge --out data_processed/vggface2/embeddings --emb-dim 512

  - Quick status:
      python scripts/embeddings.py status --out data_processed/vggface2/embeddings
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
from tqdm.auto import tqdm
import os

LOGGER = logging.getLogger("embeddings")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# load manifest.csv to dataframe and return cropped or image path
def load_manifest(manifest_path):
    df = pd.read_csv(manifest_path)
    # pick path column (existing behavior)
    path_col = "image_path" if "image_path" in df.columns else df.columns[0]
    repo_root = Path(__file__).resolve().parents[1]  # repo root
    
    def _find_and_rel(p):
        p = str(p)
        cand = Path(p)
        if cand.exists():
            found = cand.resolve()
        else:
            # try relative to manifest parent
            mparent = Path(manifest_path).parent
            c2 = (mparent / p)
            if c2.exists():
                found = c2.resolve()
            else:
                # try relative to repo root
                c3 = (repo_root / p)
                if c3.exists():
                    found = c3.resolve()
                else:
                    # fallback: return original string (file missing)
                    return p
        # return repo-relative path if possible (keeps manifest portable)
        try:
            return os.path.relpath(found, repo_root).replace("\\", "/")
        except Exception:
            return str(found)
    df[path_col] = df[path_col].astype(str).map(_find_and_rel)
    return df, path_col

# convert string path to Path object
def resolve_path(p_str):
    p = Path(p_str)
    if p.is_absolute():
        return p
    # resolve relative paths relative to repo root
    repo_root = Path(__file__).resolve().parents[1]
    cand = (repo_root / p)
    if cand.exists():
        return cand
    return Path.cwd() / p

# resizes an image and convert to RGB
def preprocess_pil(pil_img, size=160):
    img = pil_img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    arr = (arr - 127.5) / 128.0     # scale pixels 
    # return NumPy array in (C, H, W)
    return np.transpose(arr, (2, 0, 1))

# normalizes each row
def l2_norm_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0     # to avoid division by zero
    return x / norms

# scan existing shard map files to find already processed image paths 
def existing_processed_paths(out_dir):
    out = Path(out_dir)
    # find all existing shard map files
    maps = list(out.glob("*_map_shard_*.csv")) + list(out.glob("embeddings_map_shard_*.csv"))
    processed = set()
    # get repo root for relative path resolution
    repo_root = Path(__file__).resolve().parents[1]
    for m in maps:
        try:
            df = pd.read_csv(m)
            if "image_path" in df.columns:
                for v in df["image_path"].astype(str).tolist():
                    p = Path(v)
                    # convert to repo-relative posix string
                    if p.is_absolute():
                        try:
                            processed.add(Path(os.path.relpath(p, repo_root)).as_posix())
                        except Exception:
                            processed.add(str(p))
                    else:
                        # keep repo-relative / normalized posix string
                        processed.add(Path(v).as_posix())
        except Exception:
            continue
    return processed

# run model on images in manifest in batches, save to sharded .npy and .csv files
def embed_sharded(manifest_path, out_dir, batch_size=128, image_size=160, shard_size=5000, device=None, sample_limit=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_df, path_col = load_manifest(manifest_path)
    if sample_limit:
        manifest_df = manifest_df.iloc[:sample_limit].copy()
    rows = manifest_df.to_dict("records")
    processed = existing_processed_paths(out_dir)
    todo = [r for r in rows if str(r.get(path_col, "")) not in processed]
    if not todo:
        LOGGER.info("No new rows to process.")
        return

    # load InceptionResnetV1 model
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    shard_idx = 0
    batch_images = []
    batch_meta = []
    shard_embeddings = []
    shard_map_rows = []
    total_processed = 0

    # function to flush current shard to disk to avoid large memory usage
    def flush_shard(si):
        if not shard_embeddings:
            return
        emb_arr = np.stack(shard_embeddings).astype("float32")
        emb_path = out_dir / f"embeddings_shard_{si:04d}.npy"
        map_path = out_dir / f"embeddings_map_shard_{si:04d}.csv"
        np.save(emb_path, emb_arr)
        pd.DataFrame(shard_map_rows).to_csv(map_path, index=False)
        LOGGER.info("Wrote shard %s (rows=%d)", emb_path.name, emb_arr.shape[0])

    for r in tqdm(todo, desc="rows", mininterval=0.5, leave=True):
        p_rel = r.get(path_col, "")
        p = resolve_path(p_rel)
        try:
            pil = Image.open(p)
            arr = preprocess_pil(pil, size=image_size)
        except Exception:
            shard_map_rows.append({
                "embedding_index": -1,
                "image_path": str(p_rel),
                "dataset": r.get("dataset", ""),
                "person_id": r.get("person_id", ""),
                "sha1": r.get("sha1", "")
            })
            continue

        batch_images.append(arr)
        batch_meta.append((r, p_rel))

        if len(batch_images) >= batch_size:
            x = np.stack(batch_images)  
            x = x.astype(np.float32)
            t = torch.from_numpy(x).to(device)
            with torch.no_grad():
                out = model(t).cpu().numpy()
            out = l2_norm_rows(out).astype("float32")
            for j in range(out.shape[0]):
                shard_embeddings.append(out[j])
                shard_map_rows.append({
                    "embedding_index": len(shard_map_rows),  # local index inside shard
                    "image_path": str(batch_meta[j][1]),
                    "dataset": batch_meta[j][0].get("dataset", ""),
                    "person_id": batch_meta[j][0].get("person_id", ""),
                    "sha1": batch_meta[j][0].get("sha1", "")
                })
            batch_images = []
            batch_meta = []

            if len(shard_embeddings) >= shard_size:
                flush_shard(shard_idx)
                total_processed += len(shard_embeddings)
                shard_idx += 1
                shard_embeddings = []
                shard_map_rows = []

    # final partial batch
    if batch_images:
        x = np.stack(batch_images)
        x = x.astype(np.float32)
        t = torch.from_numpy(x).to(device)
        with torch.no_grad():
            out = model(t).cpu().numpy()
        out = l2_norm_rows(out).astype("float32")
        for j in range(out.shape[0]):
            shard_embeddings.append(out[j])
            shard_map_rows.append({
                "embedding_index": len(shard_map_rows),
                "image_path": str(batch_meta[j][1]),
                "dataset": batch_meta[j][0].get("dataset", ""),
                "person_id": batch_meta[j][0].get("person_id", ""),
                "sha1": batch_meta[j][0].get("sha1", "")
            })

    # flush last shard
    flush_shard(shard_idx)
    total_processed += len(shard_embeddings)
    LOGGER.info("Total new embeddings produced: %d", total_processed)

# merge all shard .npy and .csv files into single outputs for easier downstream loading and evaluation
# drop any failed embeddings (embedding_index == -1) from final outputs
def merge_shards(out_dir, out_emb_name="embeddings.npy", out_map_name="embeddings_map.csv"):
    out = Path(out_dir)
    shard_embs = sorted(out.glob("embeddings_shard_*.npy"))
    shard_maps = sorted(out.glob("embeddings_map_shard_*.csv"))

    if not shard_embs:
        LOGGER.error("No shard embeddings found in %s", out)
        return

    if len(shard_embs) != len(shard_maps):
        LOGGER.warning("shard .npy and .csv counts differ: %d .npy files vs %d .csv files",
                       len(shard_embs), len(shard_maps))

    emb_list = []
    map_frames = []

    for se, sm in zip(shard_embs, shard_maps):
        try:
            arr = np.load(se)
        except Exception:
            LOGGER.exception("Failed to load embedding shard %s; skipping", se)
            continue

        try:
            df = pd.read_csv(sm)
        except Exception:
            LOGGER.exception("Failed to read map CSV %s; skipping", sm)
            continue

        # Keep only rows that have a matching embedding (drop failures embedded as embedding_index == -1)
        df_ok = df[df.get("embedding_index", 0) != -1].reset_index(drop=True)

        if df_ok.shape[0] != arr.shape[0]:
            LOGGER.error(
                "Mismatch in shard %s : .npy rows=%d vs ok-map rows=%d. Inspect %s and %s",
                se.name, arr.shape[0], df_ok.shape[0], se, sm
            )
            raise RuntimeError(f"Mismatch in shard {se.name}: .npy rows={arr.shape[0]} vs ok-map rows={df_ok.shape[0]}. Inspect {sm}")

        emb_list.append(arr.astype("float32"))
        map_frames.append(df_ok)

    if not emb_list:
        LOGGER.error("No embeddings loaded from shards in %s", out)
        return

    all_emb = np.concatenate(emb_list, axis=0).astype("float32")
    all_map = pd.concat(map_frames, ignore_index=True)

    # recompute global embedding_index so indices correspond to rows in all_emb
    all_map = all_map.reset_index().rename(columns={"index": "embedding_index"})

    emb_path = out / out_emb_name
    map_path = out / out_map_name

    np.save(emb_path, all_emb)
    all_map.to_csv(map_path, index=False)

    LOGGER.info("Merged embeddings -> %s (rows=%d)", emb_path, all_emb.shape[0])
    LOGGER.info("Merged map -> %s (rows=%d)", map_path, all_map.shape[0])

# print status of embedding output directory
def status(out_dir):
    out = Path(out_dir)
    shards = list(out.glob("embeddings_shard_*.npy"))
    maps = list(out.glob("embeddings_map_shard_*.csv"))
    final_emb = out / "embeddings.npy"
    final_map = out / "embeddings_map.csv"
    print("out_dir:", out)
    print("shard embeddings:", len(shards))
    print("shard maps:", len(maps))
    print("final embeddings exists:", final_emb.exists())
    print("final map exists:", final_map.exists())

# ----- Main CLI -----
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--out", default="data_processed/embeddings")
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--image-size", type=int, default=160)
    run.add_argument("--shard-size", type=int, default=5000)
    run.add_argument("--sample-limit", type=int, default=None)
    run.add_argument("--use-cuda", action="store_true")
    run.add_argument("--device", default=None)

    merge = sub.add_parser("merge")
    merge.add_argument("--out", required=True)
    merge.add_argument("--emb-dim", type=int, default=512)

    st = sub.add_parser("status")
    st.add_argument("--out", required=True)

    args = p.parse_args()

    if args.cmd == "run":
        device = torch.device("cuda") if args.use_cuda and torch.cuda.is_available() else None
        embed_sharded(args.manifest, args.out, batch_size=args.batch_size, image_size=args.image_size, shard_size=args.shard_size, device=device, sample_limit=args.sample_limit)
    elif args.cmd == "merge":
        merge_shards(args.out, emb_dim=args.emb_dim)
    elif args.cmd == "status":
        status(args.out)

if __name__ == "__main__":
    main()