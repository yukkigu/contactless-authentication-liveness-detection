"""
Config.py

Usage: File paths and other configuration variables.
  - Define paths to datasets, models, and output directories.
  - Set parameters for data processing and model evaluation.
  - Centralize configuration for easy modification and maintenance.

"""

from pathlib import Path

# ----- Project root -----
PROJECT_ROOT = Path.cwd().parent.resolve()

# ----- Raw + processed data paths -----
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"

# ----- Dataset -----
DS_VGG = "vggface2"

RAW_VGG = DATA_RAW / DS_VGG                 # data_raw/vggface2
PROCESSED_VGG = DATA_PROCESSED / DS_VGG     # data_processed/vggface2

# ----- Directory structure -----
VGG_MANIFEST_DIR = PROCESSED_VGG / "manifests"      # data_processed/vggface2/manifests
VGG_ALIGNED_DIR = PROCESSED_VGG / "aligned"         # data_processed/vggface2/aligned
VGG_EMBEDDINGS_DIR = PROCESSED_VGG / "embeddings"   # data_processed/vggface2/embeddings

# ----- Manifest paths -----
MANIFEST_BASIC = VGG_MANIFEST_DIR / "manifest_basic.csv"
MANIFEST_ENROLL = VGG_MANIFEST_DIR / "manifest_enroll.csv"
MANIFEST_PROBE = VGG_MANIFEST_DIR / "manifest_probe.csv"
MANIFEST_TRAIN = VGG_MANIFEST_DIR / "manifest_train.csv"
MANIFEST_VAL = VGG_MANIFEST_DIR / "manifest_val.csv"

# ----- Embedding settings
EMBEDDING_SIZE = 512            # Size of the embedding vectors
EMBEDDING_BATCH_SIZE = 128      # Batch size for embedding extraction
EMBEDDING_SHARD_SIZE = 2000     # Number of embeddings per shard file

# ----- Split settings -----
GLOBAL_SEED = 42                # Global random seed for reproducibility
ENROLL_FRAC = 0.8               # Fraction of data for enrollment
PROBE_FRAC = 0.2                # Fraction of data for probe

