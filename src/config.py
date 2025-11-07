import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")

# Base folders
FEATURE_STORE_DIR = Path(os.getenv("FEATURE_STORE_DIR", ROOT / "data" / "features"))
MODEL_SAVE_DIR = Path(os.getenv("MODEL_SAVE_DIR", ROOT / "data" / "models"))
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", ROOT / "data" / "raw"))
REGISTRY_FILE = Path(os.getenv("REGISTRY_FILE", MODEL_SAVE_DIR / "registry.json"))

for d in [FEATURE_STORE_DIR, RAW_DATA_DIR, MODEL_SAVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
