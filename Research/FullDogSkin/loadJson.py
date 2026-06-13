import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = Path(os.getenv("DISEASE_DATA_PATH", BASE_DIR / "soluction.json"))

def load_disease_data():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")

    with JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)
