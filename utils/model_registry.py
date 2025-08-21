import os, joblib, glob

DEFAULT_MODEL_DIR = os.path.join("models", "saved")
os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)

def save_model(model, name: str, model_dir: str = DEFAULT_MODEL_DIR) -> str:
    path = os.path.join(model_dir, f"{name}.joblib")
    joblib.dump(model, path)
    return path

def load_latest_model(model_dir: str = DEFAULT_MODEL_DIR):
    files = glob.glob(os.path.join(model_dir, "*.joblib"))
    if not files:
        raise FileNotFoundError("No models found")
    latest = max(files, key=os.path.getmtime)
    return joblib.load(latest), latest
