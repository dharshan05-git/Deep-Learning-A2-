import os
from kb_dataset import fact_dicts, build_dataset, ood_rows
HERE = os.path.dirname(os.path.abspath(__file__)); KDIR = os.path.join(HERE, "data", "kaggle")

def load():
    """Kaggle dataset if present in data/kaggle/, otherwise the built-in synthetic benchmark."""
    if os.path.isdir(KDIR) and any(f.lower().endswith((".csv", ".json", ".jsonl")) and f != "columns.json" for f in os.listdir(KDIR)):
        from kaggle_loader import load_kaggle
        facts, rows = load_kaggle(KDIR)
        if rows and facts: return facts, rows + ood_rows(), "kaggle (AI Hallucination Dataset)"
        print("[data_source] Kaggle files found but nothing usable; falling back to synthetic data")
    return fact_dicts(), build_dataset(7), "synthetic"
