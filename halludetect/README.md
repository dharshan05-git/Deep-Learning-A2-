# HalluGuard - single-pass hallucination detection with a verification engine in the loop

## Quick start
    pip install -r requirements.txt
    python engine.py          # CLI evaluation (no server needed)
    uvicorn app:app --reload  # UI + API at http://127.0.0.1:8000

## Using the Kaggle dataset (piyaldattapiyal/ai-hallucination-dataset)
1. Download it (needs a free Kaggle account), either from the dataset page or:
       pip install kaggle
       kaggle datasets download -d piyaldattapiyal/ai-hallucination-dataset -p data/kaggle --unzip
2. Make sure the .csv/.json files sit directly in `data/kaggle/`.
3. Check the column detection:   python kaggle_loader.py
   If a column is not detected, create data/kaggle/columns.json, e.g.
       {"q": "question", "k": "knowledge", "good": "right_answer", "bad": "hallucinated_answer"}
   (labelled layouts: {"q": "...", "resp": "...", "lab": "..."})
4. Run `python engine.py` or start the server. With no files in data/kaggle/ it falls back to the built-in synthetic benchmark.
Env var HG_MAX_QUESTIONS (default 1000) caps how many questions are used so the pure-Python trainer stays fast.

## How the dataset is used
- Questions + knowledge/right answers build the retrieval knowledge base (evidence for the verification engine).
- Right answers -> label 0, hallucinated answers -> label 1 (or the 0/1 label column).
- Train/calibration/test split is by question (50/25/25), so no question appears in two splits.
- The Kaggle file has no token log-probabilities, so single-pass confidence is simulated (about half of hallucinations are
  "confidently wrong"). Swap in real logprobs by replacing MockLLM / synth_lps with a model API that returns them.

Files: engine.py (detector, retriever, conformal calibration, loop), app.py (FastAPI), kaggle_loader.py + data_source.py (data),
kb_dataset.py (synthetic fallback), static/index.html (UI), data/ (synthetic dataset.csv, kb.json, kaggle/ drop folder).
