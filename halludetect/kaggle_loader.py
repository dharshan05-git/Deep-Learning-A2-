"""Loads the Kaggle 'AI Hallucination Dataset' (HaluEval + TruthfulQA + HalluRAG) from data/kaggle/.
Column names are auto-detected; override with data/kaggle/columns.json, e.g. {"q":"question","good":"right_answer","bad":"hallucinated_answer"}.
Supports two layouts:  (a) paired: correct + hallucinated answer columns   (b) labelled: one answer column + a 0/1 label column.
Run `python kaggle_loader.py` to inspect what was detected."""
import csv, glob, json, os, random, sys
from kb_dataset import synth_lps
csv.field_size_limit(10 ** 8)
ALIASES = dict(
    q=["question", "query", "prompt", "user_query", "instruction", "input"],
    k=["knowledge", "context", "passage", "document", "evidence", "source", "contexts", "reference_text"],
    good=["right_answer", "correct_answer", "best_answer", "ground_truth", "gold_answer", "truth", "reference", "reference_answer"],
    bad=["hallucinated_answer", "incorrect_answer", "wrong_answer", "hallucinated_response", "hallucinated_output"],
    resp=["response", "answer", "output", "generated_answer", "model_answer", "llm_response", "completion", "text"],
    lab=["label", "is_hallucination", "hallucination_label", "hallucination", "hallucinated", "is_hallucinated", "hallucination_flag",
         "truthful", "is_truthful", "factual", "is_factual", "correct", "is_correct"])
INVERT = {"truthful", "is_truthful", "factual", "is_factual", "correct", "is_correct"}   # 1 means NOT hallucinated
norm = lambda c: str(c).strip().lower().replace(" ", "_")

def detect(cols, override=None):
    n = {norm(c): c for c in cols}; m = {}
    for k, al in ALIASES.items():
        if override and k in override: m[k] = override[k]; continue
        for a in al:
            if a in n: m[k] = n[a]; break
    return m

def lab01(v):
    s = str(v).strip().lower()
    if s in ("1", "1.0", "true", "yes", "y", "hallucination", "hallucinated", "hallucinatory"): return 1
    if s in ("0", "0.0", "false", "no", "n", "faithful", "factual", "truthful", "correct", "non-hallucinated", "not hallucinated"): return 0
    if "halluc" in s: return 0 if s.startswith(("no", "non", "not")) else 1
    return None

def read_records(path):
    p = path.lower()
    if p.endswith(".csv"):
        with open(path, newline="", encoding="utf-8", errors="replace") as f: return list(csv.DictReader(f))
    with open(path, encoding="utf-8", errors="replace") as f:
        if p.endswith(".jsonl"): return [json.loads(l) for l in f if l.strip()]
        d = json.load(f)
    if isinstance(d, dict): d = next((v for v in d.values() if isinstance(v, list)), [d])
    return d

def inspect(kdir):
    ov = json.load(open(os.path.join(kdir, "columns.json"))) if os.path.exists(os.path.join(kdir, "columns.json")) else None
    for p in sorted(glob.glob(os.path.join(kdir, "*"))):
        if p.lower().endswith((".csv", ".json", ".jsonl")) and not p.endswith("columns.json"):
            rec = read_records(p); print(os.path.basename(p), len(rec), "records"); print("  columns:", list(rec[0].keys()) if rec else [])
            print("  detected:", detect(list(rec[0].keys()), ov) if rec else {})

def load_kaggle(kdir, max_q=int(os.environ.get("HG_MAX_QUESTIONS", 1000)), seed=7):
    ov = json.load(open(os.path.join(kdir, "columns.json"))) if os.path.exists(os.path.join(kdir, "columns.json")) else None
    r = random.Random(seed); facts, rows = {}, []
    T = lambda v, n: " ".join(str(v).split())[:n]
    for path in sorted(glob.glob(os.path.join(kdir, "*"))):
        if not path.lower().endswith((".csv", ".json", ".jsonl")) or path.endswith("columns.json"): continue
        rec = read_records(path)
        if not rec: continue
        m = detect(list(rec[0].keys()), ov)
        if "q" not in m or not ("good" in m or ("resp" in m and "lab" in m)):
            print(f"[kaggle_loader] skipped {os.path.basename(path)}: cannot map columns {list(rec[0].keys())}; add columns.json"); continue
        for x in rec:
            q = T(x.get(m["q"], ""), 300)
            if not q: continue
            f = facts.setdefault(q.lower(), dict(q=q, s="", a="", w=[])); ev = T(x.get(m["k"], ""), 1500) if "k" in m else ""
            def add(a, lab):
                a = T(a, 300)
                if a: rows.append(dict(q=q, a=a, label=lab, kind="kaggle_hallucinated" if lab else "kaggle_correct", lps=synth_lps(r, lab, min(len(a.split()), 30) + 2)))
            if "good" in m and x.get(m["good"]):
                g = T(x[m["good"]], 300); f["a"] = f["a"] or g; f["s"] = f["s"] or ev or g; add(g, 0)
                if "bad" in m and x.get(m["bad"]): add(x[m["bad"]], 1); f["w"].append(T(x[m["bad"]], 300))
            elif "resp" in m and "lab" in m:
                lab = lab01(x.get(m["lab"]))
                if lab is None: continue
                if m["lab"] in INVERT: lab = 1 - lab
                a = T(x.get(m["resp"], ""), 300)
                if ev: f["s"] = f["s"] or ev
                if lab == 0: f["a"] = f["a"] or a; f["s"] = f["s"] or a
                else: f["w"].append(a)
                add(a, lab)
    qs = sorted({x["q"] for x in rows}); r.shuffle(qs); keep = set(qs[:max_q])
    rows = [x for x in rows if x["q"] in keep]
    fl = [f for f in facts.values() if f["q"] in keep and f["s"]]
    for f in fl: f["a"] = f["a"] or f["s"][:80]
    return fl, rows

if __name__ == "__main__":
    inspect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kaggle"))
