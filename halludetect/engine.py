"""Single-pass hallucination detector + verification engine in the generation loop.
No multi-sampling: cost = 1 generation + 1 retrieval + 1 tiny classifier."""
import math, random, re, zlib
from collections import Counter
from data_source import load

STOP = set("the a an of is it in was by at to and for on as that this answer i believe think what who which how many does do are there".split())
NUMW = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
MIN_SIM = 0.30
def toks(t): return [NUMW.get(w, w) for w in re.findall(r"[a-z0-9]+", t.lower())]
def keys(t): return [w for w in toks(t) if w not in STOP]

class Retriever:                      # TF-IDF cosine (pure python, swap for a vector DB in production)
    def __init__(s, facts):
        s.facts = facts; docs = [keys(f["q"]) for f in facts]; s.cache = {}
        df = Counter(w for d in docs for w in set(d)); n = len(docs)
        s.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}; s.unk = math.log(n + 1) + 1
        s.vecs = [s._v(d) for d in docs]
    def _v(s, t):
        c = Counter(t); v = {w: c[w] * s.idf.get(w, s.unk) for w in c}
        z = math.sqrt(sum(x * x for x in v.values())) or 1
        return {w: x / z for w, x in v.items()}
    def search(s, q):
        if q in s.cache: return s.cache[q]
        qv = s._v(keys(q)); sc = [sum(x * d.get(w, 0) for w, x in qv.items()) for d in s.vecs]
        i = max(range(len(sc)), key=sc.__getitem__); s.cache[q] = (s.facts[i], sc[i]); return s.cache[q]

FACTS, ROWS, SOURCE = load()
RET = Retriever(FACTS)

def feat(q, a, lps):
    """Features: single-pass token confidence + evidence-verification signals."""
    fact, sim = RET.search(q); has = sim >= MIN_SIM
    ak = keys(a); ev = set(keys(fact["s"])) if has else set()
    sup = sum(w in ev for w in ak) / len(ak) if ak and has else 0.0           # claim support
    nm = float(has and any(w.isdigit() and w not in ev for w in ak))            # numeric mismatch
    conf = math.exp(sum(lps) / len(lps)); minc = math.exp(min(lps))
    low = sum(l < math.log(.5) for l in lps) / len(lps)
    return [conf, minc, low, sup, nm, sim if has else 0.0, conf * (1 - sup), float(not has)]
FEATS = ["mean_conf", "min_conf", "low_tok_frac", "support", "num_mismatch", "retrieval_sim", "conf_x_unsupported", "no_evidence"]

class LR:                             # tiny logistic regression, no dependencies
    def _z(s, x): return [(v - m) / d for v, m, d in zip(x, s.mu, s.sd)]
    def _p(s, z): return 1 / (1 + math.exp(-max(-30, min(30, s.b + sum(w * v for w, v in zip(s.w, z))))))
    def predict(s, x): return s._p(s._z(x))
    def fit(s, X, y, it=1200, lr=.4, l2=.01):
        d, n = len(X[0]), len(X); s.mu = [sum(r[j] for r in X) / n for j in range(d)]
        s.sd = [(sum((r[j] - s.mu[j]) ** 2 for r in X) / n) ** .5 or 1 for j in range(d)]
        s.w, s.b = [0.0] * d, 0.0; Z = [s._z(r) for r in X]
        for _ in range(it):
            gw, gb = [l2 * w for w in s.w], 0.0
            for z, t in zip(Z, y):
                e = s._p(z) - t
                for j in range(d): gw[j] += e * z[j] / n
                gb += e / n
            s.w = [w - lr * g for w, g in zip(s.w, gw)]; s.b -= lr * gb
        return s

def conformal_t(hall_scores, alpha):
    """Split-conformal: flag if score >= t. For an exchangeable new hallucination,
    P(score < t) <= alpha  =>  miss-rate (incl. confidently wrong) <= alpha."""
    s = sorted(hall_scores); k = math.floor(alpha * (len(s) + 1))
    return 0.0 if k < 1 else s[k - 1]

def auroc(sc, y):
    P = [s for s, t in zip(sc, y) if t]; N = [s for s, t in zip(sc, y) if not t]
    return sum((p > n) + .5 * (p == n) for p in P for n in N) / (len(P) * len(N))

class MockLLM:                        # offline stand-in; replace with any API that returns token logprobs
    def generate(s, q, evidence=None, attempt=0):
        r = random.Random(zlib.crc32(f"{q}|{attempt}".encode())); fact, sim = RET.search(q)
        if evidence: ans, conf = evidence["a"], .95
        elif sim >= MIN_SIM:
            if r.random() < .6 and fact["w"]: ans, conf = r.choice(fact["w"]), r.choice([.93, .9, .55])   # hallucinates, often confidently
            else: ans, conf = fact["a"], .92
        else: ans, conf = "I believe it is 1987.", .9
        lps = [math.log(min(.99, max(.05, conf + r.uniform(-.04, .04)))) for _ in range(len(toks(ans)) + 2)]
        return ans, lps

class System:
    def __init__(s, alpha=.1, seed=7):
        s.alpha, s.llm, s.source = alpha, MockLLM(), SOURCE; r = random.Random(seed)
        qs = sorted({x["q"] for x in ROWS}); r.shuffle(qs); n = len(qs)   # split by question: no leakage across splits
        grp = {q: (0 if i < n // 2 else 1 if i < 3 * n // 4 else 2) for i, q in enumerate(qs)}
        tr, ca, te = ([x for x in ROWS if grp[x["q"]] == g] for g in range(3))
        F = lambda R: [feat(x["q"], x["a"], x["lps"]) for x in R]; Y = lambda R: [x["label"] for x in R]
        Ftr, Fca = F(tr), F(ca)
        it = 1200 if len(tr) < 600 else 300; s.fused = LR().fit(Ftr, Y(tr), it=it); s.base = LR().fit([f[:3] for f in Ftr], Y(tr), it=it)
        s.t = conformal_t([s.fused.predict(f) for f, x in zip(Fca, ca) if x["label"]], alpha)
        s.tb = conformal_t([s.base.predict(f[:3]) for f, x in zip(Fca, ca) if x["label"]], alpha)
        s.metrics = s._eval(te); s.sizes = dict(train=len(tr), calibration=len(ca), test=len(te))
    def _eval(s, te):
        Fx = [feat(x["q"], x["a"], x["lps"]) for x in te]; y = [x["label"] for x in te]
        H = [i for i, v in enumerate(y) if v]; C = [i for i, v in enumerate(y) if not v]
        W = [i for i in H if Fx[i][0] >= .8]; out = {}
        for name, sc, t in (("Uncertainty-only baseline (single-pass confidence)", [s.base.predict(f[:3]) for f in Fx], s.tb),
                            ("Proposed: verification engine + conformal fusion", [s.fused.predict(f) for f in Fx], s.t)):
            fl = [v >= t for v in sc]
            out[name] = dict(auroc=round(auroc(sc, y), 3), recall=round(sum(fl[i] for i in H) / len(H), 3),
                             false_alarm=round(sum(fl[i] for i in C) / len(C), 3),
                             recall_confident_wrong=round(sum(fl[i] for i in W) / max(1, len(W)), 3))
        return dict(methods=out, meta=dict(alpha=s.alpha, test_size=len(te), hallucinations=len(H), confidently_wrong=len(W)))

    def check(s, q, a, lps=None, conf=None):
        if not lps: lps = [math.log(min(.99, max(.01, conf or .9)))] * (len(toks(a)) + 2)
        x = feat(q, a, lps); p = s.fused.predict(x); fact, sim = RET.search(q); has = sim >= MIN_SIM
        verdict = "unverifiable" if not has else ("supported" if x[3] >= .999 and not x[4] else "contradicted")
        return dict(p_hallucination=round(p, 3), flagged=p >= s.t, threshold=round(s.t, 3), verdict=verdict,
                    evidence=fact["s"] if has else None, retrieval_sim=round(sim, 2), features=dict(zip(FEATS, [round(v, 3) for v in x])),
                    guarantee=f"Miss-rate on hallucinations <= {s.alpha:.0%} (split-conformal, exchangeable data)")

    def guarded(s, q, max_iters=2):
        """Generate -> verify -> (if flagged) regenerate grounded in evidence -> accept or abstain."""
        trace, ev = [], None
        for i in range(max_iters + 1):
            ans, lps = s.llm.generate(q, ev, i); v = s.check(q, ans, lps)
            trace.append(dict(attempt=i, answer=ans, grounded=ev is not None, **v))
            if not v["flagged"]: return dict(final=ans, status="accepted", trace=trace)
            fact, sim = RET.search(q)
            if sim < MIN_SIM: break
            ev = fact
        return dict(final="I can't verify this answer reliably.", status="abstained", trace=trace)

if __name__ == "__main__":
    S = System(); print("data:", S.source, "| facts:", len(FACTS), S.sizes, "threshold", round(S.t, 3))
    for k, v in S.metrics["methods"].items(): print(k, v)
    print(S.metrics["meta"])
    for q in [FACTS[0]["q"], "Who won the 2031 FIFA World Cup?"]:
        g = S.guarded(q); print(q, "->", g["status"], "|", g["final"], "| attempts:", len(g["trace"]))
