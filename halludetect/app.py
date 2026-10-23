import os
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from engine import System, FACTS, ROWS, SOURCE

app = FastAPI(title="HalluGuard: real-time hallucination detection")
S = System(); HERE = os.path.dirname(__file__)

class Check(BaseModel):
    question: str; answer: str
    confidence: Optional[float] = None
    token_logprobs: Optional[List[float]] = None
class Ask(BaseModel): question: str

@app.get("/")
def home(): return FileResponse(os.path.join(HERE, "static", "index.html"))
@app.post("/api/check")
def check(b: Check): return S.check(b.question, b.answer, b.token_logprobs, b.confidence)
@app.post("/api/guarded")
def guarded(b: Ask): return S.guarded(b.question)
@app.get("/api/metrics")
def metrics(): return {**S.metrics, "split": S.sizes, "source": SOURCE}
@app.get("/api/questions")
def questions(): return [f["q"] for f in FACTS[:150]] + ["Who won the 2031 FIFA World Cup?"]
@app.get("/api/dataset")
def dataset(limit: int = 50):
    return [dict(question=x["q"], answer=x["a"], label=x["label"], kind=x["kind"]) for x in ROWS[:limit]]
