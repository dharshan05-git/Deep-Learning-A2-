"""Knowledge base + synthetic benchmark generator.
Each fact: (question, evidence sentence, canonical answer, [plausible wrong answers])."""
import csv, json, math, os, random

FACTS = [
 ("What is the capital of Australia?", "Canberra is the capital of Australia.", "Canberra", ["Sydney", "Melbourne"]),
 ("Who wrote Pride and Prejudice?", "Jane Austen wrote Pride and Prejudice in 1813.", "Jane Austen", ["Charlotte Bronte", "Emily Dickinson"]),
 ("At what temperature does water boil at sea level?", "Water boils at 100 degrees Celsius at sea level.", "100 degrees Celsius", ["90 degrees Celsius", "110 degrees Celsius"]),
 ("Which is the largest planet in the Solar System?", "Jupiter is the largest planet in the Solar System.", "Jupiter", ["Saturn", "Neptune"]),
 ("In which year did humans first land on the Moon?", "Apollo 11 landed humans on the Moon in 1969.", "1969", ["1972", "1965"]),
 ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au.", "Au", ["Ag", "Gd"]),
 ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa.", "Leonardo da Vinci", ["Michelangelo", "Raphael"]),
 ("What is the speed of light in vacuum?", "The speed of light in vacuum is about 299792 kilometres per second.", "299792 kilometres per second", ["186000 kilometres per second", "450000 kilometres per second"]),
 ("What is the highest mountain above sea level?", "Mount Everest is the highest mountain above sea level.", "Mount Everest", ["K2", "Kangchenjunga"]),
 ("Which gas do plants absorb during photosynthesis?", "Plants absorb carbon dioxide during photosynthesis.", "carbon dioxide", ["oxygen", "nitrogen"]),
 ("Who developed the theory of general relativity?", "Albert Einstein developed the theory of general relativity.", "Albert Einstein", ["Isaac Newton", "Niels Bohr"]),
 ("What is the currency of Japan?", "The currency of Japan is the yen.", "yen", ["won", "yuan"]),
 ("How many continents are there?", "There are 7 continents on Earth.", "7", ["5", "9"]),
 ("Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928.", "Alexander Fleming", ["Louis Pasteur", "Marie Curie"]),
 ("What is the capital of Canada?", "Ottawa is the capital of Canada.", "Ottawa", ["Toronto", "Vancouver"]),
 ("What is the smallest prime number?", "The smallest prime number is 2.", "2", ["1", "3"]),
 ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet.", "William Shakespeare", ["Christopher Marlowe", "Charles Dickens"]),
 ("What is the hardest natural substance?", "Diamond is the hardest natural substance.", "Diamond", ["Quartz", "Graphite"]),
 ("In which year was Microsoft founded?", "Microsoft was founded in 1975 by Bill Gates and Paul Allen.", "1975", ["1985", "1969"]),
 ("Which planet is closest to the Sun?", "Mercury is the planet closest to the Sun.", "Mercury", ["Venus", "Mars"]),
 ("Who invented the telephone?", "Alexander Graham Bell invented the telephone.", "Alexander Graham Bell", ["Thomas Edison", "Nikola Tesla"]),
 ("What is the capital of India?", "New Delhi is the capital of India.", "New Delhi", ["Mumbai", "Kolkata"]),
 ("What is the chemical formula of water?", "The chemical formula of water is H2O.", "H2O", ["H2O2", "CO2"]),
 ("Which organ pumps blood through the human body?", "The heart pumps blood through the human body.", "heart", ["liver", "lungs"]),
]
# Questions NOT covered by the KB -> any confident answer is fabricated / unverifiable
OOD = [("Who won the 2031 FIFA World Cup?", "Brazil"), ("What is the population of the city of Zorbania?", "4.2 million"),
       ("Which company acquired Nexora Labs in 2029?", "Google"), ("Who is the current mayor of Valdoria?", "Maria Lopez"),
       ("What was the closing price of ACME stock on 3 March 2030?", "312 dollars"), ("Which prize did Dr. Helen Voss win in 2028?", "Nobel Prize")]

def fact_dicts():
    return [dict(q=q, s=s, a=a, w=w) for q, s, a, w in FACTS]

def _lps(r, conf, n, dip=False):
    out = [math.log(min(.99, max(.03, conf + r.gauss(0, .035)))) for _ in range(n)]
    if dip: out[r.randrange(n)] = math.log(r.uniform(.15, .4))
    return out

def build_dataset(seed=7):
    """Synthetic token log-probs (no LLM needed offline). KEY property: ~55% of hallucinations are
    *confidently wrong* (same confidence range as correct answers) so uncertainty-only detectors fail."""
    r = random.Random(seed); rows = []
    def add(q, a, label, kind, conf, dip=False):
        n = len(a.split()) + 2
        rows.append(dict(q=q, a=a, label=label, kind=kind, lps=_lps(r, conf, n, dip)))
    for f in fact_dicts():
        for t in ("{a}", "It is {a}.", "The answer is {a}."):
            add(f["q"], t.format(a=f["a"]), 0, "correct", r.uniform(.8, .98) if r.random() > .15 else r.uniform(.5, .8))
        for w in f["w"]:
            for t in ("{a}", "The answer is {a}."):
                if r.random() < .55: add(f["q"], t.format(a=w), 1, "confidently_wrong", r.uniform(.82, .97))
                else: add(f["q"], t.format(a=w), 1, "uncertain_wrong", r.uniform(.3, .65), dip=True)
    for q, a in OOD:
        add(q, f"The answer is {a}.", 1, "fabricated", r.uniform(.8, .96))
        add(q, a, 1, "fabricated", r.uniform(.85, .97))
    return rows


def synth_lps(r, label, n):
    """Token log-probs for datasets that ship none (e.g. the Kaggle file): same assumption as above."""
    if label == 0: return _lps(r, r.uniform(.8, .98) if r.random() > .15 else r.uniform(.5, .8), n)
    if r.random() < .55: return _lps(r, r.uniform(.82, .97), n)
    return _lps(r, r.uniform(.3, .65), n, dip=True)

def ood_rows(seed=11):
    r = random.Random(seed); rows = []
    for q, a in OOD:
        for t in ("The answer is {a}.", "{a}"):
            x = t.format(a=a); rows.append(dict(q=q, a=x, label=1, kind="fabricated", lps=_lps(r, r.uniform(.8, .97), len(x.split()) + 2)))
    return rows

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    rows = build_dataset()
    with open("data/dataset.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "question", "answer", "label", "kind", "token_logprobs"])
        for i, x in enumerate(rows): w.writerow([i, x["q"], x["a"], x["label"], x["kind"], json.dumps([round(l, 4) for l in x["lps"]])])
    json.dump(fact_dicts(), open("data/kb.json", "w"), indent=1)
    print(f"wrote {len(rows)} samples ({sum(x['label'] for x in rows)} hallucinated) + {len(FACTS)} KB facts")
