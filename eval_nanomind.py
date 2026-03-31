# ============================================================
# eval_nanomind.py — NanoMind Quality Eval Harness
# Tests factual recall on 50 hand-written Q&A pairs
# Run: python eval_nanomind.py
# Run against remote: python eval_nanomind.py --api https://your-space.hf.space
# ============================================================

import argparse
import json
import time
import requests

# ── 50 Eval Questions ────────────────────────────────────────
# Format: {"q": question, "must_contain": [any one of these keywords]}
# Pass condition: response contains at least ONE keyword (case-insensitive)

EVAL_SET = [
    # Geography (10)
    {"q": "What is the capital of France?",           "must_contain": ["paris"]},
    {"q": "What is the capital of Japan?",            "must_contain": ["tokyo"]},
    {"q": "What is the capital of Germany?",          "must_contain": ["berlin"]},
    {"q": "What is the largest ocean on Earth?",      "must_contain": ["pacific"]},
    {"q": "What continent is Brazil in?",             "must_contain": ["south america"]},
    {"q": "What is the longest river in the world?",  "must_contain": ["nile", "amazon"]},
    {"q": "What country has the most population?",    "must_contain": ["china", "india"]},
    {"q": "What is the capital of Australia?",        "must_contain": ["canberra"]},
    {"q": "What ocean is between USA and Europe?",    "must_contain": ["atlantic"]},
    {"q": "What is the smallest country in the world?", "must_contain": ["vatican"]},

    # Math & Numbers (10)
    {"q": "What is 2 + 2?",                           "must_contain": ["4", "four"]},
    {"q": "What is 10 times 10?",                     "must_contain": ["100", "hundred"]},
    {"q": "How many days are in a week?",             "must_contain": ["7", "seven"]},
    {"q": "How many hours are in a day?",             "must_contain": ["24", "twenty"]},
    {"q": "How many months are in a year?",           "must_contain": ["12", "twelve"]},
    {"q": "What is the square root of 16?",           "must_contain": ["4", "four"]},
    {"q": "How many sides does a triangle have?",     "must_contain": ["3", "three"]},
    {"q": "What is 100 divided by 4?",                "must_contain": ["25", "twenty"]},
    {"q": "How many degrees in a right angle?",       "must_contain": ["90", "ninety"]},
    {"q": "What is 5 factorial?",                     "must_contain": ["120"]},

    # Science (10)
    {"q": "What is H2O?",                             "must_contain": ["water"]},
    {"q": "What planet is closest to the Sun?",       "must_contain": ["mercury"]},
    {"q": "What gas do plants absorb?",               "must_contain": ["carbon", "co2"]},
    {"q": "What is the boiling point of water in Celsius?", "must_contain": ["100"]},
    {"q": "What force keeps us on the ground?",       "must_contain": ["gravity"]},
    {"q": "What is the chemical symbol for gold?",    "must_contain": ["au"]},
    {"q": "How many planets are in our solar system?","must_contain": ["8", "eight"]},
    {"q": "What is the speed of light approximately?","must_contain": ["300", "light"]},
    {"q": "What organ pumps blood in the human body?","must_contain": ["heart"]},
    {"q": "What is DNA?",                             "must_contain": ["deoxyribonucleic", "genetic", "gene"]},

    # Technology (10)
    {"q": "What does CPU stand for?",                 "must_contain": ["central", "processing"]},
    {"q": "What does GPU stand for?",                 "must_contain": ["graphics", "processing"]},
    {"q": "What language is Python?",                 "must_contain": ["programming"]},
    {"q": "What does HTML stand for?",                "must_contain": ["hypertext"]},
    {"q": "What is machine learning?",                "must_contain": ["data", "learn", "model"]},
    {"q": "What is a neural network?",                "must_contain": ["neuron", "layer", "brain", "network"]},
    {"q": "What does RAM stand for?",                 "must_contain": ["random", "memory"]},
    {"q": "What is an API?",                          "must_contain": ["interface", "application"]},
    {"q": "What does SQL stand for?",                 "must_contain": ["structured", "query"]},
    {"q": "What is open source software?",            "must_contain": ["source", "code", "free"]},

    # General Knowledge (10)
    {"q": "Who wrote Romeo and Juliet?",              "must_contain": ["shakespeare"]},
    {"q": "What year did World War 2 end?",           "must_contain": ["1945"]},
    {"q": "How many colors are in a rainbow?",        "must_contain": ["7", "seven"]},
    {"q": "What is the fastest land animal?",         "must_contain": ["cheetah"]},
    {"q": "What language is spoken in Brazil?",       "must_contain": ["portuguese"]},
    {"q": "Who painted the Mona Lisa?",               "must_contain": ["da vinci", "leonardo"]},
    {"q": "What is the currency of Japan?",           "must_contain": ["yen"]},
    {"q": "How many strings does a guitar have?",     "must_contain": ["6", "six"]},
    {"q": "What is the national language of China?",  "must_contain": ["mandarin", "chinese"]},
    {"q": "What is photosynthesis?",                  "must_contain": ["light", "plant", "energy", "sun"]},
]

# ── Runner ────────────────────────────────────────────────────

def run_single(api: str, question: str, temperature: float = 0.1) -> str:
    """Send one question, collect full SSE response, return assistant reply."""
    try:
        resp = requests.post(
            f"{api}/chat",
            json={
                "message":        question,
                "max_new_tokens": 80,
                "temperature":    temperature,
                "top_k":          40,
            },
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()

        full_response = ""
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                if chunk.get("type") == "done":
                    full_response = chunk.get("full_response", "")
            except json.JSONDecodeError:
                continue

        return full_response.strip()

    except requests.exceptions.ConnectionError:
        return "__CONNECTION_ERROR__"
    except Exception as e:
        return f"__ERROR: {e}__"


def run_eval(api: str = "http://localhost:7860", verbose: bool = True):
    print("=" * 60)
    print(f"NanoMind Eval Harness — {len(EVAL_SET)} questions")
    print(f"API : {api}")
    print(f"Temp: 0.1 (low for factual)")
    print("=" * 60)

    # Check server health first
    try:
        h = requests.get(f"{api}/health", timeout=10)
        health = h.json()
        engines_ready = health.get("engines_ready", 0)
        if engines_ready == 0:
            print("⚠️  Warning: No engines ready — server may still be loading")
        else:
            print(f"✅ Server healthy — {engines_ready} engine(s) ready")
    except Exception:
        print("⚠️  Could not reach /health — proceeding anyway")

    print()

    passed   = 0
    failed   = 0
    errors   = 0
    results  = []
    t_start  = time.time()

    for i, item in enumerate(EVAL_SET):
        q        = item["q"]
        keywords = item["must_contain"]

        response = run_single(api, q)

        if response.startswith("__"):
            status = "⚠"
            errors += 1
        else:
            ok = any(kw.lower() in response.lower() for kw in keywords)
            if ok:
                status = "✅"
                passed += 1
            else:
                status = "❌"
                failed += 1

        results.append({
            "q":        q,
            "response": response,
            "keywords": keywords,
            "status":   status,
        })

        if verbose:
            print(f"{status} [{i+1:02d}/50] {q}")
            if status != "✅":
                print(f"        Expected: {keywords}")
                print(f"        Got     : {response[:100]}")

    elapsed = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────
    total    = len(EVAL_SET)
    score    = passed / total * 100
    grade    = ("A" if score >= 80 else
                "B" if score >= 65 else
                "C" if score >= 50 else
                "D")

    print()
    print("=" * 60)
    print(f"RESULTS")
    print("=" * 60)
    print(f"  Passed  : {passed}/{total}")
    print(f"  Failed  : {failed}/{total}")
    print(f"  Errors  : {errors}/{total}")
    print(f"  Score   : {score:.1f}%  (Grade: {grade})")
    print(f"  Time    : {elapsed:.1f}s  ({elapsed/total:.1f}s per question)")
    print("=" * 60)
    print()
    print("Note: This eval tests factual recall only.")
    print("152M models are not expected to reason or generalize.")
    print("Score >50% at this scale is considered functional.")

    return score


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoMind Eval Harness")
    parser.add_argument(
        "--api",
        default="http://localhost:7860",
        help="API base URL (default: http://localhost:7860)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show summary, not per-question results"
    )
    args = parser.parse_args()

    run_eval(api=args.api, verbose=not args.quiet)