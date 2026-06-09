"""
evaluate.py - run the agent against eval/questions.json and print metrics.

Run (after ingest.py):
    python evaluate.py

Metrics:
- Answer correctness  : LLM-judge checks each answer is on-topic and grounded
                        in the retrieved sources (not fabricated).
- Out-of-scope refusal : fraction of unrelated questions the bot correctly declines.
- Latency             : average and p95 response time.
"""

import os
import json
import time
import statistics

from dotenv import load_dotenv
load_dotenv()

import rag

QFILE = os.getenv("EVAL_FILE", "eval/questions.json")

JUDGE_SYSTEM = """You grade a support-bot answer.
Given QUESTION, ANSWER and SOURCES, decide if the answer is correct, on-topic,
and grounded in the sources (not fabricated).
Reply with strict JSON only: {"score": 0 or 1, "reason": "short reason"}.
Give score 1 only if the answer is helpful and consistent with the sources."""

REFUSAL_HINTS = [
    "don't have", "do not have", "not have that", "contact support",
    "couldn't find", "no information", "ma'lumot", "yo'q", "не располагаю",
    "обратитесь",
]


def judge(question, answer_text, sources_text):
    try:
        r = rag.get_client().chat.completions.create(
            model=rag.LLM_MODEL, temperature=0, max_tokens=120,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content":
                    f"QUESTION: {question}\n\nANSWER: {answer_text}\n\n"
                    f"SOURCES: {sources_text[:3000]}"},
            ],
        )
        out = (r.choices[0].message.content or "").strip().strip("`")
        out = out[out.find("{"):out.rfind("}") + 1]
        return json.loads(out)
    except Exception as e:
        return {"score": 0, "reason": f"judge error: {e}"}


def declined(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in REFUSAL_HINTS)


def p95(values):
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(0.95 * len(s)) - 1))
    return s[max(idx, 0)]


def main():
    with open(QFILE, encoding="utf-8") as f:
        data = json.load(f)
    answerable = data.get("answerable", [])
    oos = data.get("out_of_scope", [])

    latencies, correct = [], 0

    print("=== Answerable questions ===")
    for item in answerable:
        q = item["question"]
        t0 = time.time()
        ans, srcs = rag.answer(q, [])
        dt = time.time() - t0
        latencies.append(dt)
        verdict = judge(q, ans, " ".join(srcs))
        correct += verdict["score"]
        mark = "OK" if verdict["score"] else "X "
        print(f"[{mark}] ({dt:4.1f}s) {q}")
        print(f"        -> {ans[:160]}")

    refused = 0
    print("\n=== Out-of-scope questions (should decline) ===")
    for item in oos:
        q = item["question"]
        t0 = time.time()
        ans, _ = rag.answer(q, [])
        dt = time.time() - t0
        latencies.append(dt)
        ok = declined(ans)
        refused += ok
        mark = "OK" if ok else "X "
        print(f"[{mark}] {q}")
        print(f"        -> {ans[:160]}")

    print("\n=== Summary ===")
    if answerable:
        print(f"Answer correctness  : {correct}/{len(answerable)} = "
              f"{correct / len(answerable):.0%}")
    if oos:
        print(f"Out-of-scope refusal: {refused}/{len(oos)} = "
              f"{refused / len(oos):.0%}")
    if latencies:
        print(f"Latency avg / p95   : {statistics.mean(latencies):.1f}s / "
              f"{p95(latencies):.1f}s")


if __name__ == "__main__":
    main()
