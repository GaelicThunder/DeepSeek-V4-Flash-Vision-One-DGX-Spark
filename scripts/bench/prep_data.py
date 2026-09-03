#!/usr/bin/env python3
"""Build the fixed item set once, so BOTH engines see byte-identical questions."""
import json, random, os, re
from datasets import load_dataset

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------- MMLU-Pro: 10-way MC, hard, sensitive to quantization ----------
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
by_cat = {}
for r in ds:
    by_cat.setdefault(r["category"], []).append(r)
rng = random.Random(20260903)
N_PER_CAT = 18
mc = []
for cat in sorted(by_cat):
    rows = sorted(by_cat[cat], key=lambda r: r["question_id"])
    rng.shuffle(rows)
    for r in rows[:N_PER_CAT]:
        if len(r["options"]) < 4:
            continue
        mc.append({"id": f"{cat[:4].lower()}-{r['question_id']}", "cat": cat,
                   "q": r["question"], "opts": list(r["options"]), "gold": int(r["answer_index"])})
json.dump(mc, open(f"{OUT}/items_mc.json", "w"))
print("MC items:", len(mc), "cats:", len(by_cat))

# ---------- MATH-500 level 5: exact-answer free generation ----------
m = load_dataset("HuggingFaceH4/MATH-500", split="test")
hard = [r for r in m if str(r.get("level")) in ("5", "Level 5")]
hard = sorted(hard, key=lambda r: r["unique_id"])[:60]
math_items = [{"id": r["unique_id"].replace("/", "_"), "cat": r.get("subject", "math"),
               "q": r["problem"], "gold": r["answer"]} for r in hard]
json.dump(math_items, open(f"{OUT}/items_math.json", "w"))
print("MATH items:", len(math_items))
