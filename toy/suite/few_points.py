"""Sum of sines with fewer training points: the training baseline (best of 9 settings) at 10 to 100 points from the 1,000-point pool.

python3 few_points.py    writes runs/few_points.jsonl and charts/sines_few_points.png at the repo root
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import make
from baseline import train, SETTINGS
from chart import OUT

HERE = os.path.dirname(os.path.abspath(__file__)); POINTS = (10, 20, 30, 50, 70, 100)
rows = []
for n in POINTS:
    d = make("regression", n / 1000); assert len(d["xtr"]) == n
    b = max((train(d, lr, wd) for lr, wd in SETTINGS), key=lambda r: r["best"])
    rows.append(dict(points=n, best=b["best"], lr=b["lr"], wd=b["wd"])); print(rows[-1], flush=True)
with open(os.path.join(HERE, "runs", "few_points.jsonl"), "w") as f: f.writelines(json.dumps(r) + "\n" for r in rows)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(POINTS, [r["best"] * 100 for r in rows], "-o", color="#1b4f91", lw=2, ms=7)
for r in rows: ax.annotate(f"{r['best'] * 100:.1f}", (r["points"], r["best"] * 100), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10, color="#444")
ax.axhline(0, color="#999", lw=1, ls=(0, (3, 3))); ax.text(100, 2, "always guess the average", ha="right", fontsize=9, color="#777")
ax.set_xlabel("training points"); ax.set_ylabel("held-out R² × 100"); ax.set_ylim(-10, 105); ax.set_xticks(POINTS)
ax.set_title("Sum of 3 sines: training score jumps between 10 and 20 points", loc="left", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sines_few_points.png"), dpi=150)
