"""Chart of Opus-written weights against the training baseline at 10% of examples. python3 opus_chart.py writes charts/opus_10.png at the repo root."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chart import NAMES, OUT, ORDER, best

HERE = os.path.dirname(os.path.abspath(__file__))
rows = {r["task"]: r for r in map(json.loads, open(os.path.join(HERE, "runs", "opus", "results.jsonl"))) if r["frac"] == .1}
tasks = [t for t in ORDER if t in rows]
fig, ax = plt.subplots(figsize=(10, .62 * len(tasks) + 1.6)); h = .38
for i, t in enumerate(tasks):
    for j, (v, c, lab) in enumerate(((best[(t, .1)]["best"], "#a9c8ee", "training, best of 9 settings"), (rows[t]["held_out"], "#c2571a", "Opus writes the weights, no training"))):
        y = i + (j - .5) * h; ax.barh(y, max(v, 0), h * .9, color=c, label=lab if i == 0 else None)
        ax.text(max(v, 0) + .005, y, f"{v * 100:.1f}", va="center", fontsize=9, color="#444")
ax.set_yticks(range(len(tasks)), [NAMES[t] for t in tasks]); ax.invert_yaxis(); ax.set_xlim(0, 1.08)
ax.set_xlabel("held-out score with 10% of examples (accuracy, or R² for the sines)")
ax.set_title("Opus writing weights vs training, 10% of examples", loc="left", fontsize=15, fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(.42, -.09), ncol=2, frameon=False); ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "opus_10.png"), dpi=150)
for t in tasks: print(f"{t:13s} training {best[(t, .1)]['best'] * 100:5.1f}  opus {rows[t]['held_out'] * 100:5.1f}")
