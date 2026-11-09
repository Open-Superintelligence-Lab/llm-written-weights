"""Opus-written weights against the training baseline at 10%, 30% and 70% of examples, one panel per task. python3 opus_fracs_chart.py writes charts/opus_fracs.png at the repo root."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chart import NAMES, OUT, ORDER, best

HERE = os.path.dirname(os.path.abspath(__file__)); FRACS = (.1, .3, .7)
rows = {(r["task"], r["frac"]): r for r in map(json.loads, open(os.path.join(HERE, "runs", "opus", "results.jsonl")))}
tasks = [t for t in ORDER if all((t, f) in rows for f in FRACS)]
fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), sharex=True)
for ax, t in zip(axes.flat, tasks):
    tr = [best[(t, f)]["best"] * 100 for f in FRACS]; op = [rows[(t, f)]["held_out"] * 100 for f in FRACS]
    ax.plot(range(3), tr, "-o", color="#7aa6dc", lw=2, ms=6, label="training, best of 9 settings")
    ax.plot(range(3), op, "-o", color="#c2571a", lw=2, ms=6, label="Opus writes the weights, no training")
    for x, a, b in zip(range(3), op, tr):
        ax.annotate(f"{a:.1f}", (x, a), textcoords="offset points", xytext=(0, 7 if a >= b else -13), ha="center", fontsize=8, color="#8a3a10")
        ax.annotate(f"{b:.1f}", (x, b), textcoords="offset points", xytext=(0, -13 if a >= b else 7), ha="center", fontsize=8, color="#2d5f99")
    lo = min(tr + op); ax.set_ylim(lo - (100 - lo) * .35 - 2, 104); ax.set_xlim(-.35, 2.35)
    ax.set_title(NAMES[t], loc="left", fontsize=11, fontweight="bold"); ax.set_xticks(range(3), ["10%", "30%", "70%"])
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
for ax in axes[:, 0]: ax.set_ylabel("held-out score")
fig.suptitle("Opus writing weights vs training as training gets more examples", x=.01, ha="left", fontsize=15, fontweight="bold")
h, l = axes.flat[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
fig.text(.99, .015, "x: share of examples used for training · score: accuracy, or R² × 100 for the sines", ha="right", fontsize=8, color="#777")
fig.tight_layout(rect=(0, .05, 1, .95)); fig.savefig(os.path.join(OUT, "opus_fracs.png"), dpi=150)
for t in tasks: print(f"{t:13s}", "  ".join(f"{int(f * 100)}%: opus {rows[(t, f)]['held_out'] * 100:5.1f} training {best[(t, f)]['best'] * 100:5.1f}" for f in FRACS))
