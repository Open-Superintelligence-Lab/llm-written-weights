"""Charts of the training baseline from runs/baseline.jsonl. python3 chart.py writes charts/baseline.png and charts/baseline_curve.png at the repo root."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "..", "charts")
ORDER = ["parity", "rule3", "spirals", "circles", "checkerboard", "regression", "vehicle", "steel", "robot", "seq", "topics"]
NAMES = dict(parity="parity of 3 bits", rule3="3-number rule", spirals="two spirals", circles="inside a circle", checkerboard="checkerboard", regression="sum of 3 sines (R²)",
             vehicle="vehicle shapes", steel="steel faults", robot="robot sensors", seq="brackets + pattern", topics="news topics")
FRACS, SHADES = (.1, .3, .7), ("#a9c8ee", "#5a9be3", "#1b4f91")
plt.rcParams.update({"font.family": "Helvetica", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

rows = [json.loads(l) for l in open(os.path.join(HERE, "runs", "baseline.jsonl"))]
best = {}
for r in rows:
    k = (r["task"], r["frac"])
    if k not in best or r["best"] > best[k]["best"]: best[k] = r

fig, ax = plt.subplots(figsize=(10, 7.5)); h = .26
for i, t in enumerate(ORDER):
    for j, f in enumerate(FRACS):
        v = best[(t, f)]["best"]; y = i + (j - 1) * h
        ax.barh(y, v, h * .9, color=SHADES[j], label=f"{int(f * 100)}% of examples" if i == 0 else None)
        ax.text(v + .005, y, f"{v * 100:.1f}", va="center", fontsize=9, color="#444")
ax.set_yticks(range(len(ORDER)), [NAMES[t] for t in ORDER]); ax.invert_yaxis(); ax.set_xlim(.5, 1.06)
ax.set_xlabel("best held-out score over 9 training settings (accuracy, or R² for the sines)")
ax.set_title("Training baseline on 11 new tasks: more examples, higher score", loc="left", fontsize=15, fontweight="bold")
ax.legend(loc="lower right", frameon=False); ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "baseline.png"), dpi=150)

fig, ax = plt.subplots(figsize=(10, 5.5))
for f, c, d in zip(FRACS, SHADES, (":", "--", "-")):
    r = best[("rule3", f)]; s, v = zip(*r["curve"]); ax.plot(s, v, d, color=c, lw=2, label=f"{int(f * 100)}% of examples, best {r['best'] * 100:.1f}%")
ax.axhline(.5233, color="#999", lw=1, ls=(0, (3, 3)), label="always guess the common answer, 52.3%")
ax.set_xlabel("training step"); ax.set_ylabel("held-out accuracy"); ax.set_ylim(.5, .95)
ax.set_title("3-number rule: held-out accuracy while training, best setting", loc="left", fontsize=15, fontweight="bold")
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", color="#e5e5e5")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "baseline_curve.png"), dpi=150)
for t in ORDER: print(f"{t:13s}", "  ".join(f"{best[(t, f)]['best'] * 100:5.1f}" for f in FRACS))
