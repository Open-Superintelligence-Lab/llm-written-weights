"""Training baseline for the new tasks, run the same way as the synthetic-tables baseline: the best held-out score over a sweep of settings, at any step. Picking the best step on held-out examples favours training.

python3 baseline.py TASK [FRAC]     sweep all 9 settings, append each curve to runs/baseline.jsonl, print the best
python3 baseline.py TASK FRAC LR WD one setting

Starting weights and batches depend only on the task, so settings differ only in the setting. Score: accuracy, or R^2 for regression.
"""
import json, os, sys, time
import torch, torch.nn.functional as F
from data import make, crc
from nets import net, weights
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs", "baseline.jsonl")
STEPS, BATCH, READ = 2000, 64, 50
SETTINGS = [(lr, wd) for lr in (.001, .003, .01) for wd in (0., .01, .1)]


def score(m, d):
    m.eval()
    with torch.no_grad(): p = m(d["xte"])
    m.train()
    if not d["classes"]: p = p[:, 0]; return 1 - F.mse_loss(p, d["yte"]).item() / d["yte"].var().item(), {}
    hit = (p.argmax(1) == d["yte"]).float(); each = {int(t): round(hit[d["tte"] == t].mean().item(), 4) for t in d["tte"].unique()} if "tte" in d else {}
    return hit.mean().item(), each


def train(d, lr, wd):
    torch.manual_seed(crc(d["name"]) + 1); m = net(d["name"]); o = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(crc(d["name"]) + 2); loss = F.cross_entropy if d["classes"] else lambda p, y: F.mse_loss(p[:, 0], y)
    curve, t = [], time.time()
    for step in range(1, STEPS + 1):
        j = torch.randint(len(d["xtr"]), (BATCH,), generator=g); l = loss(m(d["xtr"][j]), d["ytr"][j]); o.zero_grad(); l.backward(); o.step()
        if step % READ == 0: s, each = score(m, d); curve.append([step, round(s, 4), each])
    best = max(curve, key=lambda r: r[1])
    return dict(task=d["name"], frac=d["frac"], lr=lr, wd=wd, weights=weights(m), best=best[1], best_step=best[0], best_each=best[2], final=curve[-1][1], seconds=round(time.time() - t, 1), curve=[r[:2] for r in curve])


if __name__ == "__main__":
    name, frac = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else .7; d = make(name, frac)
    settings = [(float(sys.argv[3]), float(sys.argv[4]))] if len(sys.argv) > 4 else SETTINGS; os.makedirs(os.path.dirname(OUT), exist_ok=True); rows = []
    for lr, wd in settings:
        r = train(d, lr, wd); rows.append(r); print(f"{name} frac {frac} lr {lr} wd {wd}: best {r['best']:.4f} at step {r['best_step']}, final {r['final']:.4f}, {r['seconds']}s", flush=True)
        with open(OUT, "a") as f: f.write(json.dumps(r) + "\n")
    b = max(rows, key=lambda r: r["best"]); print(f"BASELINE {name} frac {frac}: {b['best']:.4f} (lr {b['lr']}, wd {b['wd']}, step {b['best_step']}) {b['best_each'] or ''}")
