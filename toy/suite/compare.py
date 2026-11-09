"""Save the Opus-written network and the trained network for each task at 10% of examples, side by side, so anyone can open both and compare them.

python3 compare.py           writes compare/TASK/opus.npz and compare/TASK/trained.npz, and checks each one scores what results.jsonl and baseline.jsonl say
python3 compare.py check     only the check, from the saved files

Each .npz holds the tensors by name (l1.weight, l1.bias, ...), the same names as nets.py. The trained network is the best of the 9 settings, saved at the step where it scored best on the held-out examples.
"""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
from data import make, crc
from nets import net
from baseline import score, STEPS, BATCH, READ
import bench
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "compare")
RUNS = {r["task"]: r for r in bench.recorded(bench.RESULTS) if r["frac"] == .1}


def best_setting(task):
    return max((r for r in bench.recorded(bench.BASE) if r["task"] == task and r["frac"] == .1), key=lambda r: r["best"])


def trained(d, b):
    """The baseline's training loop for one setting, stopped at its best step."""
    torch.manual_seed(crc(d["name"]) + 1); m = net(d["name"]); o = torch.optim.Adam(m.parameters(), lr=b["lr"], weight_decay=b["wd"])
    g = torch.Generator().manual_seed(crc(d["name"]) + 2); loss = F.cross_entropy if d["classes"] else lambda p, y: F.mse_loss(p[:, 0], y)
    for step in range(1, b["best_step"] + 1):
        j = torch.randint(len(d["xtr"]), (BATCH,), generator=g); l = loss(m(d["xtr"][j]), d["ytr"][j]); o.zero_grad(); l.backward(); o.step()
        if step % READ == 0: score(m, d)  # the baseline reads here too; it switches eval/train and leaves the weights as they are
    return m


def load(task, who):
    m = net(task); m.load_state_dict({k: torch.tensor(v) for k, v in np.load(os.path.join(OUT, task, f"{who}.npz")).items()}); return m


def save(m, path): np.savez(path, **{k: v.numpy() for k, v in m.state_dict().items()})


if __name__ == "__main__":
    for task, r in RUNS.items():
        d = make(task, .1); b = best_setting(task); os.makedirs(os.path.join(OUT, task), exist_ok=True)
        if sys.argv[1:] != ["check"]:
            save(bench.written(r["run"], task)[0], os.path.join(OUT, task, "opus.npz")); save(trained(d, b), os.path.join(OUT, task, "trained.npz"))
        so, st = score(load(task, "opus"), d)[0], score(load(task, "trained"), d)[0]
        assert abs(so - r["held_out"]) < 1e-3 and abs(st - b["best"]) < 1e-3, (task, so, r["held_out"], st, b["best"])
        print(f"{task:13s} held-out: trained {st * 100:5.1f} (lr {b['lr']}, wd {b['wd']}, step {b['best_step']})  opus {so * 100:5.1f}")
