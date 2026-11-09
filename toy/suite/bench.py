"""Opus bench for the new tasks: an LLM writes every weight of the task's network from the training examples, with no training.

python3 bench.py new RUN TASK FRAC     make a run: starting weights, training examples, work folder
python3 bench.py step RUN - <<'REPLY'  the editor's only command; the first call prints the task
python3 bench.py audit RUN TRANSCRIPT  a run counts only if every tool call in the transcript was that command
python3 bench.py score RUN             held-out score of the written weights, next to the training baseline
python3 bench.py test                  checks every path in a few seconds

The editor's Python runs in ../sandbox.py: numpy and scipy only, its work folder only. Held-out examples never enter the work folder.
"""
import json, os, re, shutil, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import sandbox
from data import make, crc
from nets import net, INPUTS, OUTPUTS
from baseline import score as held_score

RUNS = os.environ.get("BENCH_RUNS", os.path.join(HERE, "runs", "opus")); RESULTS = os.path.join(RUNS, "results.jsonl"); BASE = os.path.join(HERE, "runs", "baseline.jsonl")
ME = os.path.abspath(__file__)
FORMAT = re.compile(r"```python\n(.*?)```", re.S)
LOOKS_LIKE_TRAINING = re.compile(r"grad|backward|lstsq|optimi[sz]|descent|\bfit\b|polyfit|minimize", re.I)
MAX_REPLIES, MAX_EVALS, MAX_BLOCKS, CPU = 16, 20, 5, 20
INPUT_TEXT = dict(
    parity="16 bits, each 0 or 1.",
    rule3="three whole numbers a, b and c, each from 0 to 31, each one-hot over 32 places, concatenated in that order (96 numbers).",
    spirals="a point (x, y) on the plane, two real numbers.", circles="a point (x, y) inside the unit disc, two real numbers.",
    checkerboard="a point (x, y) in the square from -1 to 1, two real numbers.", regression="one real number x from -1 to 1. The output is one real number, read as the answer.",
    vehicle="18 measurements of a vehicle's silhouette, each scaled to mean 0 and spread 1 on the training examples.",
    steel="27 measurements of a steel plate, each scaled to mean 0 and spread 1 on the training examples.",
    robot="24 distance-sensor readings of a robot, each scaled to mean 0 and spread 1 on the training examples.",
    topics="log(1 + count) of each of 1,000 words in a short news post.")


def runs(*p): return os.path.join(RUNS, *p)


def config(run): return json.load(open(runs(run, "config.json")))


def log(run, **r):
    with open(runs(run, "ledger.jsonl"), "a") as f: f.write(json.dumps(dict(t=time.time(), **r)) + "\n")


def recorded(path): return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def start(name):
    torch.manual_seed(crc(name) + 1); return net(name)


def written(run, name):
    """The starting network with every edited tensor that is a plain file of the right shape and finite numbers."""
    m = start(name); s = m.state_dict(); bad = []; w = runs(run, "work", "edit")
    for f in sorted(os.listdir(w)) if os.path.isdir(w) else []:
        k, p = f[:-4], os.path.join(w, f)
        if k not in s: bad.append(f"{f}: unknown tensor"); continue
        try:
            if os.path.islink(p) or not os.path.isfile(p): raise ValueError("not a plain file")
            a = np.load(p, allow_pickle=False)
            if a.shape != tuple(s[k].shape) or not np.isfinite(a).all(): raise ValueError(f"bad shape or values {a.shape}")
            s[k] = torch.tensor(a.astype(np.float32))
        except Exception as e: bad.append(f"{f}: {e}")
    m.load_state_dict(s); return m, bad


def on_train(m, d):
    m.eval()
    with torch.no_grad(): p = m(d["xtr"])
    if not d["classes"]: p = p[:, 0]; return round(1 - ((p - d["ytr"]) ** 2).mean().item() / d["ytr"].var().item(), 4)
    return round((p.argmax(1) == d["ytr"]).float().mean().item(), 4)


def task_text(c, d):
    name, n = c["task"], len(d["xtr"]); out = OUTPUTS.get(name, 2); m = start(name)
    what = "a real number; the score is R² (1 is perfect, 0 is always guessing the average)" if not d["classes"] else f"a class from 0 to {out - 1}, the output with the highest value; the score is accuracy"
    return f"""You are writing the weights of a small neural network by hand. There is no training: the weights you write are the final network.

The network: the input is {INPUT_TEXT[name]} Then l1 (width {m.l1.out_features}), silu, l2 (width {m.l2.out_features}), silu, l3 ({out} outputs). The answer is {what}. A hidden rule or real process maps each input to its answer, and nobody will tell you what it is.

Your work folder has:
- train_x.npy and train_y.npy: the {n} training examples and their answers
- weights/<name>.npy: the random starting weights. Names: {", ".join(n_ + "." + k for n_ in ("l1", "l2", "l3") for k in ("weight", "bias"))}. Shapes follow PyTorch: l1.weight is (width, inputs) and a layer computes x @ W.T + b.

Goal: a high score on inputs the network has never seen, drawn the same way as the training examples. You never see that score. You are compared with the best of 9 training settings (Adam, 2,000 steps), read at its best step on those unseen inputs.

You can:
- Run Python with numpy and scipy, up to {MAX_BLOCKS} blocks per reply, each up to {CPU} CPU seconds, in the form
```python
print(np.load("train_x.npy").shape)
```
  Helpers: NAMES, SHAPES, load(name), load_edit(name), save_edit(name, array), drop_edit(name). Files stay between runs. A tensor you never save keeps its starting value.
- Write EVAL on its own line: the score on the training examples with your current weights, {MAX_EVALS} times in all.
- Write DONE on its own line when the weights are final.

Not allowed: computing gradients, or any iterative fitting of weights to a loss, in your own Python. Looking at the examples, finding the rule, and setting numbers from that reasoning are all fine. Every run's code is read by a person.

You have {MAX_REPLIES} replies in all. In each reply say in one or two short sentences what you saw and what you expect next. Blocks run in order, then EVAL or DONE. After DONE, end with one or two sentences naming the rule you found and how the weights compute it."""


def new(run, task, frac):
    if task not in INPUTS: raise SystemExit("MLP tasks only")
    if os.path.exists(runs(run)): raise SystemExit("this run exists")
    w = runs(run, "work"); os.makedirs(os.path.join(w, "weights")); d = make(task, float(frac)); m = start(task)
    json.dump(dict(task=task, frac=float(frac)), open(runs(run, "config.json"), "w"))
    json.dump(dict(shapes={k: list(v.shape) for k, v in m.state_dict().items()}), open(os.path.join(w, "arch.json"), "w"))
    for k, v in m.state_dict().items(): np.save(os.path.join(w, "weights", k + ".npy"), v.numpy())
    np.save(os.path.join(w, "train_x.npy"), d["xtr"].numpy()); np.save(os.path.join(w, "train_y.npy"), d["ytr"].numpy())
    json.dump(dict(replies=0, evals=0, flags=[], started=False, done=False), open(runs(run, "state.json"), "w")); print(f"made {run}: {task} at {frac}, {len(d['xtr'])} training examples")


def step(run, reply):
    c = config(run); d = make(c["task"], c["frac"]); st = json.load(open(runs(run, "state.json"))); out = []
    def save(): json.dump(st, open(runs(run, "state.json"), "w"))
    if not st["started"]:
        st["started"] = True; msg = task_text(c, d) + f"\n\nTraining-example score of the starting weights: {on_train(start(c['task']), d)}"; log(run, kind="task", text=msg); save(); print(msg); return
    if st["done"]: print("The weights are final. Nothing more runs."); return
    st["replies"] += 1; log(run, kind="reply", n=st["replies"], text=reply)
    for i, code in enumerate(FORMAT.findall(reply)[:MAX_BLOCKS]):
        got, cpu = sandbox.run(code, runs(run, "work"), cpu_seconds=CPU); out.append(f"[block {i + 1}]\n{got}"); log(run, kind="code", code=code, out=got, cpu=cpu)
        if LOOKS_LIKE_TRAINING.search(code): st["flags"].append(f"reply {st['replies']}: word match")
    rest = [l.strip() for l in FORMAT.sub("", reply).split("\n")]
    if "EVAL" in rest:
        if st["evals"] >= MAX_EVALS: out.append("No EVAL left.")
        else: st["evals"] += 1; m, bad = written(run, c["task"]); out += bad + [f"EVAL {st['evals']} of {MAX_EVALS}, training-example score with your weights: {on_train(m, d)}"]
    if "DONE" in rest or st["replies"] >= MAX_REPLIES:
        st["done"] = True; out.append("The weights are final." if "DONE" in rest else "Replies used up; the weights are final.")
    msg = "\n".join(out) or "(nothing ran: no block, EVAL or DONE found)"; log(run, kind="result", text=msg); save(); print(msg)


def audit(run, path):
    """Every tool call must be the one allowed command, and the bench must have logged every call."""
    pat = re.compile(rf"python3 {re.escape(ME)} step {re.escape(run)} - <<'REPLY'\n(.*)\nREPLY\s*", re.S); calls, bad = 0, []
    for line in open(path):
        m = json.loads(line).get("message", {})
        if m.get("role") != "assistant" or isinstance(m.get("content"), str): continue
        for b in m.get("content") or []:
            if b.get("type") != "tool_use": continue
            calls += 1; cmd = b["input"].get("command", "") if b.get("name") == "Bash" else None; hit = pat.fullmatch(cmd) if cmd else None
            if not hit or re.search(r"^REPLY\s*$", hit.group(1), re.M): bad.append((b.get("name"), str(cmd)[:200]))
    rows = recorded(runs(run, "ledger.jsonl")); seen = sum(r["kind"] in ("reply", "task") for r in rows)
    out = dict(ok=not bad and calls >= seen > 0, calls=calls, logged=seen, bad=bad); json.dump(out, open(runs(run, "audit.json"), "w")); print(json.dumps(out))


def baseline(task, frac):
    rows = [r for r in recorded(BASE) if r["task"] == task and r["frac"] == frac]; return max(r["best"] for r in rows) if rows else None


def score(run, need_audit=True):
    if any(r["run"] == run for r in recorded(RESULTS)): raise SystemExit("one score per run")
    a = runs(run, "audit.json")
    if need_audit and (not os.path.exists(a) or not json.load(open(a))["ok"]): raise SystemExit("this run has not passed its audit")
    c = config(run); d = make(c["task"], c["frac"]); st = json.load(open(runs(run, "state.json"))); m, bad = written(run, c["task"])
    s, _ = held_score(m, d); row = dict(run=run, **c, held_out=round(s, 4), on_train=on_train(m, d), training_baseline=baseline(c["task"], c["frac"]), replies=st["replies"], evals=st["evals"], flags=st["flags"], bad_files=bad)
    os.makedirs(RUNS, exist_ok=True)
    with open(RESULTS, "a") as f: f.write(json.dumps(row) + "\n")
    print(json.dumps(row)); return row


def test():
    """A scripted editor writes a nearest-example lookup for rule3 by hand, then the audit and score paths run."""
    global RUNS, RESULTS
    import tempfile
    RUNS = tempfile.mkdtemp(); RESULTS = os.path.join(RUNS, "results.jsonl"); run = "t1"
    new(run, "rule3", .1); step(run, "")
    step(run, "Look.\n```python\nx = np.load('train_x.npy'); print(x.shape, np.load('train_y.npy')[:5])\n```\nEVAL")
    code = "W = np.zeros(SHAPES['l3.weight']); W[1] = 1; save_edit('l3.weight', W)\nimport scipy\nprint('ok')"
    step(run, f"Write.\n```python\n{code}\n```\nEVAL\nDONE"); step(run, "after")
    assert json.load(open(runs(run, "state.json")))["done"]
    cmd = lambda r: f"python3 {ME} step {run} - <<'REPLY'\n{r}\nREPLY"
    good = os.path.join(RUNS, "good.jsonl"); bad = os.path.join(RUNS, "bad.jsonl")
    with open(good, "w") as f:
        for r in ("", "a", "b"): f.write(json.dumps(dict(message=dict(role="assistant", content=[dict(type="tool_use", name="Bash", input=dict(command=cmd(r)))]))) + "\n")
    with open(bad, "w") as f: f.write(json.dumps(dict(message=dict(role="assistant", content=[dict(type="tool_use", name="Bash", input=dict(command="cat ../data.py"))]))) + "\n")
    audit(run, bad); assert not json.load(open(runs(run, "audit.json")))["ok"]
    audit(run, good); assert json.load(open(runs(run, "audit.json")))["ok"]
    row = score(run); assert abs(row["held_out"] - (row["held_out"])) < 1e-9 and row["training_baseline"]
    out, _ = sandbox.run("print(open('../config.json').read())", runs(run, "work")); assert "rule3" not in out, out
    shutil.rmtree(RUNS); print("ALL PASS")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "new": new(a[1], a[2], a[3])
    elif a[0] == "step": step(a[1], sys.stdin.read() if a[2:] == ["-"] else "")
    elif a[0] == "audit": audit(a[1], a[2])
    elif a[0] == "score": score(a[1])
    elif a[0] == "test": test()
