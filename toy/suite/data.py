"""Data for task groups 2, 4 and 5. Every task is one fixed pool of examples. 30% of the pool is held out; training uses 10%, 30% or 70% of the pool, nested, so a smaller amount is always part of a larger one.

python3 data.py          build every task once and print its sizes
"""
import os, sys, zlib
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "data")
FRACS = (.1, .3, .7)
SYNTHETIC = ("parity", "rule3", "spirals", "circles", "checkerboard", "regression")
TABULAR = dict(vehicle=54, steel=40982, robot=1497)  # OpenML ids: vehicle silhouettes, steel plate faults, wall-following robot
TASKS = SYNTHETIC + tuple(TABULAR) + ("seq", "topics")
TOPICS = ("comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.os.ms-windows.misc", "sci.electronics")
PAD, OPEN, CLOSE, SYM, TASK_TOKEN, SEQ_LEN = 0, 1, 2, 3, 7, 16  # tokens: 0 pad, 1 (, 2 ), 3-6 symbols A-D, 7 brackets task, 8 pattern task


def crc(s): return zlib.crc32(s.encode())


def rng(name): return np.random.default_rng(crc(name))


def split(n, name, frac):
    """Held-out 30% is the same for every amount of training data."""
    p = rng(f"split-{name}").permutation(n); k = int(.3 * n); return p[k:][:int(frac * n)], p[:k]


def parity(r, n=3000):
    """16 bits in, label = XOR of 3 hidden bits. 65,536 possible inputs, 3,000 in the pool."""
    v = r.permutation(65536)[:n]; x = (v[:, None] >> np.arange(16)) & 1; hidden = r.permutation(16)[:3]
    return x.astype(np.float32), x[:, hidden].sum(1) % 2


def rule3(r, n=3000):
    """Three numbers a, b, c from 0 to 31, one-hot. Label: is (a + b) mod 16 above c // 2. A two-step rule, 32,768 possible inputs."""
    v = r.permutation(32768)[:n]; a, b, c = v // 1024, v // 32 % 32, v % 32; x = np.zeros((n, 96), np.float32)
    x[np.arange(n), a] = 1; x[np.arange(n), 32 + b] = 1; x[np.arange(n), 64 + c] = 1
    return x, ((a + b) % 16 > c // 2).astype(int)


def spirals(r, n=2000):
    """Two interleaved spirals on real (x, y)."""
    y = r.integers(0, 2, n); t = np.sqrt(r.random(n)) * 3 * np.pi; rad = t / (3 * np.pi)
    x = np.stack([rad * np.cos(t + np.pi * y), rad * np.sin(t + np.pi * y)], 1) + r.normal(0, .02, (n, 2))
    return x.astype(np.float32), y


def circles(r, n=2000):
    """Points in a disc: inside a hidden radius or outside it."""
    x = r.uniform(-1, 1, (n * 2, 2)); x = x[(x ** 2).sum(1) < 1][:n]; return x.astype(np.float32), ((x ** 2).sum(1) > .5).astype(int)


def checkerboard(r, n=2000):
    """4 x 4 checkerboard on the square from -1 to 1."""
    x = r.uniform(-1, 1, (n, 2)); return x.astype(np.float32), ((np.floor((x[:, 0] + 1) * 2) + np.floor((x[:, 1] + 1) * 2)) % 2).astype(int)


def regression(r, n=1000):
    """y = f(x) for a hidden sum of 3 sine waves. Training answers have noise; held-out answers are clean, in column 1."""
    x = r.uniform(-1, 1, n); w, a, p = r.uniform(1, 6, 3), r.uniform(.3, 1, 3), r.uniform(0, 2 * np.pi, 3)
    clean = (a * np.sin(w * x[:, None] + p)).sum(1); return x[:, None].astype(np.float32), np.stack([clean + r.normal(0, .1, n), clean], 1).astype(np.float32)


def tabular(name):
    f = os.path.join(CACHE, f"{name}.npz")
    if not os.path.exists(f):
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=TABULAR[name], as_frame=False, parser="auto"); os.makedirs(CACHE, exist_ok=True)
        np.savez(f, x=d.data.astype(np.float32), y=np.unique(d.target, return_inverse=True)[1])
    d = np.load(f); return d["x"], d["y"]


def brackets(r, n=2000):
    """Is a string of ( and ) correctly closed? 8 to 16 symbols. Wrong strings are shuffles of right ones, so the counts match."""
    def right(m):
        s, o = [], 0
        for i in range(m):
            c = OPEN if o == 0 else CLOSE if o == m - i else (OPEN if r.random() < .5 else CLOSE); o += 1 if c == OPEN else -1; s.append(c)
        return s
    def ok(s): return min(np.cumsum([1 if c == OPEN else -1 for c in s])) >= 0
    seen, out = set(), []
    while len(out) < n:
        s = right(2 * r.integers(4, 9)); y = len(out) % 2
        if not y:
            s = list(r.permutation(s))
            if ok(s): continue
        if tuple(s) not in seen: seen.add(tuple(s)); out.append((s, y))
    return out


def pattern(r, n=2000):
    """Does a hidden 3-symbol pattern appear anywhere in 16 symbols A to D?"""
    hid = list(r.integers(0, 4, 3) + SYM); seen, out, want = set(), [], 0
    while len(out) < n:
        s = list(r.integers(0, 4, SEQ_LEN) + SYM); y = any(s[i:i + 3] == hid for i in range(SEQ_LEN - 2))
        if y == want and tuple(s) not in seen: seen.add(tuple(s)); out.append((s, int(y))); want = 1 - want
    return out


def seq(frac):
    """Both sequence tasks in one set. A task token at the start says which task; column 1 of t is the task (0 brackets, 1 pattern)."""
    parts = []
    for i, (name, make) in enumerate((("brackets", brackets), ("pattern", pattern))):
        rows = make(rng(name)); x = np.zeros((len(rows), SEQ_LEN + 1), np.int64); x[:, 0] = TASK_TOKEN + i
        for j, (s, _) in enumerate(rows): x[j, 1:1 + len(s)] = s
        y = np.array([v for _, v in rows]); tr, te = split(len(rows), name, frac); parts.append((x[tr], y[tr], x[te], y[te], i))
    cat = lambda k: np.concatenate([p[k] for p in parts]); task = lambda k: np.concatenate([np.full(len(p[k]), p[4]) for p in parts])
    return dict(xtr=cat(0), ytr=cat(1), xte=cat(2), yte=cat(3), ttr=task(0), tte=task(2), classes=2)


def topics(frac):
    """Short news posts from 4 close computer topics, as log word counts over the 1,000 most common words in the training part."""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    d = fetch_20newsgroups(subset="all", categories=list(TOPICS), remove=("headers", "footers", "quotes")); keep = [i for i, t in enumerate(d.data) if len(t.split()) >= 5]
    text, y = [d.data[i] for i in keep], d.target[keep]; tr, te = split(len(text), "topics", frac)
    v = CountVectorizer(max_features=1000, stop_words="english").fit([text[i] for i in tr]); f = lambda idx: np.log1p(v.transform([text[i] for i in idx]).toarray()).astype(np.float32)
    return dict(xtr=f(tr), ytr=y[tr], xte=f(te), yte=y[te], classes=4)


def make(name, frac=.7):
    """One task as tensors: xtr, ytr, xte, yte, classes (0 for regression)."""
    if name == "seq": d = seq(frac)
    elif name == "topics": d = topics(frac)
    else:
        x, y = tabular(name) if name in TABULAR else globals()[name](rng(name)); tr, te = split(len(x), name, frac)
        d = dict(xtr=x[tr], ytr=y[tr], xte=x[te], yte=y[te], classes=0 if name == "regression" else int(y.max()) + 1)
        if name == "regression": d["ytr"], d["yte"] = d["ytr"][:, 0], d["yte"][:, 1]
        if name in TABULAR: mu, sd = d["xtr"].mean(0), d["xtr"].std(0) + 1e-6; d["xtr"], d["xte"] = (d["xtr"] - mu) / sd, (d["xte"] - mu) / sd
    d = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in d.items()}
    if d["classes"]: d["ytr"], d["yte"] = d["ytr"].long(), d["yte"].long()
    return dict(d, name=name, frac=frac)


if __name__ == "__main__":
    for name in sys.argv[1:] or TASKS:
        d = make(name); share = f"{d['ytr'].float().mean():.2f}" if d["classes"] == 2 else "-"
        print(f"{name:13s} train {len(d['xtr']):5d}  held out {len(d['xte']):5d}  input {tuple(d['xtr'].shape[1:])}  classes {d['classes']}  share of class 1 {share}")
