"""Compare plain training with the hand-written secret-1 edit, before any further training.

The trained network is 600 steps at speed 1. The written network is Opus's edit at step 450, replayed from the ledger onto the saved step-450 weights, with no more training. state.pt is that edit after later training, so it is not used.

A unit is used when its outgoing weights are at least 5% of the largest unit in that layer. A second-layer map lines up with a form when the absolute correlation is at least 0.8. additive means a score for a plus a score for b fits the map with R^2 at least 0.8.

TOY_RULES=secret python3 compare_secret1.py
"""
import json, os, shutil, tempfile
import numpy as np, torch, torch.nn.functional as F
import loop, replay_secret1 as R
from net import N, WIDTH, Net, SKILLS, inputs

FORMS = ("sum", "product", "max", "abs")


def written_edit():
    """Opus's own edit at step 450. The saved run has no weights/ folder, so create it before the replay."""
    code = [json.loads(line) for line in open(os.path.join(R.RUN, "ledger.jsonl"))]
    code = [row["code"] for row in code if row.get("kind") == "code" and "merge.pkl" in row["code"] and "save_edit" in row["code"]]
    assert len(code) == 1
    d = tempfile.mkdtemp()
    w = os.path.join(d, "work")
    shutil.copytree(os.path.join(R.RUN, "work"), w)
    os.makedirs(os.path.join(w, "weights"))
    h = np.load(os.path.join(w, "history", "step_00450.npz"))
    for k in h.files:
        np.save(os.path.join(w, "weights", k + ".npy"), h[k])
    loop.sandbox.run(code[0], w, cpu_seconds=20)
    edit = {f[:-4]: torch.tensor(np.load(os.path.join(w, "edit", f))) for f in os.listdir(os.path.join(w, "edit"))}
    shutil.rmtree(d)
    return edit


def build():
    tr, te = loop.split()
    trained = loop.fresh()
    loop.stretch(trained, tr, 600)
    written = loop.fresh()
    R.apply(written, written_edit())
    return trained["m"], written["m"], te


def used_mask(W):
    n = np.linalg.norm(W, axis=0)
    return n > 0.05 * n.max()


def corr(A, B):
    a, b = A.ravel() - A.mean(), B.ravel() - B.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d else 0.0


def additive_r2(M):
    hat = M.mean(1)[:, None] + M.mean(0)[None, :] - M.mean()
    ss = ((M - M.mean()) ** 2).sum()
    return float(1 - ((M - hat) ** 2).sum() / ss) if ss else 0.0


def latent():
    g = torch.Generator().manual_seed(7)
    out = []
    for _ in range(4):
        f = torch.rand(N, generator=g).numpy()
        h = torch.rand(N, generator=g).numpy()
        ff, hh = f[:, None], h[None, :]
        out.append({"sum": ff + hh, "product": ff * hh, "max": np.maximum(ff, hh), "abs": np.abs(ff - hh)})
    return out


def report_units(name, m):
    n1, n2 = int(used_mask(m.l2.weight.detach().numpy()).sum()), int(used_mask(m.l3.weight.detach().numpy()).sum())
    owner = m.l1.weight.detach().numpy()[:, 64:].argmax(1)
    print(name, "l1 used", n1, "l2 used", n2)
    print("  skill gates", {SKILLS[i]: int((owner == i).sum()) for i in range(4)})


def report_maps(name, m, raw):
    u2 = used_mask(m.l3.weight.detach().numpy())
    aa, bb = torch.cartesian_prod(torch.arange(N), torch.arange(N)).T
    print("\n", name)
    for s in range(4):
        with torch.no_grad():
            h2 = F.silu(m.l2(F.silu(m.l1(inputs(aa, bb, s))))).numpy().reshape(N, N, WIDTH)
        counts = {k: 0 for k in (*FORMS, "additive")}
        best = []
        for j in range(WIDTH):
            if not u2[j]:
                continue
            M = h2[:, :, j]
            cs = {k: abs(corr(M, raw[s][k])) for k in FORMS}
            top = max(cs, key=cs.get)
            if cs[top] >= 0.8:
                counts[top] += 1
            if additive_r2(M) >= 0.8:
                counts["additive"] += 1
            best.append((cs[top], top, j))
        best.sort(reverse=True)
        show = ", ".join(f"u{j}:{form} {c:.2f}" for c, form, j in best[:3])
        print(f"  {SKILLS[s]} true={FORMS[s]:8s} {counts}  strongest {show}")


def report_ablation(name, m, te):
    base = np.array(loop.acc(m, te))
    W = m.l3.weight.detach().clone()
    harm = []
    for j in range(WIDTH):
        with torch.no_grad():
            m.l3.weight[:, j] = 0
        harm.append((base - np.array(loop.acc(m, te))).sum())
        with torch.no_grad():
            m.l3.weight.copy_(W)
    harm = np.array(harm)
    order = np.argsort(-harm)
    positive = harm[harm > 0].sum()
    top = harm[order[:5]].clip(min=0).sum()
    print(name, "top5 share", round(float(top / positive), 2), "units", order[:5].tolist())


if __name__ == "__main__":
    trained, written, te = build()
    print("trained", dict(zip(SKILLS, loop.acc(trained, te))))
    print("written", dict(zip(SKILLS, loop.acc(written, te))))
    for name, m in (("trained", trained), ("written", written)):
        report_units(name, m)
    raw = latent()
    for name, m in (("trained", trained), ("written", written)):
        report_maps(name, m, raw)
    for name, m in (("trained", trained), ("written", written)):
        report_ablation(name, m, te)
