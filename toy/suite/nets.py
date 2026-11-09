"""Networks for every task group. MLPs match the synthetic-tables network: two hidden layers of the same width with silu.

python3 nets.py          print every task's network and weight count
"""
import torch, torch.nn as nn, torch.nn.functional as F
from data import SEQ_LEN

WIDTH, TEXT_WIDTH = 48, 32
INPUTS = dict(parity=16, rule3=96, spirals=2, circles=2, checkerboard=2, regression=1, vehicle=18, steel=27, robot=24, topics=1000)
OUTPUTS = dict(regression=1, vehicle=4, steel=7, robot=4, topics=4)


class Mlp(nn.Module):
    def __init__(s, inp, width, out):
        super().__init__(); s.l1 = nn.Linear(inp, width); s.l2 = nn.Linear(width, width); s.l3 = nn.Linear(width, out)
    def forward(s, x): return s.l3(F.silu(s.l2(F.silu(s.l1(x)))))


class Tiny(nn.Module):
    """2 layers, width 32, 2 attention heads. Reads the task token plus up to 16 symbols; the answer is read at the task token."""
    def __init__(s, vocab=9, width=32, heads=2, layers=2):
        super().__init__(); s.tok = nn.Embedding(vocab, width); s.pos = nn.Embedding(SEQ_LEN + 1, width)
        s.blocks = nn.ModuleList(nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0., activation=F.silu, batch_first=True, norm_first=True) for _ in range(layers))
        s.norm = nn.LayerNorm(width); s.out = nn.Linear(width, 2)
    def forward(s, x):
        h = s.tok(x) + s.pos.weight[:x.shape[1]]; pad = x == 0
        for b in s.blocks: h = b(h, src_key_padding_mask=pad)
        return s.out(s.norm(h[:, 0]))


class MnistNet(nn.Module):
    def __init__(s): super().__init__(); s.l1 = nn.Linear(784, 32); s.l2 = nn.Linear(32, 10)
    def forward(s, x): return s.l2(F.silu(s.l1(x.flatten(1))))


class CifarNet(nn.Module):
    def __init__(s):
        super().__init__(); s.c1 = nn.Conv2d(3, 16, 3, padding=1); s.c2 = nn.Conv2d(16, 32, 3, padding=1); s.c3 = nn.Conv2d(32, 64, 3, padding=1); s.l = nn.Linear(1024, 10)
    def forward(s, x):
        for c in (s.c1, s.c2, s.c3): x = F.max_pool2d(F.silu(c(x)), 2)
        return s.l(x.flatten(1))


def net(name):
    if name == "seq": return Tiny()
    if name == "mnist": return MnistNet()
    if name == "cifar10": return CifarNet()
    return Mlp(INPUTS[name], TEXT_WIDTH if name == "topics" else WIDTH, OUTPUTS.get(name, 2))


def weights(m): return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for name in (*INPUTS, "seq", "mnist", "cifar10"): print(f"{name:13s} {weights(net(name)):7,d} weights")
