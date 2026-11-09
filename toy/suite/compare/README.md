# Trained network vs Opus-written network

**Question (2–5 hours):** on the 3-number rule, how do the network Opus wrote and the trained network differ, and how does each one solve the rule? Analyze the weights, ask an AI to help you read them, and write a short writeup of the two networks and their differences.

Both networks are the same shape and saw the same 300 training examples (10%). Opus wrote every weight by hand with no training and scores 100% on the 900 held-out examples. Training (best of 9 Adam settings) scores 72.3%.

The hidden rule: the answer is 1 when `(a + b) mod 16` is above `c // 2`. Opus found it from the examples and built it in two steps: 14 first-layer units compute steps of `a + b` and of `2(a + b) − c`, and 4 second-layer units combine them. Every other unit is zero. The trained network uses all 48 units in both layers.

## Files

`rule3/opus.npz` and `rule3/trained.npz` hold the weights by name (`l1.weight` is 48 × 96, and so on). The input is `a`, `b` and `c`, each one-hot over 32 places, in that order. The network is `l1 → silu → l2 → silu → l3`, and the answer is the larger of the 2 outputs.

```python
import numpy as np
w = np.load("rule3/trained.npz")          # or rule3/opus.npz
silu = lambda x: x / (1 + np.exp(-x))
def run(x): return silu(silu(x @ w["l1.weight"].T + w["l1.bias"]) @ w["l2.weight"].T + w["l2.bias"]) @ w["l3.weight"].T + w["l3.bias"]
```

`python3 ../compare.py check` loads every file and checks it scores what the results say. `make("rule3", .1)` in `../data.py` gives the training and held-out examples.

## Steps

1. Run both networks on all 32,768 inputs. Where does the trained one disagree with the rule? Draw a grid of `(a + b) mod 16` against `c` for each network.
2. Look at each first-layer unit's weights over `a`, `b` and `c` (32 each). Which units does each network use, and what does each unit compute? Does any trained unit respond to `a + b`, the way Opus's units do?
3. Ask an AI (Claude, ChatGPT or another) to help you read the weights: paste a unit's weights or your plots and ask what it computes and how the two networks solve the rule in different ways. Check its answers by running the network.
4. Write a short writeup: how each network solves the rule, how they differ, and one chart.

**What each answer means:**
- **Same way, only less exact:** training was heading for Opus's solution and ran out of examples.
- **Different way:** training learned something else that fits the 300 examples, and the held-out mistakes show where it breaks.

The other 9 tasks are saved the same way in the folders next to this one, if you want a second task.
