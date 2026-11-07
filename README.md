# What an LLM can and can't do by writing network weights

Claude Opus 5.5 read a task's training examples and wrote every weight of a small network as literal numbers, with no training. On 14 of 16 tasks that have a rule, those weights scored higher on unseen inputs than the best of 34 training settings (91.3% against 83.3%). On CIFAR-10, which has no short rule, training won (87.8% against 71.6%).

This repository asks why, and where the limits are. [QUESTIONS.md](QUESTIONS.md) lists 12 questions, each with a result whichever way it turns out. We are using this to write a paper for ICML 2027.

![On the 16 toy tasks with a rule, normal training scored 83% on the hidden 30% of pairs and Opus 5.5 scored 91%, with no training](charts/result.png)

## Setup

```bash
pip install -r requirements.txt
python3 -m venv toy/.kernel && toy/.kernel/bin/pip install numpy scipy
cd toy && python3 search.py test
```

The last command checks every path of the bench in about 5 seconds and ends with `ALL PASS`. The sandbox that runs the LLM's Python works on macOS; the Linux path exists and has not been tested.

## Files

| File | What it does |
|---|---|
| `toy/net.py` | The network: two numbers go in, one of 8 classes comes out, 8,751 weights. |
| `toy/search.py` | The main bench. Makes a task, lets an LLM change the weights before, during, after or instead of training, and compares it with 34 training settings. |
| `toy/bigger.py` | The same on MNIST and CIFAR-10 with a larger network. |
| `toy/digits.py` | The same on small handwritten digits. |
| `toy/steps_saved.py` | How many training steps an edit saves. |
| `toy/read_rule.py`, `toy/cocluster.py`, `toy/data_gone.py` | Plain scripts with no LLM, as rivals. |
| `toy/search_runs/` | All 51 earlier Opus runs: the code each one wrote, what it got back, and its audit. |
| [RESULTS.md](RESULTS.md) | Every earlier table, the commands, and how to run an LLM as the editor. |
| [NOTES.md](NOTES.md) | The working log, with every rule and prediction written before each test. |

The licence is MIT.
