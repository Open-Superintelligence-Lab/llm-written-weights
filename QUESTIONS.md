# Questions to pick

You can choose any strong LLM as the editor. I recommend Anthropic's or OpenAI's.

Pick one question, tell Vuk which one, and post your results within one week. Write how you will run it before you run it: which tasks, which baseline, and what result would mean yes or no. Every question has a useful answer whichever way it turns out.

[RESULTS.md](RESULTS.md), section "Play an editor", shows how to run an LLM as the editor with `loop.py`. Use a fresh LLM with no tools, so that it can only change the network through the bench.

| # | Question | Start from |
|---|---|---|
| 1 | Do the LLM and training solve the same task the same way? Look inside both networks and compare what each part does. | `search.py` answers runs in `toy/search_runs/`, `net.py`; a 2–5 hour version with both networks saved is in [`toy/suite/compare/`](toy/suite/compare/README.md) |
| 2 | What does training change in the weights the LLM wrote? The LLM writes the weights of a small network, then we keep training that network and see which of the LLM's numbers training keeps and which it overwrites. | `no_training.py`, `steps_saved.py` |
| 3 | Which tasks can the LLM write weights for, and which can't it? Make a map of where writing works and where it fails. | `search.py` task kinds, `digits.py`, `bigger.py` |
| 4 | What kinds of damage can the LLM fix? Break different parts of a trained network on purpose and ask the LLM to repair each one. | `search.py` after mode |
| 5 | Which kinds of network can the LLM figure out just by reading the weights? Give it networks trained on different tasks and check if it can say what each one does. | `read_rule.py`, `data_gone.py` |
| 6 | Does a bigger network make writing weights easier or harder? Give the LLM the same task in a small network and in one 10 times bigger. | `WIDTH` in `net.py`, `bigger.py` |
| 7 | On which unusual inputs does a written network hold up better than a trained one, and on which does it break first? | `search.py` unseen and noisy kinds, `robust.py` |
| 8 | How small can the LLM make a network that still solves the task, compared with the smallest network training can find? | `WIDTH` in `net.py` |
| 9 | Do the LLM's network and the trained network get the same inputs wrong? | `no_training.py`, `cocluster.py` |
| 10 | How well can the LLM predict a network's accuracy just by reading its weights, before anyone tests it? | `search.py` controls |

## What each answer tells us

1. **Same way:** the LLM found what training finds, only faster. **Different way:** there is more than one good solution, and training misses the one the LLM writes.
2. **Training keeps the LLM's numbers:** the LLM's start is already close to what training wants. **Training overwrites them:** the head start helps only through the first steps.
3. The map itself is the result: a list of task properties that predict whether writing beats training.
4. **Fixes everything:** reading weights is enough to repair them. **Fixes only some kinds:** those kinds show what the LLM can and can't see in weights.
5. **Reads most kinds:** weights alone are a usable description of a network. **Reads only some:** those are the networks whose working can be read.
6. **Easier:** extra room lets the LLM write rules without packing them tightly. **Harder:** writing does not scale, and training's lead grows with size.
7. Either way, it gives a list of inputs where each method breaks, which no single accuracy number shows.
8. **The LLM's is smaller:** training wastes size. **Training's is smaller:** the LLM writes bulky but readable solutions.
9. **Same mistakes:** those inputs are hard for any method. **Different mistakes:** combining both networks should beat either one.
10. **Good predictions:** the LLM can check networks without running them. **Bad predictions:** reading weights is weaker than it looks.

## Before you start

- `cd toy && python3 search.py test` must end with `ALL PASS` in about 5 seconds.
- Every earlier run, with the code the LLM wrote and what it got back, is in `toy/search_runs/`. Read two of them first.
- Write your rule for a win and your prediction in [NOTES.md](NOTES.md) before the run, as the earlier entries do.
