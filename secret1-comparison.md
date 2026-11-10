# Same way or different way, on secret-1

`TOY_RULES=secret python3 compare_secret1.py` from `toy/`. Training is 600 steps at speed 1. The written network is Opus's step-450 edit, with no more training. Rule set 1: skill 0 is a sum, skill 1 a product, skill 2 a max, skill 3 an absolute difference.

| Held-out | skill 0 sum | skill 1 product | skill 2 max | skill 3 abs | mean |
|---|---|---|---|---|---|
| Training | .497 | .643 | .971 | .205 | .579 |
| Opus's edit | .795 | .857 | 1.000 | .214 | .717 |

A unit is counted when its activation changes across the 32×32 pairs of a skill and turning its outgoing weights off drops that skill. The first layer can only compute a score for `a` plus a score for `b`, because `a`, `b`, and the skill are one-hot. A product or a max has to appear in the second layer.

## Which units, and what each one computes

Opus turns a different set of second-layer units on for each skill. On the other skills those units are flat. Training keeps all 48 second-layer units active on every skill. They are copies of an additive score. The number after each unit is how much that skill's held-out accuracy falls when the unit is turned off.

**Skill 0, the sum.** Opus uses units 36, 9, and 43, which match the sum at correlation 1.00, .99, and .97 (drops .678, .633, .539). Units 5, 0, 12, and 2 match the product instead, at .97, .96, .94, and .88, and matter less (drops .211, .230, .077, .013). In the first layer, unit 7 is on for this skill only and matches the sum at 1.00. Training's accuracy sits on many sum copies. The ones that matter most are 41, 33, 8, 4, and 34, matching the sum at .86 to .91, and the largest drop is .104.

**Skill 1, the product.** Opus uses two kinds of unit. Units 15, 31, and 19 match a sum at .92 to .93 and carry the skill (drops .743, .636, .428). Units 3, 18, 6, and 27 match the product at .93 to .97 (drops .347, .217, .149, .019). In the first layer, unit 9 is on for this skill and matches a sum at .91. Training again reads sum copies: units 41, 33, 8, and 11 match the sum at .90 to .94. Unit 24 drops the most, .143, and matches the sum only at .48. Unit 15 does match the product at .90, and turning it off drops the skill by .046.

**Skill 2, the max.** Opus uses two units. Unit 28 is an additive score, correlation .63 with the true sum, and turning it off drops the skill by .851. Unit 23 is the closest to the max, correlation .40, and the drop is .422. In the first layer, units 34 and 23 are on for this skill and are additive scores (correlation .69 and .63 with the sum). Training also gets this skill from additive units, none of them shaped like the max. The largest drops are unit 2 (.237, correlation .13 with the absolute difference), unit 22 (.182, correlation .15 with the max), unit 20 (.169, correlation .58 with the sum), and unit 45 (.137, correlation .28 with the sum).

**Skill 3, the absolute difference.** Opus's units 8, 22, 24, 25, 32, and 33 move slightly and match nothing. The largest drop is .022. Training's largest drop is .046, on unit 33, whose best correlation with any form is .11. Both networks leave this rule unsolved.

Unit 47 in the written network is flat on every skill. Its outgoing weights are a fixed shift of the outputs. Turning it off drops skills 0, 1, and 2 by .665, .672, and .909.

## How each network solves the rules

Opus gives each skill its own units and switches the others off. The sum is three units that match the sum, and the accuracy of .795 falls apart if any of them is removed. The product is three sum-shaped units plus three product-shaped units, and the sum-shaped ones carry more of the .857. The max is two additive units, and the classes come out right (1.000) even though neither unit's map is the max. The absolute difference is the leftover trained units, at .214.

Training uses one pool of additive units for all four skills. The sum is in that pool many times over, so removing any one unit barely matters, and the skill still only reaches .497. The product is read from those same sum copies, which is why it reaches .643 with almost no unit shaped like a product. The max reaches .971 from the same kind of additive unit, spread across more units than Opus uses. The absolute difference is not in the pool.

The difference is that Opus's units are private to a skill and a few of them match the sum or the product exactly. Training's units are shared, additive, and interchangeable.
