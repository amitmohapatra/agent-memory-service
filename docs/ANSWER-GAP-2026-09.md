# The accuracy that is left is not in retrieval

From `benchmark/results/locomo_judged_v6.json`, over the 214 answerable rows the judge
actually graded (the other 19 are the judge-failure defect fixed in 42c7f2d).

## Where the 38 misses are

| | count | share |
|---|---|---|
| retrieval had **all** the evidence and the answer was still wrong | **28** | 74% |
| evidence missing or partial | 5 | 13% |
| other / wrong instance | 5 | 13% |

Evidence recall is 0.9828, all-evidence recall 0.9742. **Retrieval is not the bottleneck.**
Three quarters of what is left is the step that turns retrieved evidence into an answer.

## What those 28 actually are

| failure | single_hop | multi_hop | total |
|---|---|---|---|
| abstained ("I don't know") with the evidence present | 6 | 4 | **10** |
| gold is a list of 2+ items; the answer gave one | 6 | 6 | **12** |
| wrong instance / judged paraphrase | 4 | 2 | 6 |

Examples of the second row, all with `evidence_all_hit=True`:

| question | gold | produced |
|---|---|---|
| What do Melanie's kids like? | dinosaurs, **nature** | "nature and hands-on creative..." |
| What does Melanie do to destress? | **Running**, pottery | "She goes running..." |
| What musical artists has Melanie seen? | **Summer Sounds**, Matt Patterson | "Summer Sounds" |

This is why multi_hop scores 0.594 against temporal's 0.950. It is not that multi-hop
retrieval is harder here - it found the evidence in 12 of its 13 misses. It is that
multi-hop gold answers are disproportionately *lists*, and a list answered with one item is
scored wrong under the strict ruler and **correct under Mem0's lenient one**, which accepts
"at least one correct item from a list answer". A meaningful part of the gap between our
strict number and a published lenient number is this single behaviour.

## The cause is a contradiction in the answer prompt

`ANSWER_SYSTEM` (benchmark/locomo.py) says:

> "For counting or listing questions, enumerate each distinct instance the context supports"

and then ends:

> "One short phrase or sentence."

The final instruction is the one the model follows, and it directly discourages returning two
items. The enumeration rule is also scoped to "counting or listing questions", which "What do
Melanie's kids like?" does not look like, so it very likely never fires on the rows that need
it.

## The abstention dial, which must be tuned with both numbers in view

The other ten are abstentions with the evidence present. It is tempting to simply loosen
that, and it would be wrong to do it blind, because abstention is also what earns the
adversarial score:

| | v5 | v6 |
|---|---|---|
| answerable (strict) | 0.8326 | 0.8224 |
| abstention on adversarial | 0.8592 | **0.9437** |

v6 traded about one point of answerable accuracy for eight and a half points of adversarial.
That is a single dial, and it was moved without being named. Any change here has to report
both numbers or it is just moving the loss somewhere the headline does not show it.

## Headroom, honestly

Fixing the list truncation is worth up to **+5.6 points** (12 of 214). Recovering half the
over-abstentions without giving back adversarial is worth roughly **+2.3**. That puts a
realistic ceiling near **0.87-0.90 strict** with adversarial held - which is also the clearest
statement yet of why >94% under a strict, human-aligned ruler is not reachable: after this,
what remains is a handful of genuinely missing evidence and genuine judge disagreement.

## What to do, in order

1. Do **not** change `ANSWER_SYSTEM` while the 1,986-question run is in flight; it is the
   baseline the change has to be measured against.
2. Then A/B one edit: resolve the completeness/brevity contradiction, and unscope the
   enumeration rule from "counting or listing questions". Report answerable **and**
   adversarial.
3. Only then consider the abstention dial, as an explicit two-number trade.

This is a change to the harness's answerer, not to the service, and it should be reported as
such. It is legitimate because the prompt contradicts itself and loses answers whose evidence
the system had already retrieved - but it is not a retrieval improvement, and calling it one
would be dishonest.
