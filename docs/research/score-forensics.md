# Score forensics: what the scores establish—and what they do not

**Evidence date:** 2026-10-05 UTC. The official [DrivenData leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) and [problem description](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) were retrieved directly for this review. The historic rows transcribed in [`../prior-results.csv`](../prior-results.csv) came from the user's prompt and are not independently authenticated to raster bytes.

## Answer to “why did H33-H33-2-B2 get 0.2778?”

We cannot responsibly state that the named H33-H33-2-B2 TIFF received 0.2778. There is a direct attribution conflict:

1. The published [GEMSDOE32 landing page](https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html) labels the H33-2-B2 TIFF “UNSCORED,” describes 0.2747 as a **model projection**, and says no organizer score exists for that artifact.
2. The live official leaderboard displays a **0.2778** public score for participant `extradr19` (shown at rank 13 in the fetched snapshot), but does not expose a TIFF filename or link that connects the participant row to the GEMSDOE32 artifact.
3. The prompt's score-to-file mapping is therefore an owner/user claim, not a verified organizer receipt. It might be a separately uploaded file, a different participant, or a stale record. The available public evidence cannot decide which.

**Required proof for an exact explanation:** a DrivenData submission receipt or export that identifies the participant, submission identifier, file name/hash and score, plus a byte-identical downloadable raster. Without that evidence, any claim that the named method caused 0.2778 is speculation.

## What a DTI score rewards

DrivenData defines a distance-weighted Tversky score using `alpha=0.2`, `beta=0.8`, and a triangular distance kernel with 300 m support (three 100 m pixels). In the page's notation:

\[
DTI = \frac{TP_w}{TP_w + 0.2\,FP_w + 0.8\,FN_w + \epsilon}.
\]

The true-positive and false-negative terms give each labeled fault pixel credit for the best nearby predicted confidence, attenuated linearly with distance. False-positive prediction mass is also distance-weighted: predictions near a labeled fault incur less penalty than predictions far from one. Consequently:

- A geometrically well-placed, narrow trace can outperform a broad field if it preserves useful distance-weighted coverage while reducing unhelpful predicted mass.
- The competition penalizes missed labels four times as heavily as false positives in the unweighted coefficients (`0.8 / 0.2 = 4`), so “sparser is always better” is **not** a valid rule. Removing high-value coverage can reduce the score.
- A public score is a fit to the public labeled subset, not proof that the predicted line is a fault, geothermal conduit, or new discovery. The challenge explicitly notes that existing labels may be incomplete or misaligned, and its later prize round uses an expanded expert-reviewed set.
- Scores alone do not reveal an algorithm, threshold, artifact identity, precision/recall tradeoff, or private-set generalization.

## Current official public leaderboard snapshot

Fetched 2026-10-05 UTC from the official page; it is a time-sensitive snapshot, not a standing rank guarantee:

| Rank in fetched page | Participant | Best public DW-Tversky |
|---:|---|---:|
| 1 | nchuzhoy | 0.3262 |
| 2 | kinghorton42 | 0.3222 |
| 3 | alexoktaba | 0.3220 |
| 4 | DARD | 0.3195 |
| 5 | joeyfezster | 0.3163 |
| 13 | extradr19 | 0.2778 |

Thus the user's 0.3195 was a real score in the fetched snapshot, but it was no longer the displayed leader. The official first place was 0.3262. The score 0.2778 appears on the same official board but is not linked there to H33-2-B2.

## What can be said about winning with scale persistence

It is scientifically plausible that a multi-scale edge persistence field can reject isolated, scale-unstable gradient noise. It is not yet shown that this increases *fault* precision or the competition DTI. Potential-field worm trajectories have non-unique geological interpretations; multiscale persistence is a structural proxy, not direct fault truth. The user's requested product also combines two transforms of the same gravity/magnetic input, so the two tests are independent **scale operators**, not independent measurements or statistically independent evidence.

A valid test should compare the new map with a frozen incumbent at matched prediction mass on spatially separated folds, and report each fold's official-metric proxy score, output mass, coverage, and uncertainty. It should also state that the known-fault catalogue is an imperfect proxy for the newly labeled faults. No such test can be run in this checkout because neither the authorized feature/label rasters nor the prior holdout baseline is present.

**Conclusion:** beating 0.2778 or 0.3195 is possible in principle; no available evidence supports a score forecast for this new method. The live public target to beat on the fetched date was 0.3262, while the actual competition objectives also include private testing and expert review.

## Official review links

- [Competition task, metric, and submission format](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
- [Current official leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)
- [September 2026 official rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf)
- [GEMSDOE32 owner-published page](https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html) — useful for investigating the attribution contradiction; not an organizer record.
