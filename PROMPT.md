# The task brief (verbatim)

Reproduced in full and unedited, as required by the brief itself.

---

## This turn's mandated method

Build a unique, high-scoring submission TIF and a GitHub-Pages site for this repository.

The method to use, mandated for this turn:

> Multiscale worming + topological persistence — track edges surviving upward continuation on
> magnetic AND gravity layers, separately compute persistent homology on the same layers'
> gradient-magnitude surfaces, multiply continuation-steps-survived × topological birth-death range
> into one persistence score per pixel, normalize [0,1], write required format, confirm low
> correlation against every prior submission's raw output (output should be sparser/more spatially
> concentrated than a single-scale gradient map). Must be UNIQUE — no copying a previous submission
> except for learning/education.

## Standing deliverables

- Put the full prompt into the repo README.
- Produce 3–5 ranked NEW geological hypotheses — layer(s), physical signature, why it finds faults
  missing from the USGS/INGENIOUS catalogue, how it differs from anything already in the repo —
  validated on spatially-blocked holdout before spending a submission slot.
- The site must have the downloadable TIF and a "how to submit" executive summary at the very top.
- Explain why `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` (GEMSDOE32) scored 0.2778 and whether
  we can beat it.
- Work autonomously with no manual input.
- Verify line by line from official trusted sources with links.
- No hallucinations.
- Flag irregularities.
- Run 3 passes (implement → review/fix → re-check).
- Create a PR and merge to `main`.
- End with remaining-work suggestions and limitations.

## Standing constraints and corrections carried from earlier turns

- Must produce a UNIQUE submission; do not copy a previous submission unless for learning/education.
- Submission must be easy to download (click a file) and obvious at the very beginning / executive
  summary of the site.
- Predicted values must be in range [0,1] — a previously downloaded file was rejected with
  "Predicted values must be in range [0, 1]".
- Needs a unique submission name and a short distinguishing comment for the submission form.
- Data placement must be solved autonomously
  (`bash scripts/download_competition_data.sh` then `python scripts/prepare_data.py`) —
  "you need to complete the above task by yourself… no manual input".
- Verify line by line from official verified trusted sources; provide links for manual review;
  no hallucinations; flag irregularities.
- Validate on spatially-blocked holdout; do not spend a submission slot on an unvalidated idea.
- If a candidate needs new external data, name the specific free official source and confirm
  obtainability first.
- Known limitation stated by the user: no DrivenData auth → cannot auto-download
  `training_features.tif`, `labels.tif`, `sample_submission.tif`, `1m_DEM_links.csv` from the
  DrivenData data page.
- Multi-pass (3) review discipline required before finishing; create PR then merge to main.
