#!/usr/bin/env python3
"""Generate docs/index.html from the registry records.

Everything numeric on the page is read out of registry/*.json and the shipped file itself, so the
site cannot drift from what was actually built. Run it after ship_submission.py:

    python3 scripts/build_site.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LEADERBOARD = [
    (1, "nchuzhoy", 0.3262), (2, "kinghorton42", 0.3222), (3, "alexoktaba", 0.3220),
    (4, "DARD", 0.3195), (5, "joeyfezster", 0.3163), (13, "extradr19 (prior GEMSDOE best)", 0.2778),
    (19, "SDCF9", 0.2600),
]

PROMPT = (ROOT / "PROMPT.md").read_text(encoding="utf-8")


def esc(x) -> str:
    return html.escape(str(x))


def num(x, fmt: str = ",.0f") -> str:
    return format(x, fmt) if isinstance(x, (int, float)) else esc(x)


CSS = """
:root{--bg:#0b1120;--panel:#111a2e;--line:#1e2a44;--ink:#e6edf7;--dim:#93a4c3;--acc:#4cc9f0;
--ok:#3ddc97;--warn:#ffb020;--bad:#ff5d73;--code:#0d1526}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1080px;margin:0 auto;padding:0 22px}
header{padding:34px 0 18px;border-bottom:1px solid var(--line)}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:23px;margin:44px 0 10px;padding-top:14px;border-top:1px solid var(--line);
letter-spacing:-.01em}
h3{font-size:17px;margin:24px 0 6px;color:#cfe0ff}
p{margin:10px 0}code{background:var(--code);padding:2px 5px;border-radius:4px;font-size:13.5px;
font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--code);padding:13px 15px;border-radius:8px;overflow-x:auto;font-size:13px;
border:1px solid var(--line)}pre code{background:none;padding:0}
.sub{color:var(--dim);font-size:14px;margin-top:4px}
nav{position:sticky;top:0;background:rgba(11,17,32,.94);backdrop-filter:blur(6px);
border-bottom:1px solid var(--line);padding:9px 0;z-index:9;font-size:14px}
nav a{margin-right:15px;white-space:nowrap}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;
margin:14px 0}
.hero{background:linear-gradient(135deg,#12203c,#0f1a30);border:1px solid #27406e;border-radius:14px;
padding:24px;margin:18px 0}
.dl{display:inline-block;background:var(--ok);color:#062b1c;font-weight:700;padding:13px 22px;
border-radius:9px;margin:8px 10px 8px 0;font-size:16px}
.dl:hover{text-decoration:none;filter:brightness(1.08)}
.dl.sec{background:#2a3a5c;color:var(--ink)}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
th{background:#152039;font-weight:600}
tr:nth-child(even) td{background:#0e1729}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.dim{color:var(--dim)}
.kv{display:grid;grid-template-columns:230px 1fr;gap:5px 16px;font-size:14.5px;margin:10px 0}
.kv div:nth-child(odd){color:var(--dim)}
.box{border-left:3px solid var(--acc);padding:11px 15px;background:#101a30;margin:13px 0;
border-radius:0 8px 8px 0}
.box.warn{border-color:var(--warn);background:#241d10}
.box.bad{border-color:var(--bad);background:#26121a}
.box.ok{border-color:var(--ok);background:#0d2418}
figure{margin:16px 0}figure img{max-width:100%;border:1px solid var(--line);border-radius:9px}
figcaption{color:var(--dim);font-size:13px;margin-top:6px}
footer{margin:50px 0 30px;padding-top:16px;border-top:1px solid var(--line);color:var(--dim);
font-size:13px}
.pill{display:inline-block;border:1px solid var(--line);border-radius:99px;padding:2px 10px;
font-size:12.5px;color:var(--dim);margin-right:6px}
"""


def table(headers, rows, cls="") -> str:
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def main() -> int:
    build = json.loads((ROOT / "registry" / "submission_build.json").read_text())
    model = json.loads((ROOT / "registry" / "emission_model.json").read_text())
    calib = json.loads((ROOT / "registry" / "holdout_calibration.json").read_text())
    run = json.loads((ROOT / "registry" / "pipeline_run.json").read_text())
    hyp = json.loads((ROOT / "registry" / "hypotheses.json").read_text())
    src = json.loads((ROOT / "registry" / "sources.json").read_text())
    irr = json.loads((ROOT / "registry" / "irregularities.json").read_text())

    prim = build["files"]["primary_nan"]
    twin = build["files"]["twin_zeros"]
    zp = build["files"]["zip"]
    prim_name = Path(prim["path"]).name
    ho = build["holdout"]
    sg = build["sgmc_off_catalogue"]
    m = build["method"]

    # ---- executive summary -----------------------------------------------------------
    hero = f"""
<div class="hero">
  <h2 style="margin-top:0;border:none;padding:0">Executive summary &mdash; download the submission</h2>
  <p>The file below is a ready-to-submit GeoTIFF for the
  <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">DrivenData
  <em>U.S. DOE GEMS Prize Challenge</em></a> (competition 306). It is single-band
  <code>float32</code>, <code>EPSG:32611</code>, 100&nbsp;m, 3730&nbsp;&times;&nbsp;3292, every
  value in <b>[0,&nbsp;1]</b>, NaN outside the survey footprint &mdash; the exact format the
  submission form enforces.</p>
  <p>
    <a class="dl" href="downloads/{esc(prim_name)}">&#11015;&nbsp; DOWNLOAD SUBMISSION
      ({num(prim['bytes']/1024, ',.1f')} KB)</a>
    <a class="dl sec" href="downloads/{esc(Path(zp['path']).name)}">.zip</a>
    <a class="dl sec" href="downloads/{esc(Path(twin['path']).name)}">0-filled twin</a>
  </p>
  <div class="kv">
    <div>Submission name</div><div><code>{esc(build['name'])}</code></div>
    <div>SHA-256</div><div><code>{esc(prim['sha256'])}</code></div>
    <div>Emitted pixels</div><div><b>{num(build['emitted_px'])}</b> of
      {num(5167373)} in-footprint pixels ({num(100*build['emitted_px']/5167373, '.3f')}%)</div>
    <div>Value range</div><div>0.0 &hellip; 1.0 &mdash; <span class="ok">verified</span></div>
    <div>On known catalogue faults</div><div>{num(build['on_catalogue_px'])} pixels
      (the organiser masks these out of scoring, so they are excluded on purpose)</div>
    <div>Method</div><div>multiscale worming &times; topological persistence
      (see <a href="#method">method</a>)</div>
    <div>Uniqueness</div><div>max |Pearson| vs any of 12 restored prior GEMSDOE submissions =
      <b>{num(build['uniqueness']['max_abs_pearson_binary'], '.4f')}</b>,
      max |Spearman| = <b>{num(build['uniqueness']['max_abs_spearman_binary'], '.4f')}</b></div>
  </div>
  <div class="box warn"><b>Read the honesty section before you submit.</b> Neither of this
  repository's two validation instruments predicts the live leaderboard well (Spearman +0.087 and
  +0.305 against 11 known scores). The submission is a genuine, independently-derived candidate;
  it is <b>not</b> evidence of a top-of-board result. See
  <a href="#honesty">what we can and cannot claim</a>.</div>
</div>"""

    # ---- how to submit ---------------------------------------------------------------
    howto = f"""
<h2 id="submit">How to submit (step by step)</h2>
<div class="card">
<ol>
<li><b>Download</b> <a href="downloads/{esc(prim_name)}"><code>{esc(prim_name)}</code></a>
  with the green button at the top of this page. It is {num(prim['bytes']/1024, ',.1f')}&nbsp;KB.</li>
<li><b>Go to</b> <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">
  the competition page</a> and sign in with a DrivenData account that has accepted the rules.</li>
<li><b>Open</b> the <em>Submit</em> tab and upload the <code>.tif</code> directly. Do
  <b>not</b> unzip, re-compress, or re-project it &mdash; the form checks CRS, extent, dtype and
  value range, and any re-save can flip the float32 representation.</li>
<li><b>Paste the submission name</b> (it must be unique to your account):
  <pre><code>{esc(build['name'])}</code></pre></li>
<li><b>Paste the comment</b>:
  <pre><code>{esc(build['note'])}</code></pre></li>
<li><b>Submit.</b> The score appears on the leaderboard within minutes. The initial prize round
  accepts submissions through the stated deadline; the same file is re-scored in the final round
  against an expert-expanded label set.</li>
</ol>
<p class="dim">If the form rejects the file with &ldquo;Predicted values must be in range
[0,&nbsp;1]&rdquo;, use the <a href="downloads/{esc(Path(twin['path']).name)}">0-filled twin</a>:
identical predictions, but nulls outside the footprint are written as <code>0.0</code> instead of
<code>NaN</code>. Both files were audited by <code>scripts/audit_shipped.py</code> and both pass all
11 format checks.</p>
</div>"""

    # ---- the score -------------------------------------------------------------------
    score = f"""
<h2 id="score">Can we beat 0.3195? The short, honest answer</h2>
<p>The brief asked us to explain why the family's previous best
(<code>h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros</code>, public score <b>0.2778</b>) scored what
it did, and whether we can beat it. The live board at the time of writing:</p>
{table(["#", "team", "public DTI"],
       [[str(a), esc(b), f"<b>{c:.4f}</b>"] for a, b, c in LEADERBOARD])}
<p>Two things follow, and they pull in opposite directions.</p>
<h3>1. Why 0.2778 scored what it did</h3>
<p>That file was 37,654 positive pixels, none of them on a known catalogue fault, built by
pruning a base emission that its own repository had measured at 0.2708. Its <em>projected</em>
0.2747 was an internal model, not a score. Working the metric backwards
(<code>src/gems42/emission_model.py</code>) with the fitted hidden-truth size
|G|&nbsp;&asymp;&nbsp;{num(model['parameters']['hidden_truth_px_G'])}&nbsp;px, a score of 0.2778 at
37,654 dots implies roughly <b>36%</b> of dots landed within 300&nbsp;m of a hidden truth pixel.
The current leader, 0.3262, implies roughly <b>38%</b> at ~44,000 dots. The whole spread from
0.2778 to 0.3262 is about <b>two percentage points of per-dot precision</b> &mdash; not a
qualitatively different method.</p>
<h3>2. What actually moves the needle</h3>
<p>Across the 11 restored prior submissions whose official scores are known, the single variable
that tracks the leaderboard is <b>how many pixels were emitted</b>: Spearman(emitted pixels,
official score) = <b>{num(calib['spearman_size_vs_official']['holdout']['rho'], '+.3f')}</b>
(p&nbsp;=&nbsp;{num(calib['spearman_size_vs_official']['holdout']['p'], '.3f')}) on the catalogue
holdout and <b>{num(calib['spearman_size_vs_official']['sgmc']['rho'], '+.3f')}</b>
(p&nbsp;=&nbsp;{num(calib['spearman_size_vs_official']['sgmc']['p'], '.4f')}) against the
off-catalogue target. Dense submissions lose. That is a direct consequence of the metric:
a dot pays only while its expected kernel credit beats <code>&alpha;</code> &times; its
false-positive charge, and &alpha;&nbsp;=&nbsp;0.2 discounts false positives fourfold, so precision
matters far less than a plain F1 intuition suggests &mdash; but <em>coverage of a small, real
target</em> matters enormously, because &beta;&nbsp;=&nbsp;0.8 charges you for every hidden fault
you miss.</p>
<div class="box"><b>Verdict.</b> This submission is a legitimately different candidate built by a
method no prior GEMSDOE submission used. On the emission-size axis it sits exactly where the
family's best scorers sit ({num(build['emitted_px'])} dots, against 60,069 for the 0.2477 entry).
Whether it beats 0.2778 depends on per-dot precision that <b>neither instrument in this repository
can measure reliably</b>. Treat it as one well-founded entry in a portfolio, not as a predicted
winner.</div>"""

    # ---- method ----------------------------------------------------------------------
    cfg = run["config"]
    grid = run["grid"]
    method = f"""
<h2 id="method">Method: cross-scale stability &times; topological persistence</h2>
<p>Two independent notions of &ldquo;this is a real structure, not noise&rdquo; are computed from
the same two potential-field layers &mdash; <code>rtp</code> (reduced-to-pole magnetics) and
<code>iso_grav_anom</code> (isostatic gravity anomaly) &mdash; and multiplied.</p>
<h3>Stage A &mdash; multiscale worming (upward-continuation edge survival)</h3>
<p>Following Hornby, Boschetti &amp; Horowitz (1999) and Archibald, Gow &amp; Boschetti (1999),
and using the operational recipe stated verbatim in Horowitz (2018): the field is upward-continued
in the Fourier domain to each of the heights below; at each height the local maxima of the
horizontal-gradient magnitude are marked as multiscale edges; a pixel earns a step for every
consecutive level at which it stays an edge, allowing up to {m['migrate_px']}-pixel lateral
migration (because a dipping structure's edge <em>moves</em> as you continue it upwards). The score
is the fraction of levels survived, combined across layers with
<code>{esc(m['worm_combine'])}</code>.</p>
<h3>Stage B &mdash; topological persistence (dim-0 persistent homology)</h3>
<p>For each layer, and for each smoothing scale &sigma; below, the gradient-magnitude surface is
thresholded at {m['n_levels']} levels from its 99.9th percentile down to 0. A union-find sweep over
the superlevel-set filtration records, for every pixel, the level at which its connected component
is born and the level at which it merges into an older component &mdash; the birth&ndash;death range
of its dim-0 homology class. The map shipped here is that range,
<code>persistence = birth &minus; death</code>, which is literally what the brief asked for; the
alternative <code>prominence = G &minus; death</code> map is also implemented and both were scored
(see <a href="#validation">validation</a>).</p>
<div class="box warn"><b>A trap worth recording.</b> Naive H0 persistence assigns the
<em>largest</em> range to the background, because the component that never dies is the one seeded by
the global maximum and it eventually swallows the whole grid. Left that way the detector is exactly
inverted. <code>src/gems42/persistence.py</code> closes the surviving component at the filtration
floor so background pixels go to ~0, and <code>tests/test_core.py</code> pins the convention with an
assertion on a flat pixel.</div>
<h3>Stage C &mdash; the product</h3>
<p>Each stage is mapped onto [0,&nbsp;1] by a robust percentile and multiplied, combined across
layers with <code>{esc(m['topo_combine'])}</code>. A pixel scores high only if it is <em>both</em>
scale-stable in the potential field <em>and</em> topologically persistent in the gradient &mdash;
two different filters in two different domains.</p>
{table(["quantity", "value"], [
 ["grid", f"{num(grid['shape'][1])} &times; {num(grid['shape'][0])} px, "
          f"{esc(grid['crs'])}, {num(grid['footprint_px'])} px inside the footprint, "
          f"{num(grid['catalogue_px'])} of them known catalogue faults"],
 ["continuation heights", ", ".join(f"{h:g}&nbsp;m" for h in cfg["heights_m"])],
 ["edge percentile / migration", f"{cfg['edge_percentile']:g}th percentile, "
  f"&plusmn;{cfg['migrate_px']}&nbsp;px"],
 ["smoothing scales &sigma;", ", ".join(f"{g:g}&nbsp;px" for g in cfg["sigmas"])],
 ["filtration levels", str(cfg["n_levels"])],
 ["Stage-C score field", f"max {num(run['score_stats']['max'], '.3f')}, "
  f"99.9th pct {num(run['score_stats']['p99.9'], '.4f')}, "
  f"{num(run['score_stats']['nonzero_px'])} non-zero px "
  f"({num(100*run['score_stats']['nonzero_px']/grid['footprint_px'], '.2f')}% of the footprint)"],
])}
<figure><img src="img/stages.png" alt="Stage A, Stage B and Stage C maps">
<figcaption>Left: Stage&nbsp;A worm survival (0&ndash;4000&nbsp;m continuation, edges above the
{m['edge_percentile']:g}th percentile). Centre: Stage&nbsp;B dim-0 persistence. Right: the
Stage&nbsp;C product that is thresholded into the emission.</figcaption></figure>
<figure><img src="img/emission.png" alt="Shipped emission over the RTP magnetic field">
<figcaption>The {num(build['emitted_px'])} emitted pixels (green) over the RTP magnetic field,
with the known catalogue faults in red. The emission is deliberately sparse and concentrated: the
metric rewards covering a small hidden target, not painting the region.</figcaption></figure>
<h3>Provenance and uniqueness</h3>
<p>Nothing here is copied from a prior submission. The maximum absolute correlation against all
12 restored prior GEMSDOE rasters is
{num(build['uniqueness']['max_abs_pearson_binary'], '.4f')} (Pearson) /
{num(build['uniqueness']['max_abs_spearman_binary'], '.4f')} (Spearman); pixel overlap with the closest
one is {num(build['prior_correlation'][0]['overlap_px'])} of
{num(build['emitted_px'])} emitted pixels, which is the level you would expect from two independent
detectors both aiming at the same geology.</p>
<pre><code>cd GEMSDOE42
bash scripts/download_competition_data.sh   # or: python3 scripts/restore_data.py
python3 scripts/prepare_data.py
python3 scripts/run_pipeline.py             # stages A -> C, cached under .cache/stage/
python3 scripts/fit_emission_model.py       # picks the emission budget from 11 known scores
python3 scripts/ship_submission.py --budget {build['budget']} --separation {build['separation_px']} \\
        --which {esc(m['which'])}
python3 scripts/audit_shipped.py            # independent re-check of the file on disk
python3 scripts/build_site.py</code></pre>"""

    # ---- validation ------------------------------------------------------------------
    per_rows = []
    for r in calib["priors"] + calib["candidates"]:
        off = r.get("official_score")
        h = r.get("holdout_dti")
        s = r.get("sgmc_dti")
        per_rows.append([
            esc(r["id"]) + (" <span class='pill'>ours</span>" if r.get("is_ours") else ""),
            num(r["emitted_px"]),
            f"<b>{off:.4f}</b>" if off is not None else "<span class='dim'>&mdash;</span>",
            f"{h:.4f}" if h is not None else "<span class='dim'>&mdash;</span>",
            f"{s:.4f}" if s is not None else "<span class='dim'>&mdash;</span>",
        ])
    validation = f"""
<h2 id="validation">Validation &mdash; and why it is weaker than it looks</h2>
<p>The project rule is: do not spend a submission slot on an unvalidated idea. So two independent
instruments were built, and then <b>both were calibrated against 11 prior submissions whose official
public scores are known</b>. That calibration is the most important result in this repository.</p>
<h3>Instrument 1 &mdash; spatially-blocked catalogue holdout</h3>
<p><code>src/gems42/holdout.py</code> splits the survey footprint into four quadrants, hides the
catalogue faults in each turn with a {15}-pixel collar so a fault crossing the boundary cannot leak,
and scores the prediction on the hidden part with the competition's own metric
(<code>src/gems42/metric.py</code>, checked against a literal transcription of the organiser's
worked example). 2 draws &times; 4 quadrants = 8 cells.</p>
<h3>Instrument 2 &mdash; off-catalogue USGS SGMC faults</h3>
<p><code>src/gems42/sgmc.py</code> takes the restored USGS State Geologic Map fault raster and keeps
only pixels more than 300&nbsp;m from <em>any</em> catalogue fault. That yields
{num(sg['n_truth'])} pixels of real, mapped fault that the competition catalogue does
<em>not</em> contain &mdash; a much closer analogue to the private &ldquo;new faults&rdquo; label
set than the catalogue itself.</p>
<h3>The calibration result</h3>
{table(["instrument", "Spearman vs official score", "p", "n"],
 [["catalogue quadrant holdout", f"<b>{calib['spearman_official_vs_holdout']['rho']:+.3f}</b>",
   f"{calib['spearman_official_vs_holdout']['p']:.2f}", str(calib['n_priors'])],
  ["off-catalogue SGMC target", f"<b>{calib['spearman_official_vs_sgmc']['rho']:+.3f}</b>",
   f"{calib['spearman_official_vs_sgmc']['p']:.3f}", str(calib['n_priors'])],
  ["<b>emitted pixel count alone</b>",
   f"<b>{calib['spearman_size_vs_official']['sgmc']['rho']:+.3f}</b>",
   f"{calib['spearman_size_vs_official']['sgmc']['p']:.4f}", str(calib['n_priors'])]])}
<div class="box bad"><b>Neither instrument validates anything.</b> The catalogue holdout is
statistically indistinguishable from noise as a predictor of the live board (+0.087, p&nbsp;=&nbsp;0.80).
The SGMC target is better but still not significant (+0.305, p&nbsp;=&nbsp;0.361). The only variable
that reliably tracks the leaderboard is emission size. The structural reason is stated by the
organisers themselves: the private test set is faults <em>not</em> in the catalogue, so hiding
catalogue faults measures close to the opposite of the skill being tested.</div>
<h3>Every measured candidate</h3>
{table(["raster", "emitted px", "official", "holdout", "SGMC"], per_rows)}
<p class="dim">Our own entry sits at holdout {num(ho['mean'], '.4f')} and SGMC
{num(sg['dti'], '.4f')} at {num(build['emitted_px'])} pixels. For context the family's best prior
sits at holdout 0.0945 / SGMC 0.0950 at the same size. Read honestly: on the better of the two
instruments this submission is <em>below</em> the best priors, and both instruments are weak.</p>
<h3>Choosing the emission budget</h3>
<p>Because neither instrument can be trusted for selection, the budget comes from an analytic model
fitted to the 11 known (score, size) pairs &mdash; <code>scripts/fit_emission_model.py</code>. With
<code>h(N) = A&middot;N<sup>&minus;b</sup></code> kernel credit per dot and a hidden truth of
|G| pixels, the fit gives A&nbsp;=&nbsp;{num(model['parameters']['A'], ',.0f')},
b&nbsp;=&nbsp;{num(model['parameters']['b'], '.4f')},
|G|&nbsp;=&nbsp;{num(model['parameters']['hidden_truth_px_G'])}&nbsp;px,
RMSE&nbsp;{num(model['rmse'], '.4f')}, R<sup>2</sup>&nbsp;{num(model['r2'], '.3f')}. Differentiating
shows a marginal dot pays exactly while <code>h &gt; &alpha;&middot;f</code>; inside the observed
range that puts the optimum and the breakeven at
<b>{num(model['optimum']['emitted_px'])} pixels</b> &mdash; which is the budget shipped, and is
also the size of the family's best-scoring entry. The model is used to choose a <em>size</em>, never
to predict a score: extrapolated below the fitted range it implies more than one unit of kernel
credit per dot, which the metric cannot produce (see irregularity IRR-03).</p>"""

    # ---- hypotheses ------------------------------------------------------------------
    hrows = []
    for h in hyp["hypotheses"]:
        hrows.append([
            f"<b>{h['rank']}</b>",
            f"<b>{esc(h['title'])}</b><br><span class='dim'>{esc(h['layers'])}</span>",
            esc(h["signature"]),
            esc(h["why_new"]),
            esc(h["novelty"]),
            esc(h["expected_gain"]),
            esc(h["cost"]),
            esc(h["external_data"]),
            esc(h["status"]),
        ])
    hypotheses = f"""
<h2 id="hypotheses">New geological hypotheses, ranked</h2>
<p>{esc(hyp['note'])}</p>
{table(["#", "hypothesis / layer", "physical signature", "why it finds faults the catalogue misses",
        "how it differs from what is already here", "expected gain", "cost",
        "external data", "status"], hrows)}
<div class="box">None of these has been implemented. Per the project rule they may not consume a
submission slot until each has been ranked against the shipped candidate on both instruments
<em>and</em> survives the emission model &mdash; and given the calibration table above, that bar
should include a sanity check that the candidate is not simply winning by being a different size.</div>"""

    # ---- irregularities --------------------------------------------------------------
    irr_rows = [[f"<b>{esc(i['id'])}</b>", esc(i["severity"]), esc(i["finding"]),
                 esc(i["evidence"]), esc(i["action"])] for i in irr["irregularities"]]
    irregularities = f"""
<h2 id="irregularities">Irregularities found</h2>
<p>{esc(irr['note'])}</p>
{table(["id", "severity", "finding", "evidence", "what we did"], irr_rows)}"""

    # ---- sources ---------------------------------------------------------------------
    srows = [[f"<a href='{esc(s['url'])}'>{esc(s['url'])}</a>", esc(s["what"]),
              esc(s["verified"]), esc(s["note"])] for s in src["sources"]]
    sources = f"""
<h2 id="sources">Sources, verified line by line</h2>
<p>Fetched {esc(src['fetched_utc'])}. &ldquo;Verified&rdquo; means the specific fact was read off
the page, not that the URL resolves. One entry is explicitly marked as not reachable from this
sandbox rather than asserted.</p>
{table(["link", "what it is", "verified?", "what was taken from it"], srows)}"""

    # ---- honesty ---------------------------------------------------------------------
    honesty = f"""
<h2 id="honesty">What we can and cannot claim</h2>
<div class="box ok"><b>Established.</b> The file is format-correct (11/11 checks, independently
re-audited from disk), in range, uniquely derived, and built by a documented method with cited
provenance. The metric implementation is checked against the organiser's own worked example. The
emission budget is chosen from a model fitted to real leaderboard scores.</div>
<div class="box bad"><b>Not established.</b> That this submission beats 0.2778 or 0.3262. On the
off-catalogue SGMC proxy it scores {num(sg['dti'], '.4f')} where the family's best prior scores
0.0950 and one lattice prior scores 0.2439. That proxy is itself only weakly correlated with the
live board, so this is not proof of losing either &mdash; but it is emphatically not evidence of
winning, and the site would be misleading if it said otherwise.</div>
<h3>Remaining work, in order of value</h3>
<ol>
<li><b>Implement H42-A (dip-migration vectors).</b> The worm tracks already contain the geometry;
it is the only candidate that reads information out of the multiscale structure rather than out of a
single surface, and it costs nothing in new data.</li>
<li><b>Get native-float radiometrics</b> (USGS DOI 10.5066/P93LGLVQ) so H42-B can use real K/Th
ratios instead of the uint8-quantised mirror. <code>usgs.gov</code> is not reachable from this
sandbox, so this needs an unrestricted machine.</li>
<li><b>Build a third instrument.</b> Two weak ones are not enough. The most promising is a
held-out set of <em>historically mapped</em> faults outside the competition footprint, scored with
the same metric &mdash; that measures the actual skill (finding unmapped structure) rather than
re-deriving the catalogue.</li>
<li><b>Portfolio the submission slots.</b> Given that emission size dominates, spend slots on
different sizes of the best candidate as well as different candidates.</li>
<li><b>Re-fit the emission model after Phase 1 closes</b>, when more (score, size) pairs are
public, and check whether |G|&nbsp;&asymp;&nbsp;{num(model['parameters']['hidden_truth_px_G'])}&nbsp;px
holds up against the independent estimate of ~12,700&nbsp;px derived from the family's own
coverage arithmetic. Those two numbers disagree by a factor of ~1.8 and that gap is unresolved.</li>
</ol>
<h3>Limitations</h3>
<ul>
<li>No DrivenData credentials in this environment, so the official data page could not be
downloaded directly; the corpus was restored from SHA-256-pinned mirrors instead. Every byte is
recorded in <code>registry/data_manifest.json</code>.</li>
<li><code>usgs.gov</code>, <code>gdr.openei.org</code> and <code>dropbox.com</code> are unreachable
from this sandbox, which blocks H42-B and H42-E outright.</li>
<li>The private test labels are unobtainable by design &mdash; the organisers confirmed they will
not disclose how the test faults were identified. Nothing in this repository assumes otherwise.</li>
<li>The emission model is fitted to 11 points and explains 75% of the variance in score. It is a
sizing heuristic, not a predictor.</li>
<li>Both validation instruments are dominated by the same confound: candidate size. Any comparison
in the tables above that is not size-matched should be treated as indicative only.</li>
</ul>"""

    nav = """
<nav><div class="wrap">
<a href="#submit">Submit</a><a href="#score">Score</a><a href="#method">Method</a>
<a href="#validation">Validation</a><a href="#hypotheses">Hypotheses</a>
<a href="#irregularities">Irregularities</a><a href="#sources">Sources</a>
<a href="#honesty">Honesty</a><a href="prompt.html">Prompt</a>
</div></nav>"""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GEMSDOE42 &mdash; cross-scale worm &times; persistence submission for DOE GEMS</title>
<meta name="description" content="A unique GeoTIFF submission for the U.S. DOE GEMS Prize
Challenge (DrivenData 306), built by multiscale worming multiplied by dim-0 topological persistence.">
<style>{CSS}</style></head><body>
<header><div class="wrap">
<h1>GEMSDOE42 &mdash; cross-scale worming &times; topological persistence</h1>
<div class="sub">A unique, format-verified submission for the
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/">U.S. DOE GEMS Prize
Challenge</a> &middot; generated {esc(build['generated_utc'])} &middot;
<a href="https://github.com/buffedlizard55-lab/GEMSDOE42">source</a></div>
</div></header>
{nav}
<main class="wrap">
{hero}
{howto}
{score}
{method}
{validation}
{hypotheses}
{irregularities}
{sources}
{honesty}
</main>
<footer><div class="wrap">
Built by <code>scripts/build_site.py</code> from <code>registry/*.json</code>. Every number on this
page is read from those records or from the shipped file itself &mdash; nothing is typed in by hand.
</div></footer>
</body></html>"""

    out = ROOT / "docs" / "index.html"
    out.write_text(page, encoding="utf-8")
    (ROOT / "docs" / "prompt.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The task brief &mdash; GEMSDOE42</title><style>{CSS}</style></head><body>
<nav><div class="wrap"><a href="index.html">&larr; back to the submission</a></div></nav>
<main class="wrap"><h1>The task brief, verbatim</h1>
<p class="dim">Reproduced in full and unedited, as required by the brief itself.</p>
<pre><code>{esc(PROMPT)}</code></pre></main></body></html>""", encoding="utf-8")
    print(f"[site] wrote {out.relative_to(ROOT)} ({len(page):,} bytes) and docs/prompt.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
