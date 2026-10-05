# Metric and submission format (verbatim sources)

## Sources (verified 2026-10-05)
- Metric + format: <https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/>
- About/resources: <https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/>
- Leaderboard: <https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/>
- Rules PDF: <https://docs.nlr.gov/docs/fy26osti/96647.pdf>
- Reference solution: <https://github.com/drivendataorg/gems-prize-reference-solution>

## Metric (DTI)
- k(d) = max(1 − d/300m, 0); TPw/FPw/FNw as on p.967; α=0.2, β=0.8.
- Worked example: 3.00/1.89/2.00 → 0.6027 (page prints 0.60, rounded).
- Identity: DTI = T/(0.2(T+S−M) + 0.8|G|); add mass iff k > 0.2·DTI.
- Leaderboard 2026-10-05: #1 nchuzhoy 0.3262 → credit bar 0.0652.

## Format
- Single-band float32 GeoTIFF in [0,1]; EPSG:32611; 100 m; same bounds;
  null/NaN outside. Portal accepts .tif or .zip containing one GeoTIFF.
- Grid measured here: 3730×3292, Affine(100, 0, 243350, 0, −100, 4508550),
  footprint 5,167,373, catalogue 60,988.
- The `"Predicted values must be in range [0, 1]"` error has two mechanisms:
  (a) float32 −3.4e38 sentinel written through; (b) NaN nodata on a strict
  validator. Primary ships all-finite, nodata=None, plus a NaN-outside twin.

## Structure
- Initial round ($50K): fixed private pre-competition hidden set.
- Final round ($250K): expanded set incl. expert-verified discoveries from
  all submissions. Same file scored twice; experts review submissions.
