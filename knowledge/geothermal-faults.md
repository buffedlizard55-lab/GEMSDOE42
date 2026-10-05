# Why hidden faults show in magnetics + gravity (Great Basin)

## Sources (verified 2026-10-05)
- GEMS about page (fault primer; gravity/magnetic surveys infer faults;
  faults as fluid conduits):
  <https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/>
- USGS GeoDAWN surveys (aeromag + radiometric for subsurface structure):
  <https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and>
- GDR 1391 INGENIOUS regional compilation (DOI 10.15121/1881483):
  <https://gdr.openei.org/submissions/1391>
- GBCGE INGENIOUS project: <https://gbcge.org/current-projects/ingenious/>

## Geological reasoning (why H42-1 targets these layers)
1. **Normal-fault systems** (Basin-and-Range extension) juxtapose basement
   against basin fill → density contrast → **gravity gradient** across the
   fault, even where the scarp is buried or eroded (no surface trace to map).
2. Faults host **alteration + fluid flow** that destroys magnetite
   (demagnetisation) or juxtaposes volcanic units → **magnetic gradient**;
   RTP centers the anomaly over the source for interpretation.
3. **Buried faults are exactly the hidden set**: the competition's test
   faults were manually identified by experts as missing from USGS; INGENIOUS
   exists to find hidden geothermal systems. Surface-mapped faults are the
   training labels; geophysical edges are the discovery channel.
4. **Why cross-scale, not single-scale:** sedimentary/volcanic cover,
   survey-line noise, and shallow clutter all produce strong single-scale
   gradients. A through-going fault is a crustal-scale discontinuity: its
   edge persists upward (deep root) and across smoothing (coherent ridge),
   while clutter does not. The holdout PRIMARY result (+0.0125, 4/4) is the
   empirical counterpart of this argument.
