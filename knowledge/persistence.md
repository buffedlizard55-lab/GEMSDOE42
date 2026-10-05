# Topological persistence (scale-space form)

## Sources (verified 2026-10-05)
- Edelsbrunner, H., Letscher, D. & Zomorodian, A. (2002). "Topological
  persistence and simplification." *Discrete & Computational Geometry* 28.
  (Standard reference; birth/death/elder-rule definitions.)
- Med-TDA persistent-homology primer (birth/death in sub/superlevel
  filtrations): <https://medtda.readthedocs.io/en/latest/theory/persistent_homology.html>
- PixHomology distributed image-persistence algorithm (birth = relative
  maxima, death = saddle/merging value):
  <https://arxiv.org/html/2404.08245>

## Key results used here
1. **0-dim persistence of superlevel sets** tracks connected components from
   high to low: each local maximum is born at its value; when two components
   merge at a saddle, the younger dies (**elder rule**); persistence =
   birth − death (superlevel) measures significance.
2. Across a **smoothing sweep** instead of a level sweep, the same
   birth/death bookkeeping measures which maxima survive simplification of
   the surface — cross-scale topological stability.
3. Image-scale exact persistence (lower-star filtration over 12M cells) is
   computationally prohibitive here; discrete maxima tracking with the elder
   rule is the standard practical reduction and preserves the
   birth/death/persistence semantics per maximum.

## Our reduction to practice (`src/gems42/persistence.py`)
- Gaussian sigmas 0/1/2/4/8/16 px on the h=0 gradient-magnitude surfaces.
- 3×3 maxima above the per-scale median (permissive; strictness comes from
  survival, not thresholding).
- Greedy nearest-neighbour linking within 3 px, oldest-first (elder rule);
  persistence_steps = death − birth + 1, splatted at the birth pixel.
