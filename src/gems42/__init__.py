"""GEMSDOE42 — cross-scale stability fault detector.

Multiscale "worming" (Hornby, Boschetti & Horowitz 1999; Archibald, Gow & Boschetti 1999)
multiplied by dim-0 persistent homology of gradient-magnitude surfaces across a sweep of
smoothing scales.  A candidate must be corroborated by two *independent* notions of scale:

1. how many upward-continuation steps a potential-field edge survives (source depth), and
2. how long the gradient ridge it sits on lives in the superlevel-set filtration (topology).

Neither term alone is a threshold on a single scale.
"""

__version__ = "0.1.0"
