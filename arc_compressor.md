hand-coded VAE decoder that reconstructs ARC grid patterns using a special structure called a multitensor system.


The goal is:

Convert ARC puzzles (grids) into a compressed latent representation
and decode them back → guess the transformation rules.

This is NOT trained.
Weights are random but symmetric — ARC is solved through enumerating many forward passes, not learning.

🎯 WHY a “compressor” for ARC puzzles?

ARC puzzles are small grids (5×5, 10×10).
But the rules are high-level (“copy this object here”, “reflect this”, “repeat pattern”).

ARC puzzles are small grids (5×5, 10×10).
But the rules are high-level (“copy this object here”, “reflect this”, “repeat pattern”).

Humans see structure, not raw pixels.

This compressor maps grids → a tensor space → operations that mimic:
symmetry
directionality
grouping
spatial patterns
color relationships
mask prediction
copy/move/shift/cummax patterns

This “VAE decoder” is actually a rule generator.
