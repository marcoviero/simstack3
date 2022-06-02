# The SIMSTACK Algorithm

If you have not read Viero+13, and even if you have, SIMSTACK may seem like a blackbox. But it is important to understand how SIMSTACK works so that you appreciate where it fails.  This document aims to describe, as simply as possible, the algorithm: what it does and what it does not do.

SIMSTACK is distinctly different from traditional thumbnail stacking. In thumbnail stacking, cutouts around galaxy positions are summed together and an image emerges, while noise cancels out. SIMSTACK should instead be thought of as model fitting (or more accurately, real-space component separation) where the model being fit is a set of layers that together sum up to reproduce the map. Each layer represents a narrow selection of galaxies which are assumed to have similar infrared properties (LIR, dust mass, dust temperature, etc.)  

The more precise your assumptions, the better 

Why it works... double counting.

Where it fails... incomplete.  Large areas, not many objects.

How to improve...

