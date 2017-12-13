
DEPICTIVE : DEtermining Parameter Influence on Cell To cell Variability Through the Inference of Variance Explained
===========================================

This package provides simple tools for dissecting sources of cell-to-cell variability from single cell data and perturbation experiments.  In addition we provide tools for simulation tools for testing inference strategy.

For tutorials on using these tools please refer to our [wiki](https://github.com/robert-vogel/depictive/wiki).

Dependencies
------------
To use this package you will need:

- Python (2.7)
- Numpy (>= 1.12.0)
- SciPy (>= 0.18.1)
- matplotlib (>= 2.0.0)

Using the package
-----------------
Simply download the code and place files in the directory in which you will be scripting in.  Or, save it in a directory and use Python's sys package to temporarily add these functions to your Python path, for example.

```Python
import sys
sys.path.append('/you_path/depictive/code')

import fit
```

Citation
--------

Please don't forget to check out our paper on [bioRxiv](https://www.biorxiv.org/content/early/2017/10/10/201160) and reference us
```
@article {2017mito_sims,
  author = {Santos, L.C. and Vogel, R. and Chipuk, J.E. and
    Birtwistle, M.R. and Stolovitzky, G. and Meyer, P.},
  title = {Origins of fractional control in regulated cell death},
  year = {2017},
  doi = {10.1101/201160},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2017/10/10/201160},
  eprint = {https://www.biorxiv.org/content/early/2017/10/10/201160.full.pdf},
  journal = {bioRxiv}
}
```
