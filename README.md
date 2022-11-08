Code to run the experiments of the Neurips 2022 paper [On the Symmetries of Deep
Learning Models and their Internal
Representations](https://arxiv.org/abs/2205.14258).

# Overview

This repository is currently organized into a module `model_symmetries` with
submodules `stitching` and `alignment`, corresponding to sections 4 and 5 of
the paper (for the network dissection results of section 6 we used the
implementation at
[https://github.com/CSAILVision/NetDissect-Lite](https://github.com/CSAILVision/NetDissect-Lite)).

In addition there are some submodules containing code shared across `stitching`
and `alignment`, namely
- `models.py,` `datasets.py`, `train.py` and `plotting.py` (self explanatory)
- `zoo.py`: utilities to train a bunch of models from independent random seeds
- `constants.py`: specify a directory in which to store data/models/results by
  defining the variable `data_dir`.

## `stitching`

The key classes for stitching layers and stitched models are in `stitching.py`.
In particular, we direct attention towards the `Birkhoff` class, which
implements for our approach using PGD on the Birkhoff polytope of doubly
stochastic matrices.

`train.py` has more options than is typical, due to a few major implementation
considerations:
1. The need to make sure that when stitching, we *only* update parameters of the
   stitching layer.
2. The overhead of PGD and extra $-\ell_2$ regularization.
3. The necessity of a no-grad training epoch before validation.

The main experiment script is `cifar10_stitching.py`. This also has many
options, due to the number of combinations of model/stitching layer type we
consider.

In order to run the experiments stitching Compact Convolutional Transformers,
you will need
[https://github.com/SHI-Labs/Compact-Transformers](https://github.com/SHI-Labs/Compact-Transformers),
which is included as a Git submodule of this repository at
`model_symmetries/ct`. To initialize and update it, run
``` bash
git submodule init && git submodule update
```

## `alignment`

Core functions are located in `alignment.py`. The $G_{\mathrm{ReLU}}$-Procrustes
and CKA metrics are `wreath_{procrustes,cka}` (the group $G_{\mathrm{ReLU}}$ is
an example of a [wreath product](https://en.wikipedia.org/wiki/Wreath_product),
hence the name).

## Visualization

`plotting.py` contains functions for displaying stitching penalties and
dissimilarity metrics, which can be run in the notebook `plotting.ipynb`.

## Parallelization

We ran these experiments on a cluster managed by
[SLURM](https://slurm.schedmd.com/documentation.html) -- files ending in
`.slurm` are SLURM batch files. In order to distribute the many sweeps in these
experiments across nodes of the cluster, we submitted batches to the queue using
loops found in the bash scripts (files ending in `.sh`). **WARNING**: executing
these scripts will consume many GPU days.

# Citation

If you find this code useful, please cite our paper.

```bibtex
@article{modelsyms2022,
  doi = {10.48550/ARXIV.2205.14258},
  url = {https://arxiv.org/abs/2205.14258},
  author = {Godfrey, Charles and Brown, Davis and Emerson, Tegan and Kvinge, Henry},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {On the Symmetries of Deep Learning Models and their Internal Representations},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

# Notice

This research was supported by the Mathematics for Artificial Reasoning in Science (MARS)
initiative at Pacific Northwest National Laboratory. It was conducted under the Laboratory Directed
Research and Development (LDRD) Program at at Pacific Northwest National Laboratory (PNNL), a
multiprogram National Laboratory operated by Battelle Memorial Institute for the U.S. Department
of Energy under Contract DE-AC05-76RL01830.
