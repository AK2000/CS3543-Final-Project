# Learning to Reorder Edges in Sparse Neural Networks
CS3543 Final Project, Winter 2023, Alok Kamatar

This repository implements two algorithms for repordering edges.

## Getting Started

Before beginning make sure that you have `make` and `clang` installed.

Make sure to run
`git submodule update --init` to get the necessary data from the environment.

To set up the conda environment, run:
`conda env create --name envname --file=environment.yml`

We also need the `eigen` header only library. Download `eigen` from https://eigen.tuxfamily.org/index.php?title=Main_Page. Place the uncrompessed header files in `extern/eigen-x.x.x`

Then, navigate to `SpDnn/src` and run `make`.

## Run the Experiments
Run all commands from the base directory.

First, generate the random graphs:
```
python scripts/generateRandomDnns.py
```

To run the baselines
```
bash scripts/run_baselines.sh
```

To run the EGO experiments. Run:
```
bash scripts/experiments.sh
```

## Generate The Plots
You will need jupyter notebook installed on the system. Then navigate to `notebooks/` and run `jupyter notebook`.