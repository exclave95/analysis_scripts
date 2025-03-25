# Custom analysis scripts for molecular dynamics simulations involving clay minerals
+ Written/adapted by Jakub Ličko
+ Tested with Python version 3.8.16
+ Python library installation pre-requisites (includes library version that the codes were tested with).
  1. MDAnalysis (2.4.3)
  2. numpy (1.24.3)
  3. matplotlib (3.7.1)
  4. pandas (2.0.1)
  5. scienceplots (2.1.1) - optional, but recommended to make attractive plots with LaTeX

## USAGE:
+ The scripts were written and tested with GROMACS trajectories.
+ The scripts are intended to be run in the command line, taking GROMACS-like arguments (e.g. -t trajout.xtc -s topol.tpr).
  1. specific arguments for each code are specified at the top of each file

## NOTE:
+ The codes in this repository are still under active development.
