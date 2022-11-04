#!/usr/bin/env bash
# {your SBATCH header here}
 
# your environment specifications here (e.t. activate your conda env) 

python -m model_symmetries.alignment.cifar10_alignment --repeats=$3  --arch=$1 --similarity=$2