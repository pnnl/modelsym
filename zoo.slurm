#!/usr/bin/env bash
# {your SBATCH header here}
 
# your environment specifications here (e.t. activate your conda env) 

python -m model_symmetries.zoo --num_models=100 --arch=$1 --num_chunks=$2 --chunk=$3