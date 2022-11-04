#!/usr/bin/env bash
# {your SBATCH header here}
 
# your environment specifications here (e.t. activate your conda env) 

python -m model_symmetries.stitching.cifar10_stitching --retrain_repeats=$3 --stitch_repeats=1  --stitch_epochs=20  --lowrank --name=$1'rank'$2 --arch=$1  --rank=$2 --stitch_lr=0.001 --noschedule --nopermwarmup;