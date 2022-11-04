#!/usr/bin/env bash
# {your SBATCH header here}
 
# your environment specifications here (e.t. activate your conda env) 

python -m model_symmetries.stitching.cifar10_stitching --retrain_repeats=$2 --stitch_repeats=1  --stitch_epochs=20 --name=$1'standard' --arch=$1 --nopermwarmup --noschedule --stitch_lr=0.001;