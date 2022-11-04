#!/usr/bin/env bash
# {your SBATCH header here}
 
# your environment specifications here (e.t. activate your conda env) 

python -m model_symmetries.stitching.cifar10_stitching_crossval --retrain_repeats=$2 --stitch_repeats=5  --stitch_epochs=5  --birkhoff  --alpha=crossval --stitch_lr=0.001 --noschedule --bn_epoch  --name=$1'birkhoff' --arch=$1 --num_chunks=$3 --chunk=$4