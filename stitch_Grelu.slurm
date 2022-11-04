#!/usr/bin/env bash
# {your SBATCH header here}
 
# your environment specifications here (e.t. activate your conda env) 

python -m model_symmetries.stitching.cifar10_stitching --retrain_repeats=$2 --stitch_repeats=1  --stitch_epochs=20  --birkhoff  --alpha=0.1 --stitch_lr=0.001 --noschedule --bn_epoch  --name=$1'birkhoff' --arch=$1