#!/usr/bin/env bash

# a runs over model archs, i runs over chunks of the 100 independent seeds (to
# train 4 batches of 25 models in parallel rather than 100 in serial). 
for a in {'mCNN','resnet20','resnet18','cwCNN'}; do
    for i in {0..3}; do
        sbatch --job-name=$a'zoo' -o $a'zoo.out' -e $a'zoo.out' zoo.slurm $a 4 $i;
    done;
done