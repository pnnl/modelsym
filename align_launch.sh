#!/usr/bin/env bash

# procrustes
for a in {'mCNN','resnet20','resnet18','cwCNN'}; do
    for d in {'procrustes','ortho_procrustes'}; do
        sbatch --job-name=$a$d -o $a$d'.out' -e $a$d'.out' align.slurm $a $d 32;
    done;
done     

# cka
for a in {'mCNN','resnet20','resnet18','cwCNN'}; do
    for d in {'cka','ortho_cka'}; do
        sbatch --job-name=$a$d -o $a$d'.out' -e $a$d'.out' align.slurm $a $d 16;
    done;
done