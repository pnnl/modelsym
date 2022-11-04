#!/usr/bin/env bash
# strongly recommend running w/ echo lines first!

# Grelu and standard stitching
for s in {'Grelu','standard'}; do
    sbatch --job-name=mCNN$s -o mCNN$s.out -e mCNN$s.out stitch_$s.slurm mCNN 32;
done

for a in {'resnet20','resnet18'}; do
    for s in {'Grelu','standard'}; do
        sbatch --job-name=$a$s -o $a$s.out -e $a$s.out stitch_$s.slurm $a 16;
    done;
done

# reduced rank stitching
for r in {1,2,4}; do
    sbatch --job-name=mCNNlowrank -o mCNNlowrank.out -e mCNNlowrank.out stitch_rr.slurm mCNN $r 32;
done 

for a in {'resnet20','resnet18'}; do
    for r in {1,2,4}; do
        sbatch --job-name=$a'lowrank' -o $a'lowrank'.out -e $a'lowrank'.out stitch_rr.slurm $a $r 16;
    done;
done

# lasso stitching
for l in {'0.0001','0.001','0.01','0.1'}; do
    sbatch --job-name=mCNNlasso -o mCNNlasso.out -e mCNNlasso.out stitch_lasso.slurm mCNN $l 32;
done 

for a in {'resnet20','resnet18'}; do
    for l in {'0.0001','0.001','0.01','0.1'}; do
        sbatch --job-name=$a'lasso' -o $a'lasso'.out -e $a'lasso'.out stitch_lasso.slurm $a $l 16;
    done;
done

# Grelu stitching with crossval.
for c in {0..7}; do
    sbatch -p dlt_shared --job-name=mCNNGrelu_crossval -o mCNNGrelu_crossval.out -e mCNNGrelu_crossval.out stitch_Grelu_crossval.slurm mCNN 32 8 $c;
done

for a in {'resnet20','resnet18'}; do
    for c in {0..7}; do
        sbatch -p a100_shared --job-name=$a'Grelu_crossval' -o $a'Grelu_crossval'.out -e $a'Grelu_crossval'.out stitch_Grelu_crossval.slurm $a 16 8 $c;
    done;
done 


