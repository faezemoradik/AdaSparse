#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-user=<faeze.moradi@mail.utoronto.ca>
#SBATCH --mail-type=ALL


source /pkgs/anaconda3/etc/profile.d/conda.sh
conda activate myenv

cd ~/AdaSparse


python main.py -learning_rate 1e-4 -batch_size 50 -myseed ${SEED} \
-num_epoch 60 -dataset 'CIFAR10' -datasplit 'non-iid' -model_name 'ResNet9' \
-method ${ME} -kappa ${KAPPA} -epsilon 15.0 -delay_dist 'uniform' -alpha 0.0