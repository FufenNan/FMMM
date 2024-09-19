#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# sbatch train_vq.sh

# cd /home/epinyoan/git/MaskText2Motion/T2M-BD/experiments
# screen -S temp ~/git/MaskText2Motion/T2M-BD/experiments/train_vq.sh

#SBATCH --job-name=1GPU
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. /home/haoyum3/anaconda3/etc/profile.d/conda.sh
cd /home/haoyum3/MMM
conda activate MMM
name='vq_general_decoder2'
dataset_name='t2m'
debug='f'
export CUDA_VISIBLE_DEVICES=4
python3 train_vq_general.py \
    --batch-size 256 \
    --lr 1e-4 \
    --total-iter 300000  \
    --lr-scheduler 200000 \
    --nb-code 256 \
    --code-dim 32 \
    --width 512 \
    --down-t 2 \
    --depth 3 \
    --dilation-growth-rate 3 \
    --out-dir output \
    --dataname ${dataset_name} \
    --vq-act relu \
    --quantizer ema_reset \
    --loss-vel 0.5 \
    --recons-loss l1_smooth \
    --exp-name ${name} \
    --warm-up-iter 1000\
    --teacher-pth "output/vq/2024-08-16-23-15-13_multi_vq_256_32_5/net_last.pth"\
    # --wo-trajectory
sleep 500