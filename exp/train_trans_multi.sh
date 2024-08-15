#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# sbatch train_trans.sh

# cd /home/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# screen -L -Logfile HML3D_44_upperEdit_transMaskLower_moveUpperDown_1crsAttn_noRandLen_dropTxt.1 -S temp ~/git/MaskText2Motion/T2M-BD/experiments/train_trans_uplow.sh

#SBATCH --job-name=HML3D_44_upperEdit_transMaskLower_moveUpperDown_1crsAttn_noRandLen_dropTxt.1
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. /home/haoyum3/anaconda3/etc/profile.d/conda.sh
cd /home/haoyum3/MMM
conda activate MMM
name='trans_multi_1' # TEMP
dataset_name='t2m'
vq_name='2024-08-12-23-14-09_vq5_multi'
debug='f'
export CUDA_VISIBLE_DEVICES=1,2,4,5
# export CUDA_LAUNCH_BLOCKING=1
# --resume-trans /home/epinyoan/git/MaskText2Motion/T2M-BD/output/2023-04-08-08-16-27_2_train_withEval/net_last.pth
MULTI_BATCH=4

python3 train_t2m_trans_multi.py  \
    --exp-name ${name} \
    --batch-size $((128*MULTI_BATCH)) \
    --num-layers 9 \
    --embed-dim-gpt 1280 \
    --nb-code 8192 \
    --code-dim 40 \
    --n-head-gpt 16 \
    --block-size 51 \
    --ff-rate 4 \
    --drop-out-rate 0.1 \
    --vq-name ${vq_name} \
    --out-dir output/${dataset_name} \
    --total-iter $((300000/MULTI_BATCH)) \
    --lr-scheduler $((150000/MULTI_BATCH)) \
    --lr 0.0001 \
    --dataname ${dataset_name} \
    --down-t 2 \
    --depth 3 \
    --quantizer ema_reset \
    --eval-iter $((10000/MULTI_BATCH)) \
    --pkeep 0.5 \
    --dilation-growth-rate 3 \
    --vq-act relu\
    --resume-trans /home/haoyum3/MMM/output/t2m/2024-08-14-12-01-17_trans_multi_1/net_last.pth
    sleep 500

# original setting
# --batch-size $((128*num_gpu)) \
# --num-layers 9 \
# --embed-dim-gpt 1024 \
# --n-head-gpt 16 \
# --total-iter $((300000/num_gpu)) \
# --lr-scheduler $((150000/num_gpu)) \
# --eval-iter $((10000/num_gpu)) \