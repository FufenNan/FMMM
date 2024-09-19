. /home/haoyum3/anaconda3/etc/profile.d/conda.sh
cd /home/haoyum3/MMM
conda activate MMM
name='trans_time_trajectory' # TEMP
dataset_name='t2m'
vq_name='2024-08-19-00-17-48_vq_general_decoder'
debug='f'
# id:[0,1,2,3,4,5] gpu:[0,5,1,2,3,4]
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
MULTI_BATCH=6

python3 finetune_t2m_trans_time.py  \
    --exp-name ${name} \
    --batch-size $((16*MULTI_BATCH)) \
    --num-layers 9 \
    --num-local-layer 0 \
    --embed-dim-gpt 512 \
    --nb-code 256 \
    --code-dim 32 \
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
    --resume-trans /home/haoyum3/MMM/output/t2m/2024-09-07-08-37-23_trans_time_2/net_last.pth \
    --teacher-pth /home/haoyum3/MMM/output/vq/2024-08-16-23-15-13_multi_vq_256_32_5/net_last.pth \
    --resume-pth /home/haoyum3/MMM/output/vq/2024-08-19-00-17-48_vq_general_decoder/net_last.pth
    sleep 500
# original setting
# --batch-size $((128*num_gpu)) \
# --num-layers 9 \
# --embed-dim-gpt 1024 \
# --n-head-gpt 16 \
# --total-iter $((300000/num_gpu)) \
# --lr-scheduler $((150000/num_gpu)) \
# --eval-iter $((10000/num_gpu)) \