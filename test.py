import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="2" #,1,2,3
from exit.utils import animate3d, t2m_bone
import numpy as np
import torch
from tqdm import tqdm
from exit.utils import visualize_2motions
from utils.motion_process import recover_from_ric
from generate import MMM
from dataset import dataset_TM_eval,dataset_TM_train
from utils.word_vectorizer import WordVectorizer

w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader('t2m', True, 8, w_vectorizer)
class Temp:
    def __init__(self, extra_args=None):
        print('mock:: opt')
        if extra_args is not None:
            for i in extra_args:
                self.__dict__[i] = extra_args[i]
args = Temp()

args.dataname = args.dataset_name = 't2m'
args.nb_code = 8192
args.code_dim = 40
args.output_emb_width = 512
args.down_t = 2
args.stride_t = 2
args.width = 512
args.depth = 3
args.dilation_growth_rate = 3
args.quantizer = 'ema_reset'
args.mu = 0.99
args.clip_dim = 512
args.embed_dim_gpt = 1280
args.block_size = 51
args.num_layers = 9
args.n_head_gpt = 16
args.drop_out_rate = 0.1
args.ff_rate = 4
# download pretrain upper body edting model by running "bash dataset/prepare/download_model_upperbody.sh" as described in read me section 2.4.
# args.resume_pth = './output/vq/vq_upperbody/net_last.pth'
# args.resume_trans = './output/t2m/trans_upperbody/net_last.pth'
args.resume_pth = '/home/haoyum3/MMM/output/vq/2024-08-12-23-14-09_vq5_multi/net_last.pth'
args.resume_trans = '/home/haoyum3/MMM/output/temp/trans_multi_22500.pth'
args.num_local_layer = 2
args.mean = val_loader.dataset.mean
args.std = val_loader.dataset.std
mmm_upper = MMM(args, is_multi_edit=True).cuda()
args.resume_pth = '/home/haoyum3/MMM/output/vq/2024-06-03-20-22-07_retrain/net_last.pth'
args.resume_trans = '/home/haoyum3/MMM/output/t2m/2024-06-04-09-29-20_trans_name_b128/net_last.pth'
args.embed_dim_gpt = 1024
args.code_dim = 32
mmm_generate=MMM(args)

text = ['A man waving his left arm.'] # 92
m_length = torch.tensor([156])
pred_pose = mmm_generate(text, m_length.cuda(), rand_pos=False)
k = 0
# _motions = pred_pose[0, :m_length].detach().cpu().numpy() * val_loader.dataset.std + val_loader.dataset.mean
# _motions = recover_from_ric(torch.from_numpy(_motions).float(), 22).numpy()
# animate3d(_motions, BONE_LINK=t2m_bone, first_total_standard=63, axis_visible=True)

token_len = int(pred_pose.shape[1]/4)
lower_mask = torch.zeros((pred_pose.shape[0], token_len), dtype=int)
lower_mask[:, 10:-10 ] = 1
lower_mask[0]
pred_pose_upper = mmm_upper.multi_edit(pred_pose, m_length, ['A man waving his left arm.'], pred_idx=0,target_mask=lower_mask)
print(pred_pose_upper.shape)