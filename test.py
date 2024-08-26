# import os
# import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"]="0" #,1,2,3
# from exit.utils import animate3d, t2m_bone
# import numpy as np
# import torch
# from tqdm import tqdm
# from exit.utils import visualize_2motions
# from utils.motion_process import recover_from_ric
# from generate import MMM
# from dataset import dataset_TM_eval
# from utils.word_vectorizer import WordVectorizer
# w_vectorizer = WordVectorizer('./glove', 'our_vab')
# val_loader = dataset_TM_eval.DATALoader('t2m', True, 32, w_vectorizer)
# class Temp:
#     def __init__(self, extra_args=None):
#         print('mock:: opt')
#         if extra_args is not None:
#             for i in extra_args:
#                 self.__dict__[i] = extra_args[i]
# args = Temp()

# args.dataname = args.dataset_name = 't2m'
# args.nb_code = 256
# args.code_dim = 32
# args.output_emb_width = 512
# args.down_t = 2
# args.stride_t = 2
# args.width = 512
# args.depth = 3
# args.dilation_growth_rate = 3
# args.quantizer = 'ema_reset'
# args.mu = 0.99
# args.clip_dim = 512


# args.embed_dim_gpt = 512
# args.block_size = 51
# args.num_layers = 9
# args.n_head_gpt = 16
# args.drop_out_rate = 0.1
# args.ff_rate = 4
# # Make you you already download pretrained model in read me section 2.3
# args.resume_pth = './output/vq/2024-08-19-00-17-48_vq_general_decoder/net_last.pth'
# args.resume_trans = './output/t2m/2024-08-23-17-56-48_trans_time_1/net_last.pth'
# args.num_local_layer = 0
# args.mean = val_loader.dataset.mean
# args.std = val_loader.dataset.std
# mmm = MMM(args,is_time=True).cuda()
# text = ['A person walks'] # 92
# m_length = torch.tensor([156])
# pred_pose = mmm(text, m_length.cuda(), rand_pos=False)
# print(pred_pose.shape)
# k = 0
# _motions = pred_pose[k, :m_length[k]].detach().cpu().numpy() * val_loader.dataset.std + val_loader.dataset.mean
# print(_motions.shape)
# _motions = recover_from_ric(torch.from_numpy(_motions).float(), 22).numpy()
# print(_motions.shape)


from exit.utils import animate3d, t2m_bone
import numpy as np
import torch
from tqdm import tqdm
from exit.utils import visualize_2motions
from utils.motion_process import recover_from_ric
from generate import MMM
from dataset import dataset_TM_eval
from utils.word_vectorizer import WordVectorizer

w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader('t2m', True, 32, w_vectorizer)
class Temp:
    def __init__(self, extra_args=None):
        print('mock:: opt')
        if extra_args is not None:
            for i in extra_args:
                self.__dict__[i] = extra_args[i]
args = Temp()

args.dataname = args.dataset_name = 't2m'
args.nb_code = 8192
args.code_dim = 32
args.nb_code = 256
args.code_dim = 32
args.output_emb_width = 512
args.down_t = 2
args.stride_t = 2
args.width = 512
args.depth = 3
args.dilation_growth_rate = 3
args.quantizer = 'ema_reset'
args.mu = 0.99
args.clip_dim = 512

args.embed_dim_gpt = 1024
args.embed_dim_gpt = 512
args.block_size = 51
args.num_layers = 9
args.n_head_gpt = 16
args.drop_out_rate = 0.1
args.ff_rate = 4
args.resume_trans='./output/t2m/2024-06-04-09-29-20_trans_name_b128/net_last.pth'
args.resume_pth='./output/vq/2024-06-03-20-22-07_retrain/net_last.pth'
args.resume_pth = './output/vq/2024-08-19-00-17-48_vq_general_decoder/net_last.pth'
args.resume_trans = './output/t2m/2024-08-23-17-56-48_trans_time_1/net_last.pth'
args.num_local_layer = 0

args.mean = val_loader.dataset.mean
args.std = val_loader.dataset.std
mmm = MMM(args,is_time=True).cuda()
texts = ['A person is walking']*5
m_length = torch.tensor([196]).cuda()
motion1 = mmm.multi_generate(m_length,texts)
edit_idx = torch.tensor([0,1]).cuda()
token_len = int(motion1.shape[1]/4)
mask = torch.zeros((motion1.shape[0], token_len), dtype=int)
mask[:, 10:-10 ] = 1
motion2 = mmm.multi_edit(motion1, m_length, texts,edit_idx,mask)
# k = 0
# _motions = pred_pose[k, :m_length[k]].detach().cpu().numpy() * val_loader.dataset.std + val_loader.dataset.mean
# print(_motions.shape)
# _motions = recover_from_ric(torch.from_numpy(_motions).float(), 22).numpy()
# print(_motions.shape)
# animate3d(_motions, BONE_LINK=t2m_bone, first_total_standard=63, axis_visible=True)