import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
from tqdm import tqdm
from exit.utils import get_model, generate_src_mask, init_save_folder
from models.vqvae_multi import VQVAE_MULTI_V2 as VQVAE_MULTI
##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
torch.cuda.set_device(0)
# os.makedirs(args.out_dir, exist_ok = True)

w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit' : 
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    
else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
val_loader = dataset_TM_eval.DATALoader(args.dataname, False,
                                        32,
                                        w_vectorizer,
                                        unit_length=2**args.down_t)
args.out_dir = os.path.join(args.out_dir, args.codebook_name)
##### ---- Load Codebook ---- #####
if not os.path.exists(args.out_dir):
    print(f'creating new codebook directory at {args.out_dir}')
    os.makedirs(args.out_dir)

net = VQVAE_MULTI(args, ## use args to define different parameters in different quantizers
                    args.nb_code,#8192
                    args.code_dim,#32
                    args.output_emb_width,#512
                    args.down_t,#2
                    args.stride_t,#2
                    args.width,#512
                    args.depth,#3
                    args.dilation_growth_rate,#3
                    args.vq_act,#'relu'
                    args.vq_norm,#None
                    {'mean': torch.from_numpy(val_loader.dataset.mean).cuda().float(), 
                    'std': torch.from_numpy(val_loader.dataset.std).cuda().float()},
                    True)

if args.resume_pth : 
    print('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)

dict_codebook = {'left_arm': net.quantizer_left_arm.codebook, 'right_arm': net.quantizer_right_arm.codebook, 'right_leg': net.quantizer_right_leg.codebook,'left_leg': net.quantizer_left_leg.codebook, 'spine': net.quantizer_spine.codebook}
##### ---- Save Codebook ---- #####
torch.save(dict_codebook, os.path.join(args.out_dir, 'codebook_dict.pth'))
print(f'codebook_dict saved at {os.path.join(args.out_dir, 'dict_codebook_dict.pth')}')
