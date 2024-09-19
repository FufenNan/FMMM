import os 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
from models.vqvae_multi import VQVAE_MULTI
from models.vqvae_general import VQVAE_decode
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_eval
import models.t2m_trans as trans
import models.t2m_timesformer as trans_time
from options.get_eval_option import get_opt
import options.option_vq as option_vq
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from exit.utils import base_dir, init_save_folder

##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = f'{args.out_dir}/eval'
os.makedirs(args.out_dir, exist_ok = True)
init_save_folder(args)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
#TODO 
val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb
clip_model = TextCLIP(clip_model)
if args.teacher_pth:
    net= VQVAE_MULTI(args, ## use args to define different parameters in different quantizers
                        args.nb_code,#8192
                        args.code_dim,#32
                        args.down_t,#2
                        args.stride_t,#2
                        args.width,#512
                        args.depth,#3
                        args.dilation_growth_rate,#3
                        args.vq_act,#'relu'
                        None,#None
                        {'mean': torch.from_numpy(val_loader.dataset.mean).cuda().float(), 
                        'std': torch.from_numpy(val_loader.dataset.std).cuda().float()},
                        True)
    logger.info('loading teacher checkpoint from {}'.format(args.teacher_pth))
    ckpt = torch.load(args.teacher_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
elif args.resume_pth:
    teacher_net= VQVAE_MULTI(args, ## use args to define different parameters in different quantizers
                    args.nb_code,#8192
                    args.code_dim,#32
                    args.down_t,#2
                    args.stride_t,#2
                    args.width,#512
                    args.depth,#3
                    args.dilation_growth_rate,#3
                    args.vq_act,#'relu'
                    None,#None
                    {'mean': torch.from_numpy(val_loader.dataset.mean).cuda().float(), 
                    'std': torch.from_numpy(val_loader.dataset.std).cuda().float()},
                    True)
    net= VQVAE_decode(args, ## use args to define different parameters in different quantizers
                            teacher_net,
                            args.nb_code,#8192
                            args.code_dim,#32
                            args.down_t,#2
                            args.stride_t,#2
                            args.width,#512
                            args.depth,#3
                            args.dilation_growth_rate,#3
                            args.vq_act,#'relu'
                            None,#None
                            )
    print ('loading checkpoint from {}'.format(args.resume_pth))
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)

net.eval()
net.cuda()

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
repeat_time = 20

from tqdm import tqdm
for i in tqdm(range(repeat_time)):
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper,wo_trajectory=args.wo_trajectory)
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)

print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)