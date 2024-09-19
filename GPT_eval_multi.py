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
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from exit.utils import base_dir, init_save_folder

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
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

class VQVAE_WRAPPER(torch.nn.Module):
    def __init__(self, vqvae) :
        super().__init__()
        self.vqvae = vqvae
        
    def forward(self, *args, **kwargs):
        return self.vqvae(*args, **kwargs)
net=VQVAE_WRAPPER(net)

trans_encoder = trans_time.Text2Motion_Transformer(vqvae=net,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()

# net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
#                        args.nb_code,
#                        args.code_dim,
#                        args.output_emb_width,
#                        args.down_t,
#                        args.stride_t,
#                        args.width,
#                        args.depth,
#                        args.dilation_growth_rate)


# trans_encoder = trans_time.Text2Motion_Transformer(net,
#                                 num_vq=args.nb_code, 
#                                 embed_dim=args.embed_dim_gpt, 
#                                 clip_dim=args.clip_dim, 
#                                 block_size=args.block_size, 
#                                 num_layers=args.num_layers, 
#                                 num_local_layer=args.num_local_layer, 
#                                 n_head=args.n_head_gpt, 
#                                 drop_out_rate=args.drop_out_rate, 
#                                 fc_rate=args.ff_rate)


# print ('loading checkpoint from {}'.format(args.resume_pth))
# ckpt = torch.load(args.resume_pth, map_location='cpu')
# net.load_state_dict(ckpt['net'], strict=True)
# net.eval()
# net.cuda()

# if args.resume_trans is not None:
#     print ('loading transformer checkpoint from {}'.format(args.resume_trans))
#     ckpt = torch.load(args.resume_trans, map_location='cpu')
#     trans_encoder.load_state_dict(ckpt['trans'], strict=True)
# trans_encoder.train()
# trans_encoder.cuda()

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []
repeat_time = 20

from tqdm import tqdm
for i in tqdm(range(repeat_time)):
    pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_time_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper)
    # pred_pose_eval, pose, m_length, clip_text, \
    # best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper, dataname=args.dataname, save = False, num_repeat=11, rand_pos=True)
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)
    multi.append(best_multi)

print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)
print('multi: ', sum(multi)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
multi = np.array(multi)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)


# import os 
# import torch
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# import json
# import clip

# import options.option_transformer as option_trans
# import models.vqvae as vqvae
# import utils.utils_model as utils_model
# import utils.eval_trans as eval_trans
# from dataset import dataset_TM_eval
# import models.t2m_trans as trans
# from options.get_eval_option import get_opt
# from models.evaluator_wrapper import EvaluatorModelWrapper
# import warnings
# warnings.filterwarnings('ignore')
# from exit.utils import base_dir, init_save_folder

# ##### ---- Exp dirs ---- #####
# args = option_trans.get_args_parser()
# torch.manual_seed(args.seed)

# args.out_dir = f'{args.out_dir}/eval'
# os.makedirs(args.out_dir, exist_ok = True)
# init_save_folder(args)

# ##### ---- Logger ---- #####
# logger = utils_model.get_logger(args.out_dir)
# writer = SummaryWriter(args.out_dir)
# logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# from utils.word_vectorizer import WordVectorizer
# w_vectorizer = WordVectorizer('./glove', 'our_vab')
# val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

# dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

# wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
# eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

# ##### ---- Network ---- #####
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
# clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
# clip_model.eval()
# for p in clip_model.parameters():
#     p.requires_grad = False

# # https://github.com/openai/CLIP/issues/111
# class TextCLIP(torch.nn.Module):
#     def __init__(self, model) :
#         super(TextCLIP, self).__init__()
#         self.model = model
        
#     def forward(self,text):
#         with torch.no_grad():
#             word_emb = self.model.token_embedding(text).type(self.model.dtype)
#             word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
#             word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
#             word_emb = self.model.transformer(word_emb)
#             word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
#             enctxt = self.model.encode_text(text).float()
#         return enctxt, word_emb
# clip_model = TextCLIP(clip_model)

# net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
#                        args.nb_code,
#                        args.code_dim,
#                        args.output_emb_width,
#                        args.down_t,
#                        args.stride_t,
#                        args.width,
#                        args.depth,
#                        args.dilation_growth_rate)


# trans_encoder = trans.Text2Motion_Transformer(net,
#                                 num_vq=args.nb_code, 
#                                 embed_dim=args.embed_dim_gpt, 
#                                 clip_dim=args.clip_dim, 
#                                 block_size=args.block_size, 
#                                 num_layers=args.num_layers, 
#                                 num_local_layer=args.num_local_layer, 
#                                 n_head=args.n_head_gpt, 
#                                 drop_out_rate=args.drop_out_rate, 
#                                 fc_rate=args.ff_rate)


# print ('loading checkpoint from {}'.format(args.resume_pth))
# ckpt = torch.load(args.resume_pth, map_location='cpu')
# net.load_state_dict(ckpt['net'], strict=True)
# net.eval()
# net.cuda()

# if args.resume_trans is not None:
#     print ('loading transformer checkpoint from {}'.format(args.resume_trans))
#     ckpt = torch.load(args.resume_trans, map_location='cpu')
#     trans_encoder.load_state_dict(ckpt['trans'], strict=True)
# trans_encoder.train()
# trans_encoder.cuda()


# fid = []
# div = []
# top1 = []
# top2 = []
# top3 = []
# matching = []
# multi = []
# repeat_time = 20

# from tqdm import tqdm
# for i in tqdm(range(repeat_time)):
#     pred_pose_eval, pose, m_length, clip_text, \
#     best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper, dataname=args.dataname, save = False, num_repeat=11, rand_pos=True)
#     fid.append(best_fid)
#     div.append(best_div)
#     top1.append(best_top1)
#     top2.append(best_top2)
#     top3.append(best_top3)
#     matching.append(best_matching)
#     multi.append(best_multi)

# print('final result:')
# print('fid: ', sum(fid)/repeat_time)
# print('div: ', sum(div)/repeat_time)
# print('top1: ', sum(top1)/repeat_time)
# print('top2: ', sum(top2)/repeat_time)
# print('top3: ', sum(top3)/repeat_time)
# print('matching: ', sum(matching)/repeat_time)
# print('multi: ', sum(multi)/repeat_time)

# fid = np.array(fid)
# div = np.array(div)
# top1 = np.array(top1)
# top2 = np.array(top2)
# top3 = np.array(top3)
# matching = np.array(matching)
# multi = np.array(multi)
# msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
# logger.info(msg_final)