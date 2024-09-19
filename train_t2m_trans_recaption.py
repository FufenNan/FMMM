import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip
import options.option_transformer as option_trans
from models.vqvae_multi import VQVAE_MULTI
from models.vqvae_general import VQVAE_decode
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train,dataset_TM_eval,dataset_tokenize,dataset_TM_train_recaption,dataset_TM_eval_recaption
import models.t2m_timesformer as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from exit.utils import get_model, visualize_2motions
from tqdm import tqdm
from exit.utils import get_model, visualize_2motions, generate_src_mask, init_save_folder, uniform, cosine_schedule
from einops import rearrange, repeat
import torch.nn.functional as F
from exit.utils import base_dir

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

# args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
init_save_folder(args)

# [TODO] make the 'output/' folder as arg
args.vq_dir = f'./output/vq/{args.vq_name}' #os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
if args.wo_trajectory:
    codebook_dir = f'{args.vq_dir}/codebook_wo_trajectory/'
else:
    codebook_dir = f'{args.vq_dir}/codebook/'
# args.resume_pth = f'{args.vq_dir}/net_last.pth'
os.makedirs(args.vq_dir, exist_ok = True)
os.makedirs(codebook_dir, exist_ok = True)
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.out_dir+'/html', exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
BODY_PARTS=['left arm','right arm','left leg','right leg','spine']

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
#TODO 
#val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)
val_loader = dataset_TM_eval_recaption.DATALoader(args.dataname, False, 32, w_vectorizer)
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
# if args.teacher_pth:
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

trans_encoder = trans.Text2Motion_Transformer(vqvae=net,
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
trans_encoder = torch.nn.DataParallel(trans_encoder)

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss(reduction='none')

##### ---- get code ---- #####
##### ---- Dataloader ---- #####
if len(os.listdir(codebook_dir)) == 0:
    train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t)
    for batch in tqdm(train_loader_token,position=0, leave=True):
        pose, name = batch
        if args.wo_trajectory:
            pose[:,:,1:3] = 0
        bs, seq = pose.shape[0], pose.shape[1]

        pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
        target = net.vqvae.teacher_net(pose, type='encode')
        target = target.cpu().numpy()
        np.save(pjoin(codebook_dir, name[0] +'.npy'), target)

#TODO 
train_loader = dataset_TM_train_recaption.DATALoader(args.dataname, args.batch_size, args.nb_code, codebook_dir, unit_length=2**args.down_t,multi_sep=True)
train_loader_iter = dataset_TM_train.cycle(train_loader)

        
##### ---- Training ---- #####
best_fid=1000 
best_iter=0 
best_div=100 
best_top1=0 
best_top2=0 
best_top3=0 
best_matching=100 
#TODO
pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_time_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper,is_recaption=True)

def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num*100/mask.sum()

# while nb_iter <= args.total_iter:
for nb_iter in tqdm(range(1, args.total_iter + 1), position=0, leave=True):
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len, recaption_text = batch
    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens    # (bs, 26)
    target = target.cuda()
    batch_size, max_len = target.shape[:2]

    # Random Drop Text
    # text_mask = np.random.random(len(clip_text)) > .05
    # clip_text = np.array(clip_text)
    # clip_text[~text_mask] = ''
    
    #[b,77]
    text = clip.tokenize(clip_text, truncate=True).cuda()
    #[bs,512],[bs,77,512]
    feat_clip_text, word_emb = clip_model(text)
    #[bs,1,5,512]
    feat_clip_text = feat_clip_text.unsqueeze(1).repeat(1,5,1)
    #local text cfg
    feat_clip_local_texts = []
    for i,body_part in enumerate(BODY_PARTS):
        clip_local_text = recaption_text[body_part]
        local_text = clip.tokenize(clip_local_text, truncate=True).cuda()
        feat_clip_local_text, _ = clip_model(local_text)
        feat_clip_local_texts.append(feat_clip_local_text)
        #TODO
        #feat_clip_text[:,i,:] = feat_clip_local_text+args.w*(feat_clip_text[:,i,:]-feat_clip_local_text)
        #Interpolation
        #feat_clip_text[:,i,:] = args.w*feat_clip_local_text+(1-args.w)*feat_clip_text[:,i,:]
    # [INFO] Swap input tokens
    feat_clip_text = torch.cat([feat_clip_text, torch.stack(feat_clip_local_texts, dim=1)], dim=-1)
    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(target.shape,
                                                device=target.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(target.shape,
                                                device=target.device))
    # random only motion token (not pad token). To prevent pad token got mixed up.
    seq_mask_no_end = generate_src_mask(max_len, m_tokens_len).unsqueeze(-1).repeat(1,1,5)
    mask = torch.logical_or(mask, ~seq_mask_no_end).int()
    r_indices = torch.randint_like(target, args.nb_code)
    input_indices = mask*target+(1-mask)*r_indices

    # Time step masking
    mask_id = get_model(net).vqvae.num_code + 2
    # rand_time = uniform((batch_size,), device = target.device)
    # rand_mask_probs = cosine_schedule(rand_time)
    rand_mask_probs = torch.zeros(batch_size,device = m_tokens_len.device).float().uniform_(0.5, 1)
    # rand_mask_probs = cosine_schedule(rand_mask_probs)
    num_token_masked = (m_tokens_len * rand_mask_probs).round().clamp(min = 1)
    batch_randperm = torch.rand((batch_size, max_len),device = target.device)
    batch_randperm = batch_randperm.argsort(dim=-1)
    mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
    mask_token = mask_token.unsqueeze(-1).repeat(1,1,5)

    # masked_target = torch.where(mask_token, input=input_indices, other=-1)
    masked_input_indices = torch.where(mask_token, mask_id, input_indices)

    seq_mask = generate_src_mask(max_len, m_tokens_len+1)
    att_txt = None # CFG: torch.rand((seq_mask.shape[0], 1)) > 0.1
    cls_pred = trans_encoder(masked_input_indices, feat_clip_text, src_mask = seq_mask, att_txt=att_txt, word_emb=word_emb)[:, 1:]
    cls_pred = cls_pred.squeeze(2)
    # [INFO] Compute xent loss as a batch
    
    weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
    cls_pred_seq_masked = cls_pred[seq_mask_no_end, :].view(-1, cls_pred.shape[-1])
    target_seq_masked = target[seq_mask_no_end]
    weight_seq_masked = weights[seq_mask_no_end]
    loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked, reduction = 'none')
    loss_cls = (loss_cls * weight_seq_masked).sum()
    # global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    if nb_iter % args.print_iter ==  0 :
        probs_seq_masked = torch.softmax(cls_pred_seq_masked, dim=-1)
        _, cls_pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)
        target_seq_masked = torch.masked_select(target, seq_mask_no_end)
        right_seq_masked = (cls_pred_seq_masked_index == target_seq_masked).sum()

        writer.add_scalar('./Loss/all', loss_cls, nb_iter)
        writer.add_scalar('./ACC/every_token', right_seq_masked*100/seq_mask_no_end.sum(), nb_iter)
        
        # [INFO] log mask/nomask separately
        no_mask_token = ~mask_token * seq_mask_no_end
        writer.add_scalar('./ACC/masked', get_acc(cls_pred, target, mask_token), nb_iter)
        writer.add_scalar('./ACC/no_masked', get_acc(cls_pred, target, no_mask_token), nb_iter)

        # msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        # logger.info(msg)

    if nb_iter==0 or nb_iter % args.eval_iter ==  0 or nb_iter == args.total_iter:
        num_repeat = 1
        rand_pos = False
        if nb_iter == args.total_iter:
            num_repeat = -30
            rand_pos = True
            # val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)
            val_loader = dataset_TM_eval_recaption.DATALoader(args.dataname, True, 32, w_vectorizer)
        pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_time_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, dataname=args.dataname, num_repeat=num_repeat, rand_pos=rand_pos,is_recaption=True)
        # for i in range(4):
        #     x = pose[i].detach().cpu().numpy()
        #     y = pred_pose_eval[i].detach().cpu().numpy()
        #     l = m_length[i]
        #     caption = clip_text[i]
        #     cleaned_name = '-'.join(caption[:200].split('/'))
        #     visualize_2motions(x, val_loader.dataset.std, val_loader.dataset.mean, args.dataname, l, y, save_path=f'{args.out_dir}/html/{str(nb_iter)}_{cleaned_name}_{l}.html')

    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            