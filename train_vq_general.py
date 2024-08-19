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
from models.vqvae_general import HumanVQVAE_GENERAL
from models.vqvae_multi import VQVAE_MULTI_V2

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
torch.cuda.set_device(0)
args.codebook_dir = os.path.join(args.out_dir, f'codebook',args.codebook_name)
args.out_dir = os.path.join(args.out_dir, f'vq') # /{args.exp_name}
# os.makedirs(args.out_dir, exist_ok = True)
init_save_folder(args)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))



w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit' : 
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    
else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
train_loader = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)

train_loader_iter = dataset_VQ.cycle(train_loader)

val_loader = dataset_TM_eval.DATALoader(args.dataname, False,
                                        32,
                                        w_vectorizer,
                                        unit_length=2**args.down_t)
##### ---- Codebook ---- #####
#检查args.codebook_dir是否存在，如果存在则加载，否则创建一个新的
logger.info(f'loading codebook from {args.codebook_dir}')
codebook_dict = torch.load(os.path.join(args.codebook_dir, 'codebook_dict.pth'))

    
##### ---- Network ---- #####
net= HumanVQVAE_GENERAL(args, ## use args to define different parameters in different quantizers
                        list(codebook_dict.values()),
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
                        )
if args.teacher_pth:
    teacher_net= VQVAE_MULTI_V2(args, ## use args to define different parameters in different quantizers
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
    logger.info('loading checkpoint from {}'.format(args.teacher_pth))
    teacher_ckpt=torch.load(args.teacher_pth, map_location='cpu')
    teacher_net.load_state_dict(teacher_ckpt['net'], strict=True)
    teacher_net.cuda()
    teacher_net.eval()

if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
#net = torch.nn.DataParallel(net)
net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
  

Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in tqdm(range(1, args.warm_up_iter)):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float() # (bs, 64, dim)

    pred_motion, loss, perplexity = net(gt_motion)
    loss_motion = Loss(pred_motion, gt_motion)
    loss_vel = Loss.forward_joint(pred_motion, gt_motion)
    
    loss = loss_motion + args.commit * loss + args.loss_vel * loss_vel
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_perplexity, avg_commit,avg_classification = 0., 0., 0.,0.
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)

for nb_iter in tqdm(range(1, args.total_iter + 1)):
    #[256,64,263]
    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float() # bs, nb_joints, joints_dim, seq_len
    gt_idx = None
    if args.teacher_pth:
        gt_idx = teacher_net(gt_motion,type='encode')
    pred_motion, commit_loss, classification_loss, perplexity = net(gt_motion, gt_idx)

    loss_motion = Loss(pred_motion, gt_motion)
    loss_vel = Loss.forward_joint(pred_motion, gt_motion)
    
    loss = loss_motion + args.commit * (commit_loss+classification_loss) + args.loss_vel * loss_vel
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += commit_loss.item()
    avg_classification += classification_loss.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        writer.add_scalar('./Train/L1', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        writer.add_scalar('./Train/Classification', avg_classification, nb_iter)

        logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}\t Classification. {avg_classification:.5f}")
        
        avg_recons, avg_perplexity, avg_commit,avg_classification = 0., 0., 0.,0.

    if nb_iter % args.eval_iter==0 :
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper)
        