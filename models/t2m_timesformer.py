import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding
from exit.utils import cosine_schedule, uniform, top_k, gumbel_sample, top_p
from tqdm import tqdm
from einops import rearrange, repeat
from exit.utils import get_model, generate_src_mask

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from timesformer_pytorch.rotary import apply_rot_emb, AxialRotaryEmbedding, RotaryEmbedding

# helpers

def exists(val):
    return val is not None

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# time token shift

def shift(t, amt):
    if amt is 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))

class PreTokenShift(nn.Module):
    def __init__(self, frames, fn):
        super().__init__()
        self.frames = frames
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        f, dim = self.frames, x.shape[-1]
        cls_x, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (f n) d -> b f n d', f = f)

        # shift along time frame before and after

        dim_chunk = (dim // 3)
        chunks = x.split(dim_chunk, dim = -1)
        chunks_to_shift, rest = chunks[:3], chunks[3:]
        shifted_chunks = tuple(map(lambda args: shift(*args), zip(chunks_to_shift, (-1, 0, 1))))
        x = torch.cat((*shifted_chunks, *rest), dim = -1)

        x = rearrange(x, 'b f n d -> b (f n) d')
        x = torch.cat((cls_x, x), dim = 1)
        return self.fn(x, *args, **kwargs)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

def attn(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)

    if exists(mask):
        max_neg_value = torch.float('-inf')
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class Time_Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, mask = None, rot_emb = None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        q = q * self.scale
        # rearrange across time or space
        q, k, v = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q, k, v))
        # add rotary embeddings, if applicable
        if exists(rot_emb):
            q, k = apply_rot_emb(q, k, rot_emb)
        # attention
        out = attn(q, k, v, mask = mask)
        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)
        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        # combine heads out
        return self.to_out(out)

# main classes

class TimeSformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        image_height = 5,
        image_width = 32,
        patch_height = 1,
        patch_width = 32,
        channels = 1,
        depth = 1,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.1,
        ff_dropout = 0.1,
        rotary_emb = True,
        shift_tokens = False
    ):
        super().__init__()
        assert image_height % patch_height == 0, 'Image height must be divisible by the patch height.'
        assert image_width % patch_width == 0, 'Image width must be divisible by the patch width.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_height * patch_width

        self.heads = heads
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions + 1, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout = ff_dropout)
            time_attn = Time_Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            spatial_attn = Time_Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)

            if shift_tokens:
                time_attn, spatial_attn, ff = map(lambda t: PreTokenShift(num_frames, t), (time_attn, spatial_attn, ff))

            time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff))

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))


    def forward(self, video, mask = None):
            b, f, _, h, w, *_, device, ph, pw = *video.shape, video.device, self.patch_height, self.patch_width
            assert h % ph == 0 and w % pw == 0, f'height {h} and width {w} of video must be divisible by the patch height {ph} and patch width {pw}'
            # calculate num patches in height and width dimension, and number of total patches (n)
            hp, wp = (h // ph), (w // pw)
            n = hp * wp
            # video to patch embeddings
            x = rearrange(video, 'b f c (hp ph) (wp pw) -> b (f hp wp) (ph pw c)', ph=ph, pw=pw)
            # tokens = self.to_patch_embedding(video)
            # add cls token TODO add text token
            # positional embedding
            frame_pos_emb = None
            image_pos_emb = None
            if not self.use_rotary_emb:
                x += self.pos_emb(torch.arange(x.shape[1], device = device))
            else:
                frame_pos_emb = self.frame_rot_emb(f, device = device)
                # TODO delete
                image_pos_emb = self.image_rot_emb(hp, wp, device = device)
            # calculate masking for uneven number of frames
            frame_mask = None
            # if exists(mask):
            #     mask_with_cls = F.pad(mask, (1, 0), value = True)
            #     frame_mask = repeat(mask_with_cls, 'b f -> (b h n) () f', n = n, h = self.heads)
            # time and space attention
            for (time_attn, spatial_attn, ff) in self.layers:
                x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, rot_emb = frame_pos_emb) + x
                x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, rot_emb = image_pos_emb) + x
                x = ff(x) + x
            return x


class Text2Motion_Transformer(nn.Module):

    def __init__(self, 
                vqvae,
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                num_local_layer=0, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.n_head = n_head
        self.trans_base = CrossCondTransBase(vqvae, num_vq, embed_dim, clip_dim, block_size, num_layers, num_local_layer, n_head, drop_out_rate, fc_rate)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq

        # self.skip_trans = Skip_Connection_Transformer(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)

    def get_block_size(self):
        return self.block_size

    def forward(self, *args, type='forward', **kwargs):
        '''type=[forward, sample]'''
        if type=='forward':
            return self.forward_function(*args, **kwargs)
        elif type=='sample':
            return self.sample(*args, **kwargs)
        elif type=='inpaint':
            return self.inpaint(*args, **kwargs)
        else:
            raise ValueError(f'Unknown "{type}" type')
        
    def get_attn_mask(self, src_mask, att_txt=None):
        if att_txt is None:
            att_txt = torch.tensor([[True]]*src_mask.shape[0]).to(src_mask.device)
        src_mask = torch.cat([att_txt, src_mask],  dim=1)
        B, T = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)
        return src_mask

    def forward_function(self, idxs, clip_feature, src_mask=None, att_txt=None, word_emb=None):
        if src_mask is not None:
            src_mask = self.get_attn_mask(src_mask, att_txt)
        feat = self.trans_base(idxs, clip_feature, src_mask, word_emb)#[bs,51,1,5,d]
        logits = self.trans_head(feat, src_mask)

        return logits

    def sample(self, clip_feature, word_emb, m_length=None, if_test=False, rand_pos=True, CFG=-1, token_cond=None, max_steps=10):
        max_length = 49
        batch_size = clip_feature.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq
        shape = (batch_size, 5, self.block_size - 1)
        topk_filter_thres = .9
        starting_temperature = 1.0
        scores = torch.ones(shape, dtype=torch.float32, device=clip_feature.device)
        
        m_tokens_len = torch.ceil((m_length) / 4).long()
        src_token_mask = generate_src_mask(self.block_size - 1, m_tokens_len + 1).unsqueeze(1).repeat(1,5,1)
        src_token_mask_noend = generate_src_mask(self.block_size - 1, m_tokens_len).unsqueeze(1).repeat(1,5,1)
        if token_cond is not None:
            ids = token_cond.clone()
            ids[~src_token_mask_noend] = pad_id
            num_token_cond = (ids == mask_id).sum(-1)
        else:
            ids = torch.full(shape, mask_id, dtype=torch.long, device=clip_feature.device)
        
        ids[~src_token_mask] = pad_id
        ids.scatter_(-1, m_tokens_len[..., None, None].long().repeat(1, 5, 1), end_id)
        
        sample_max_steps = torch.round(max_steps / max_length * m_tokens_len) + 1e-8
        for step in range(max_steps):
            timestep = torch.clip(step / (sample_max_steps), max=1)
            if len(m_tokens_len) == 1 and step > 0 and torch.clip(step - 1 / (sample_max_steps), max=1).cpu().item() == timestep:
                break
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)
            
            if token_cond is not None:
                num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
                scores[token_cond != mask_id] = 0
            
            scores[~src_token_mask_noend] = 0
            scores = scores / scores.sum(-1, keepdim=True)
            
            sorted, sorted_score_indices = scores.sort(dim=-1,descending=True)
            
            ids[~src_token_mask] = pad_id
            ids.scatter_(-1, m_tokens_len[..., None, None].long().repeat(1, 5, 1), end_id)
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[-1], num_token_masked).unsqueeze(1).repeat(1, 5, 1)
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1).unsqueeze(-1) - 1).repeat(1, 5, 1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index * ~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices, mask_id)
            logits = self.forward(ids.permute(0,2,1), clip_feature, src_token_mask[...,0], word_emb=word_emb)[:, 1:]
            logits = logits.squeeze(2)
            filtered_logits = logits
            if rand_pos:
                temperature = 1
            else:
                temperature = 0
            
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1).permute(0, 2, 1)
            is_mask = ids == mask_id
            ids = torch.where(
                is_mask,
                pred_ids,
                ids
            )
            probs_without_temperature = logits.softmax(dim=-1).permute(0, 2, 1, 3)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, 0)
        if if_test:
            return ids.permute(0,2,1)
        return ids.permute(0,2,1)
    
    def inpaint(self, tokens, clip_feature, word_emb, edit_idxs, m_length=None, if_test=False, rand_pos=True, CFG=-1, token_cond=None, max_steps=10):
        max_length = 49
        batch_size = clip_feature.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq
        shape = (batch_size, 5, self.block_size - 1)
        topk_filter_thres = .9
        starting_temperature = 1.0
        scores = torch.ones(shape, dtype=torch.float32, device=clip_feature.device)
        con_idxs = torch.tensor([i for i in range(5) if i not in edit_idxs]).to(clip_feature.device)
        m_tokens_len = torch.ceil((m_length) / 4).long()
        src_token_mask = generate_src_mask(self.block_size - 1, m_tokens_len + 1).unsqueeze(1).repeat(1,5,1)
        src_token_mask_noend = generate_src_mask(self.block_size - 1, m_tokens_len).unsqueeze(1).repeat(1,5,1)
        ids = tokens.clone()
        con_ids = tokens[:, con_idxs, :].clone()
        # ids[~src_token_mask] = pad_id
        # ids.scatter_(-1, m_tokens_len[..., None, None].long().repeat(1, 5, 1), end_id)
        sample_max_steps = torch.round(max_steps / max_length * m_tokens_len) + 1e-8
        for step in range(max_steps):
            timestep = torch.clip(step / (sample_max_steps), max=1)
            if len(m_tokens_len) == 1 and step > 0 and torch.clip(step - 1 / (sample_max_steps), max=1).cpu().item() == timestep:
                break
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)
            # if token_cond is not None:
            #     num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
            #     scores[token_cond != mask_id] = 0
            scores[~src_token_mask_noend] = 0
            scores = scores / scores.sum(-1, keepdim=True)
            
            sorted, sorted_score_indices = scores.sort(dim=-1,descending=True)
            
            ids[~src_token_mask] = pad_id
            ids.scatter_(-1, m_tokens_len[..., None, None].long().repeat(1, 5, 1), end_id)
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[-1], num_token_masked).unsqueeze(1).repeat(1, 5, 1)
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1).unsqueeze(-1) - 1).repeat(1, 5, 1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index * ~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices[:,edit_idxs,:], mask_id)
            logits = self.forward(ids.permute(0,2,1), clip_feature, src_token_mask[...,0], word_emb=word_emb)[:, 1:]
            logits = logits.squeeze(2)
            filtered_logits = logits
            if rand_pos:
                temperature = 1
            else:
                temperature = 0
            
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1).permute(0, 2, 1)
            is_mask = ids == mask_id
            ids = torch.where(
                is_mask,
                pred_ids,
                ids
            )
            probs_without_temperature = logits.softmax(dim=-1).permute(0, 2, 1, 3)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, 0)
            ids[:,con_idxs,:] = con_ids
        if if_test:
            return ids.permute(0,2,1)
        return ids.permute(0,2,1)
    
    # def inpaint(self, first_tokens, last_tokens, clip_feature=None, word_emb=None, inpaint_len=2, rand_pos=False):
    #     # support only one sample
    #     assert first_tokens.shape[0] == 1
    #     assert last_tokens.shape[0] == 1
    #     max_steps = 20
    #     max_length = 49
    #     batch_size = first_tokens.shape[0]
    #     mask_id = self.num_vq + 2
    #     pad_id = self.num_vq + 1
    #     end_id = self.num_vq
    #     shape = (batch_size, self.block_size - 1)
    #     scores = torch.ones(shape, dtype = torch.float32, device = first_tokens.device)
        
    #     # force add first / last tokens
    #     first_partition_pos_idx = first_tokens.shape[1]
    #     second_partition_pos_idx = first_partition_pos_idx + inpaint_len
    #     end_pos_idx = second_partition_pos_idx + last_tokens.shape[1]

    #     m_tokens_len = torch.ones(batch_size, device = first_tokens.device)*end_pos_idx

    #     src_token_mask = generate_src_mask(self.block_size-1, m_tokens_len+1)
    #     src_token_mask_noend = generate_src_mask(self.block_size-1, m_tokens_len)
    #     ids = torch.full(shape, mask_id, dtype = torch.long, device = first_tokens.device)
        
    #     ids[:, :first_partition_pos_idx] = first_tokens
    #     ids[:, second_partition_pos_idx:end_pos_idx] = last_tokens
    #     src_token_mask_noend[:, :first_partition_pos_idx] = False
    #     src_token_mask_noend[:, second_partition_pos_idx:end_pos_idx] = False
        
    #     # [TODO] confirm that these 2 lines are not neccessary (repeated below and maybe don't need them at all)
    #     ids[~src_token_mask] = pad_id # [INFO] replace with pad id
    #     ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id

    #     temp = []
    #     sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8

    #     if clip_feature is None:
    #         clip_feature = torch.zeros(1, 512).to(first_tokens.device)
    #         att_txt = torch.zeros((batch_size,1), dtype=torch.bool, device = first_tokens.device)
    #     else:
    #         att_txt = torch.ones((batch_size,1), dtype=torch.bool, device = first_tokens.device)

    #     for step in range(max_steps):
    #         timestep = torch.clip(step/(sample_max_steps), max=1)
    #         rand_mask_prob = cosine_schedule(timestep) # timestep #
    #         num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)
    #         # [INFO] rm no motion frames
    #         scores[~src_token_mask_noend] = 0
    #         # [INFO] rm begin and end frames
    #         scores[:, :first_partition_pos_idx] = 0
    #         scores[:, second_partition_pos_idx:end_pos_idx] = 0
    #         scores = scores/scores.sum(-1)[:, None] # normalize only unmasked token
            
    #         sorted, sorted_score_indices = scores.sort(descending=True) # deterministic
            
    #         ids[~src_token_mask] = pad_id # [INFO] replace with pad id
    #         ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id
    #         ## [INFO] Replace "mask_id" to "ids" that have highest "num_token_masked" "scores" 
    #         select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
    #         # [INFO] repeat last_id to make it scatter_ the existing last ids.
    #         last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1)-1)
    #         sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index*~select_masked_indices)
    #         ids.scatter_(-1, sorted_score_indices, mask_id)

    #         # [TODO] force replace begin/end tokens b/c the num mask will be more than actual inpainting frames
    #         ids[:, :first_partition_pos_idx] = first_tokens
    #         ids[:, second_partition_pos_idx:end_pos_idx] = last_tokens
            
    #         logits = self.forward(ids, clip_feature, src_token_mask, word_emb=word_emb)[:,1:]
    #         filtered_logits = logits #top_k(logits, topk_filter_thres)
    #         if rand_pos:
    #             temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
    #         else:
    #             temperature = 0 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

    #         # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
    #         # pred_ids = filtered_logits.argmax(dim = -1)
    #         pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
    #         is_mask = ids == mask_id
    #         temp.append(is_mask[:1])
            
    #         ids = torch.where(
    #                     is_mask,
    #                     pred_ids,
    #                     ids
    #                 )
            
    #         probs_without_temperature = logits.softmax(dim = -1)
    #         scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
    #         scores = rearrange(scores, '... 1 -> ...')
    #         scores = scores.masked_fill(~is_mask, 0)
    #     return ids

class Time_Block(nn.Module):
    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = TimeSformer(
            dim=embed_dim,
            num_frames=block_size,
            image_height=5,
            image_width=embed_dim,
            patch_height=1,
            patch_width=embed_dim,
            depth=1,
            heads=n_head,
            dim_head=embed_dim//n_head,
            ff_dropout=drop_out_rate,
            attn_dropout=drop_out_rate)

    def forward(self, x, src_mask=None):
        output = self.attn(self.ln1(x), src_mask)
        x = x + output.reshape(x.shape)
        return x
    
class CrossAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, 77)).view(1, 1, block_size, 77))
        self.n_head = n_head

    def forward(self, x,word_emb):
        B, T, C = x.size()
        B, N, D = word_emb.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, N) -> (B, nh, T, N)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, N) x (B, nh, N, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block_crossatt(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.attn = CrossAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x,word_emb):
        x = x + self.attn(self.ln1(x), self.ln3(word_emb))
        x = x + self.mlp(self.ln2(x))
        return x

class CrossCondTransBase(nn.Module):

    def __init__(self, 
                vqvae,
                num_vq=1024, 
                embed_dim=256, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                num_local_layer = 1,
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.vqvae = vqvae
        # self.tok_emb = nn.Embedding(num_vq + 3, embed_dim).requires_grad_(False) 
        self.learn_tok_emb = nn.Embedding(3, int(self.vqvae.vqvae.code_dim))# [INFO] 3 = [end_id, blank_id, mask_id]
        self.to_emb = nn.Linear(self.vqvae.vqvae.code_dim, embed_dim)

        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Time_Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers-num_local_layer)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.num_local_layer = num_local_layer
        if num_local_layer > 0:
            self.word_emb = nn.Linear(clip_dim, embed_dim*5)
            self.cross_att = nn.Sequential(*[Block_crossatt(embed_dim*5, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_local_layer)])
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self,idx, clip_feature, src_mask, word_emb):
        #TODO
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx.shape[:2]
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            learn_idx = [idx[...,i] >= self.vqvae.vqvae.num_code for i in range(5)]

            code_dim = self.vqvae.vqvae.code_dim
            token_embeddings = torch.empty((*idx.shape, code_dim), device=idx[0].device)
            quantizers = [
                self.vqvae.vqvae.teacher_net.quantizer_left_arm,
                self.vqvae.vqvae.teacher_net.quantizer_right_arm,
                self.vqvae.vqvae.teacher_net.quantizer_left_leg,
                self.vqvae.vqvae.teacher_net.quantizer_right_leg,
                self.vqvae.vqvae.teacher_net.quantizer_spine
            ]
            for i in range(5):
                token_embeddings[:,:,i,:][~learn_idx[i]] = quantizers[i].dequantize(idx[...,i][~learn_idx[i]]).requires_grad_(False)
                token_embeddings[:,:,i,:][learn_idx[i]] = self.learn_tok_emb(idx[...,i][learn_idx[i]] - self.vqvae.vqvae.num_code)
            token_embeddings = self.to_emb(token_embeddings)  # [bs,t,5,d]

            if self.num_local_layer > 0:
                token_embeddings = token_embeddings.view(b, t, -1)
                word_emb = self.word_emb(word_emb)
                token_embeddings = self.pos_embed(token_embeddings)  # [bs,50,5*256]
                for module in self.cross_att:
                    token_embeddings = module(token_embeddings, word_emb)
                token_embeddings = token_embeddings.view(b, t, 5, -1)
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
        x = token_embeddings.unsqueeze(2)
        if len(x.shape)==4:
            x = x.unsqueeze(2) #[bs,t,1,51,d]
        for block in self.blocks:
            x = block(x, src_mask)

        return x  # [bs,50,1024]


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Time_Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, src_mask):
        if len(x.shape) == 4:
            x = x.squeeze(2)
        for block in self.blocks:
            x = block(x, src_mask)
        x = self.ln_f(x)
        #TODO check head
        logits = self.head(x)
        return logits

    


        

