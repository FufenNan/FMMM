import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA_Frozen, QuantizeReset
from models.t2m_trans import Decoder_Transformer, Encoder_Transformer
from exit.utils import generate_src_mask
import torch
from utils.humanml_utils_v2 import UPPER_JOINT_Y_MASK,HML_LEFT_ARM_MASK,HML_RIGHT_ARM_MASK,HML_LEFT_LEG_MASK,HML_RIGHT_LEG_MASK,HML_ROOT_MASK,HML_SPINE_MASK,OVER_LAP_LOWER_MASK
import numpy as np
class VQVAE_MULTI_V2(nn.Module):
    def __init__(self,
                 args,
                 nb_code=256,
                 code_dim=32,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 moment=None,
                 sep_decoder=False):
        super().__init__()
        if args.dataname == 'kit':
            self.nb_joints = 21
            output_dim = 251
            upper_dim = 120        
            lower_dim = 131  
        else:
            self.nb_joints = 22
            output_dim = 263
            arm_dim=48
            leg_dim=59
            spine_dim=60
            root_dim=7
        self.code_dim = code_dim
        if moment is not None:
            self.moment = moment
            self.register_buffer('mean_upper', torch.tensor([0.1216, 0.2488, 0.2967, 0.5027, 0.4053, 0.4100, 0.5703, 0.4030, 0.4078, 0.1994, 0.1992, 0.0661, 0.0639], dtype=torch.float32))
            self.register_buffer('std_upper', torch.tensor([0.0164, 0.0412, 0.0523, 0.0864, 0.0695, 0.0703, 0.1108, 0.0853, 0.0847, 0.1289, 0.1291, 0.2463, 0.2484], dtype=torch.float32))
 
        self.sep_decoder = sep_decoder
        if self.sep_decoder:
            self.decoder_left_arm = Decoder(arm_dim, int(code_dim), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
            self.decoder_right_arm = Decoder(arm_dim, int(code_dim), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        
            self.decoder_right_leg = Decoder(leg_dim, int(code_dim), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
            self.decoder_left_leg = Decoder(leg_dim, int(code_dim), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)  
            self.decoder_spine = Decoder(spine_dim, int(code_dim), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)  
            self.decoder = Decoder(output_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        


        self.num_code = nb_code

        self.encoder_left_arm = Encoder(arm_dim, int(code_dim), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_right_arm = Encoder(arm_dim, int(code_dim), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        
        self.encoder_right_leg = Encoder(leg_dim, int(code_dim), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_left_leg = Encoder(leg_dim, int(code_dim), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm) 
        self.encoder_spine = Encoder(spine_dim, int(code_dim), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        
        self.quantizer_left_arm = QuantizeEMAReset(nb_code, int(code_dim), args)
        self.quantizer_right_arm = QuantizeEMAReset(nb_code, int(code_dim), args)
        self.quantizer_right_leg = QuantizeEMAReset(nb_code, int(code_dim), args)
        self.quantizer_left_leg = QuantizeEMAReset(nb_code, int(code_dim), args)
        self.quantizer_spine = QuantizeEMAReset(nb_code, int(code_dim), args)
        # self.quantizer_left_arm = QuantizeEMA(nb_code, int(code_dim), args)
        # self.quantizer_right_arm = QuantizeEMA(nb_code, int(code_dim), args)
        # self.quantizer_right_leg = QuantizeEMA(nb_code, int(code_dim), args)
        # self.quantizer_left_leg = QuantizeEMA(nb_code, int(code_dim), args)
        # self.quantizer_spine = QuantizeEMA(nb_code, int(code_dim), args)

        self.upper_map = []
        self.lower_map = []
        self.get_mapper()

    def rand_emb_idx(self, x_quantized, quantizer, idx_noise):
        # x_quantized = x_quantized.detach()
        x_quantized = x_quantized.permute(0,2,1)
        mask = torch.bernoulli(idx_noise * torch.ones((*x_quantized.shape[:2], 1),
                                                device=x_quantized.device))
        r_indices = torch.randint(int(self.num_code/2), x_quantized.shape[:2], device=x_quantized.device)
        r_emb = quantizer.dequantize(r_indices)
        x_quantized = mask * r_emb + (1-mask) * x_quantized
        x_quantized = x_quantized.permute(0,2,1)
        return x_quantized
    
    def normalize(self, data):
        return (data - self.moment['mean']) / self.moment['std']
    
    def denormalize(self, data):
        return data * self.moment['std'] + self.moment['mean']
    
    def normalize_upper(self, data):
        return (data - self.mean_upper) / self.std_upper
    
    def denormalize_upper(self, data):
        return data * self.std_upper + self.mean_upper

    def get_mapper(self):
        overlap_upper_mask = HML_RIGHT_ARM_MASK & HML_LEFT_ARM_MASK
        overlap_lower_mask = HML_RIGHT_LEG_MASK & HML_LEFT_LEG_MASK
        upper_indices = np.nonzero(overlap_upper_mask)[0]
        lower_indices = np.nonzero(overlap_lower_mask)[0]
        cnt=0
        for i,t in enumerate(HML_RIGHT_LEG_MASK):
            if i in lower_indices:
                self.lower_map.append(cnt)
                cnt+=1
            elif t:
                cnt+=1
        cnt=0
        for i,t in enumerate(HML_LEFT_ARM_MASK):
            if i in upper_indices:
                self.upper_map.append(cnt)
                cnt+=1
            elif t:
                cnt+=1

    def shift_upper_down(self, data):
        data = data.clone()
        data = self.denormalize(data)
        shift_y = data[..., 3:4].clone()
        data[..., UPPER_JOINT_Y_MASK] -= shift_y
        _data = data.clone()
        data = self.normalize(data)
        data[..., UPPER_JOINT_Y_MASK] = self.normalize_upper(_data[..., UPPER_JOINT_Y_MASK])
        return data
    
    def shift_upper_up(self, data):
        _data = data.clone()
        data = self.denormalize(data)
        data[..., UPPER_JOINT_Y_MASK] = self.denormalize_upper(_data[..., UPPER_JOINT_Y_MASK])
        shift_y = data[..., 3:4].clone()
        data[..., UPPER_JOINT_Y_MASK] += shift_y
        data = self.normalize(data)
        return data
    
    def merge_upper_lower(self,left_arm_emb, right_arm_emb, right_leg_emb, left_leg_emb, spine_emb):
        motion = torch.empty(*left_arm_emb.shape[:2], 263).to(left_arm_emb.device)
        motion[..., HML_LEFT_ARM_MASK] = left_arm_emb
        motion[..., HML_RIGHT_ARM_MASK] = right_arm_emb
        motion[..., HML_RIGHT_LEG_MASK] = right_leg_emb
        motion[..., HML_LEFT_LEG_MASK] = left_leg_emb
        motion[..., HML_SPINE_MASK] = spine_emb
        motion[...,OVER_LAP_LOWER_MASK]=(left_leg_emb[...,self.lower_map]+right_leg_emb[...,self.lower_map])/2
        return motion
    
    def forward(self, x, *args, type='full', **kwargs):
        '''type=[full, encode, decode]'''
        if type=='full':
            x = x.float()
            #TODO fix shift upper down
            x = self.shift_upper_down(x)
            left_arm_emb = x[..., HML_LEFT_ARM_MASK]
            right_arm_emb = x[..., HML_RIGHT_ARM_MASK]
            left_leg_emb = x[..., HML_LEFT_LEG_MASK]
            right_leg_emb = x[..., HML_RIGHT_LEG_MASK]
            spine_emb = x[..., HML_SPINE_MASK]

            left_arm_emb = self.preprocess(left_arm_emb)
            left_arm_emb = self.encoder_left_arm(left_arm_emb)
            left_arm_emb, loss_left_arm, perplexity = self.quantizer_left_arm(left_arm_emb)
            
            right_arm_emb = self.preprocess(right_arm_emb)
            right_arm_emb = self.encoder_right_arm(right_arm_emb)
            right_arm_emb, loss_right_arm, perplexity = self.quantizer_right_arm(right_arm_emb)

            left_leg_emb = self.preprocess(left_leg_emb)
            left_leg_emb = self.encoder_left_leg(left_leg_emb)
            left_leg_emb, loss_left_leg, perplexity = self.quantizer_left_leg(left_leg_emb)

            right_leg_emb = self.preprocess(right_leg_emb)
            right_leg_emb = self.encoder_right_leg(right_leg_emb)
            right_leg_emb, loss_right_leg, perplexity = self.quantizer_right_leg(right_leg_emb)

            spine_emb = self.preprocess(spine_emb)
            spine_emb = self.encoder_spine(spine_emb)
            spine_emb, loss_spine, perplexity = self.quantizer_spine(spine_emb)

            loss = loss_left_arm + loss_right_arm + loss_left_leg + loss_right_leg + loss_spine

            #TODO check rand_emb_idx
            if 'idx_noise' in kwargs and kwargs['idx_noise'] > 0:
                upper_emb = self.rand_emb_idx(upper_emb, self.quantizer_upper, kwargs['idx_noise'])
                lower_emb = self.rand_emb_idx(lower_emb, self.quantizer_lower, kwargs['idx_noise'])


            # x_in = self.preprocess(x)
            # x_encoder = self.encoder(x_in)
        
            # ## quantization
            # x_quantized, loss, perplexity  = self.quantizer(x_encoder)

            ## decoder
            if self.sep_decoder:
                x_d_left_arm = self.decoder_left_arm(left_arm_emb)
                x_d_left_arm = self.postprocess(x_d_left_arm)
                x_d_right_arm = self.decoder_right_arm(right_arm_emb)
                x_d_right_arm = self.postprocess(x_d_right_arm)
                x_d_right_leg = self.decoder_right_leg(right_leg_emb)
                x_d_right_leg = self.postprocess(x_d_right_leg)
                x_d_left_leg = self.decoder_left_leg(left_leg_emb)
                x_d_left_leg = self.postprocess(x_d_left_leg)
                x_d_spine = self.decoder_spine(spine_emb)
                x_d_spine = self.postprocess(x_d_spine)

                x_out = self.merge_upper_lower(x_d_left_arm, x_d_right_arm, x_d_right_leg, x_d_left_leg, x_d_spine)
                x_out = self.shift_upper_up(x_out)

            else:
                x_quantized = torch.cat([left_arm_emb, right_arm_emb, right_leg_emb, left_leg_emb,spine_emb], dim=1)
                x_decoder = self.decoder(x_quantized)
                x_out = self.postprocess(x_decoder)            
            return x_out, loss, perplexity
        elif type=='encode':
            N, T, _ = x.shape
            x = self.shift_upper_down(x)

            left_arm_emb = x[..., HML_LEFT_ARM_MASK]
            left_arm_emb = self.preprocess(left_arm_emb)
            left_arm_emb = self.encoder_left_arm(left_arm_emb)
            left_arm_emb = self.postprocess(left_arm_emb)
            left_arm_emb = left_arm_emb.reshape(-1, left_arm_emb.shape[-1])
            left_arm_code_idx = self.quantizer_left_arm.quantize(left_arm_emb)
            left_arm_code_idx = left_arm_code_idx.view(N, -1)

            right_arm_emb = x[..., HML_RIGHT_ARM_MASK]
            right_arm_emb = self.preprocess(right_arm_emb)
            right_arm_emb = self.encoder_right_arm(right_arm_emb)
            right_arm_emb = self.postprocess(right_arm_emb)
            right_arm_emb = right_arm_emb.reshape(-1, right_arm_emb.shape[-1])
            right_arm_code_idx = self.quantizer_right_arm.quantize(right_arm_emb)
            right_arm_code_idx = right_arm_code_idx.view(N, -1)

            left_leg_emb = x[..., HML_LEFT_LEG_MASK]
            left_leg_emb = self.preprocess(left_leg_emb)
            left_leg_emb = self.encoder_left_leg(left_leg_emb)
            left_leg_emb = self.postprocess(left_leg_emb)
            left_leg_emb = left_leg_emb.reshape(-1, left_leg_emb.shape[-1])
            left_leg_code_idx = self.quantizer_left_leg.quantize(left_leg_emb)
            left_leg_code_idx = left_leg_code_idx.view(N, -1)

            right_leg_emb = x[..., HML_RIGHT_LEG_MASK]
            right_leg_emb = self.preprocess(right_leg_emb)
            right_leg_emb = self.encoder_right_leg(right_leg_emb)
            right_leg_emb = self.postprocess(right_leg_emb)
            right_leg_emb = right_leg_emb.reshape(-1, right_leg_emb.shape[-1])
            right_leg_code_idx = self.quantizer_right_leg.quantize(right_leg_emb)
            right_leg_code_idx = right_leg_code_idx.view(N, -1)

            spine_emb = x[..., HML_SPINE_MASK]
            spine_emb = self.preprocess(spine_emb)
            spine_emb = self.encoder_spine(spine_emb)
            spine_emb = self.postprocess(spine_emb)
            spine_emb = spine_emb.reshape(-1, spine_emb.shape[-1])
            spine_code_idx = self.quantizer_spine.quantize(spine_emb)
            spine_code_idx = spine_code_idx.view(N, -1)

            code_idx = torch.cat([left_arm_code_idx.unsqueeze(-1), right_arm_code_idx.unsqueeze(-1), left_leg_code_idx.unsqueeze(-1), \
                                  right_leg_code_idx.unsqueeze(-1),spine_code_idx.unsqueeze(-1)], dim=-1)
            return code_idx

        elif type=='decode':
            if self.sep_decoder:
                x_d_left_arm = self.quantizer_left_arm.dequantize(x[..., 0])
                x_d_left_arm = x_d_left_arm.permute(0, 2, 1).contiguous()
                x_d_left_arm = self.decoder_left_arm(x_d_left_arm)
                x_d_left_arm = self.postprocess(x_d_left_arm)

                x_d_right_arm = self.quantizer_right_arm.dequantize(x[..., 1])
                x_d_right_arm = x_d_right_arm.permute(0, 2, 1).contiguous()
                x_d_right_arm = self.decoder_right_arm(x_d_right_arm)
                x_d_right_arm = self.postprocess(x_d_right_arm)

                x_d_left_leg = self.quantizer_left_leg.dequantize(x[..., 2])
                x_d_left_leg = x_d_left_leg.permute(0, 2, 1).contiguous()
                x_d_left_leg = self.decoder_left_leg(x_d_left_leg)
                x_d_left_leg = self.postprocess(x_d_left_leg)

                x_d_right_leg = self.quantizer_right_leg.dequantize(x[..., 3])
                x_d_right_leg = x_d_right_leg.permute(0, 2, 1).contiguous()
                x_d_right_leg = self.decoder_right_leg(x_d_right_leg)
                x_d_right_leg = self.postprocess(x_d_right_leg)
                
                x_d_spine = self.quantizer_spine.dequantize(x[..., 4])
                x_d_spine = x_d_spine.permute(0, 2, 1).contiguous()
                x_d_spine = self.decoder_spine(x_d_spine)
                x_d_spine = self.postprocess(x_d_spine)

                x_out= self.merge_upper_lower(x_d_left_arm, x_d_right_arm, x_d_right_leg, x_d_left_leg, x_d_spine)
                x_out = self.shift_upper_up(x_out)
                return x_out
            else:
                x_d_left_arm = self.quantizer_left_arm.dequantize(x[..., 0])
                x_d_right_arm = self.quantizer_right_arm.dequantize(x[..., 1])
                x_d_left_leg = self.quantizer_left_leg.dequantize(x[..., 2])
                x_d_right_leg = self.quantizer_right_leg.dequantize(x[..., 3])
                x_d_spine = self.quantizer_spine.dequantize(x[..., 4])
                x_d = torch.cat([x_d_left_arm, x_d_right_arm, x_d_right_leg, x_d_left_leg,x_d_spine], dim=-1)
                x_d = x_d.permute(0, 2, 1).contiguous()
                x_decoder = self.decoder(x_d)
                x_out = self.postprocess(x_decoder)
                return x_out
        elif type=='motion_emb':
            N, T, _ = x.shape
            # TODO shift upper down in decoder?
            x = self.shift_upper_down(x)

            left_arm_emb = x[..., HML_LEFT_ARM_MASK]
            left_arm_emb = self.preprocess(left_arm_emb)
            left_arm_emb = self.encoder_left_arm(left_arm_emb)
            left_arm_emb = self.postprocess(left_arm_emb)
            left_arm_emb = left_arm_emb.reshape(-1, left_arm_emb.shape[-1])
            left_arm_code_idx = self.quantizer_left_arm.quantize(left_arm_emb)
            left_arm_code_idx = left_arm_code_idx.view(N, -1)
            left_arm_emb = self.quantizer_left_arm.dequantize(left_arm_code_idx)

            right_arm_emb = x[..., HML_RIGHT_ARM_MASK]
            right_arm_emb = self.preprocess(right_arm_emb)
            right_arm_emb = self.encoder_right_arm(right_arm_emb)
            right_arm_emb = self.postprocess(right_arm_emb)
            right_arm_emb = right_arm_emb.reshape(-1, right_arm_emb.shape[-1])
            right_arm_code_idx = self.quantizer_right_arm.quantize(right_arm_emb)
            right_arm_code_idx = right_arm_code_idx.view(N, -1)
            right_arm_emb = self.quantizer_right_arm.dequantize(right_arm_code_idx)

            left_leg_emb = x[..., HML_LEFT_LEG_MASK]
            left_leg_emb = self.preprocess(left_leg_emb)
            left_leg_emb = self.encoder_left_leg(left_leg_emb)
            left_leg_emb = self.postprocess(left_leg_emb)
            left_leg_emb = left_leg_emb.reshape(-1, left_leg_emb.shape[-1])
            left_leg_code_idx = self.quantizer_left_leg.quantize(left_leg_emb)
            left_leg_code_idx = left_leg_code_idx.view(N, -1)
            left_leg_emb = self.quantizer_left_leg.dequantize(left_leg_code_idx)

            right_leg_emb = x[..., HML_RIGHT_LEG_MASK]
            right_leg_emb = self.preprocess(right_leg_emb)
            right_leg_emb = self.encoder_right_leg(right_leg_emb)
            right_leg_emb = self.postprocess(right_leg_emb)
            right_leg_emb = right_leg_emb.reshape(-1, right_leg_emb.shape[-1])
            right_leg_code_idx = self.quantizer_right_leg.quantize(right_leg_emb)
            right_leg_code_idx = right_leg_code_idx.view(N, -1)
            right_leg_emb = self.quantizer_right_leg.dequantize(right_leg_code_idx)

            spine_emb = x[..., HML_SPINE_MASK]
            spine_emb = self.preprocess(spine_emb)
            spine_emb = self.encoder_spine(spine_emb)
            spine_emb = self.postprocess(spine_emb)
            spine_emb = spine_emb.reshape(-1, spine_emb.shape[-1])
            spine_code_idx = self.quantizer_spine.quantize(spine_emb)
            spine_code_idx = spine_code_idx.view(N, -1)
            spine_emb = self.quantizer_spine.dequantize(spine_code_idx)

            x_emb= torch.cat([left_arm_emb, right_arm_emb, left_leg_emb,right_leg_emb,spine_emb], dim=-1)
            return x_emb.permute(0,2,1).contiguous()
        elif type=='token_emb':
            left_arm_emb = self.quantizer_left_arm.dequantize(x[..., 0])
            right_arm_emb = self.quantizer_right_arm.dequantize(x[..., 1])
            left_leg_emb = self.quantizer_left_leg.dequantize(x[..., 2])
            right_leg_emb = self.quantizer_right_leg.dequantize(x[..., 3])
            spine_emb = self.quantizer_spine.dequantize(x[..., 4])

            x_emb= torch.cat([left_arm_emb, right_arm_emb, left_leg_emb,right_leg_emb,spine_emb], dim=-1)
            return x_emb.permute(0,2,1).contiguous()

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x
    
    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x