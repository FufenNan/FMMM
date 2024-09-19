import torch.nn as nn
import torch
from models.encdec import Encoder, Decoder, Decoder_Speed
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA_Frozen, QuantizeReset
from models.t2m_trans import Decoder_Transformer, Encoder_Transformer
from exit.utils import generate_src_mask

class VQVAE_GENERAL(nn.Module):
    def __init__(self,
                 args,
                 codebooks,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        output_dim = 251 if args.dataname == 'kit' else 263
        self.encoder = Encoder(output_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(output_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        
        self.quantizer = QuantizeEMA_Frozen(nb_code, code_dim, args)
        self.quantizer.load_codebooks(codebooks=codebooks)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x, gt_idx=None):
        
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        ## quantization
        x_quantized, commit_loss, classification_loss, perplexity  = self.quantizer(x_encoder, gt_idx)
        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, classification_loss, perplexity


    def forward_decoder(self, x):
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0

        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()

        # pad_mask = pad_mask.unsqueeze(1)
        # x_d = x_d * ~pad_mask
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out

class VQVAE_decode(nn.Module):
    def __init__(self,
                 args,
                 teacher_net,
                 nb_code=1024,
                 code_dim=32,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        output_dim = 251 if args.dataname == 'kit' else 263
        self.decoder = Decoder(output_dim, code_dim*5, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.teacher_net = teacher_net
        self.teacher_net.eval()


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x

    def forward(self, x,type='full'):
        if type == 'full':
            x_emb = self.teacher_net(x,type='motion_emb')
            x_decoder = self.decoder(x_emb)
            x_out = self.postprocess(x_decoder)
            x_out = self.teacher_net.shift_upper_up(x_out)
            return x_out
        elif type == 'decode':
            x_emb = self.teacher_net(x,type='token_emb')
            x_decoder = self.decoder(x_emb)
            x_out = self.postprocess(x_decoder)
            x_out = self.teacher_net.shift_upper_up(x_out)
            return x_out
        
    def forward_decoder(self, x):
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0

        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()

        # pad_mask = pad_mask.unsqueeze(1)
        # x_d = x_d * ~pad_mask
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out

class VQVAE_decode_speed(nn.Module):
    def __init__(self,
                 args,
                 teacher_net,
                 nb_code=1024,
                 code_dim=32,
                 down_t=3,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 condition_dim=8):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        output_dim = 251 if args.dataname == 'kit' else 263
        self.decoder = Decoder_Speed(output_dim, code_dim*5, down_t, width, depth, dilation_growth_rate, activation=activation, norm=norm,condition_dim=condition_dim)
        self.teacher_net = teacher_net
        self.teacher_net.eval()


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x

    def forward(self, x, speed=None, type='full'):
        if speed is None:
            speed = torch.zeros(x_emb.shape[0],8,x_emb.shape[2]).to(x.device)
        if speed.shape[1] != 8:
            speed = speed.permute(0,2,1)
        if type == 'full':
            x_emb = self.teacher_net(x,type='motion_emb')
            x_out = self.decoder(x_emb,speed)
            x_out = self.postprocess(x_out)
            x_out = self.teacher_net.shift_upper_up(x_out)
            return x_out
        elif type == 'decode':
            x_emb = self.teacher_net(x,type='token_emb')
            x_out = self.decoder(x_emb,speed)
            x_out = self.postprocess(x_out)
            x_out = self.teacher_net.shift_upper_up(x_out)
            return x_out
        
    def forward_decoder(self, x):
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0

        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()

        # pad_mask = pad_mask.unsqueeze(1)
        # x_d = x_d * ~pad_mask
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out
    