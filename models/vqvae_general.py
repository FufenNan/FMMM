import torch.nn as nn
from models.encdec import Encoder, Decoder
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

class VQVAE_decode_only(nn.Module):
    def __init__(self,
                 args,
                 teacher_net,
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

    def forward(self, x):
        x_emb = self.teacher_net(x,type='token_emb')
        ## quantization
        ## decoder
        x_decoder = self.decoder(x_emb)
        x_out = self.postprocess(x_decoder)
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

class HumanVQVAE_GENERAL(nn.Module):
    def __init__(self,
                 args,
                 coodebooks,
                 nb_code=512,
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
        
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_GENERAL(args, coodebooks,nb_code, code_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def forward(self, x, gt_idx=None,type='full'):
        '''type=[full, encode, decode]'''
        if type=='full':
            x_out, commit_loss, classification_loss, perplexity = self.vqvae(x,gt_idx)
            return x_out, commit_loss, classification_loss, perplexity
        elif type=='encode':
            b, t, c = x.size()
            quants = self.vqvae.encode(x) # (N, T)
            return quants
        elif type=='decode':
            x_out = self.vqvae.forward_decoder(x)
            return x_out
        else:
            raise ValueError(f'Unknown "{type}" type')
        