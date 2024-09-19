import torch.nn as nn
from models.resnet import Resnet1D
import torch
class PrintModule(nn.Module):
    def __init__(self, me=''):
        super().__init__()
        self.me = me

    def forward(self, x):
        print(self.me, x.shape)
        return x
    
class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Decoder_Speed(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, width, down_t, depth, dilation_growth_rate, activation, norm, condition_dim):
        super().__init__()
        self.condition_projector = nn.Conv1d(condition_dim, width, kernel_size=1)
        self.main_projector = nn.Conv1d(output_emb_width, width, kernel_size=1)
        
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, condition):
        condition_emb = self.condition_projector(condition)
        x = self.main_projector(x)
        x = torch.cat([x, condition_emb], dim=1)
        return self.blocks(x)

class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Decoder_Speed(nn.Module):
    def __init__(self,
                input_emb_width = 3,
                output_emb_width = 512,
                down_t = 3,
                width = 512,
                depth = 3,
                dilation_growth_rate = 3, 
                activation='relu',
                norm=None,
                condition_dim=8):
        super().__init__()
        self.condition_projector = nn.Conv1d(condition_dim, width, kernel_size=1)
        # nn.init.zeros_(self.condition_projector.bias)
        # if self.condition_projector.bias is not None:
        #     nn.init.zeros_(self.condition_projector.bias)
        self.main_projector = nn.Conv1d(output_emb_width, width, kernel_size=1)
        
        blocks = []
        blocks.append(nn.ReLU())

        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))

        self.model = nn.Sequential(*blocks)
    #TODO add or concat
    def forward(self, x, condition):
        condition_emb = self.condition_projector(condition)
        x = self.main_projector(x) + condition_emb
        #x = torch.cat([x, condition_emb], dim=1)
        return self.model(x)
    
