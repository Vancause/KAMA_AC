import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Conv_DW(nn.Module):
    def __init__(self, in_channel, out_channel, stride=(1,1)):
        super(Conv_DW, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True),)

    def forward(self, x):
        # pdb.set_trace()
        return self.conv(x)
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        # after avg_pool
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            # fc->bn
            self.gate_c.add_module('gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]))
            # self.gate_c.add_module('gate_c_bn_%d'%(i+1), nn.BatchNorm2d(gate_channels[i+1]))
            self.gate_c.add_module('gate_c_relu_%d'%(i+1), nn.ReLU())
        # final_fc
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        # Global avg pool
        # pdb.set_trace()
        avg_pool = F.avg_pool2d(in_tensor, (in_tensor.size(2), in_tensor.size(3)), stride=(in_tensor.size(2), in_tensor.size(3))).squeeze(-1).squeeze(-1)
        # C∗H∗W -> C*1*1 -> C*H*W
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatiaGate(nn.Module):
    # dilation value and reduction ratio, set d = 4 r = 16
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatiaGate, self).__init__()
        self.gate_s = nn.Sequential()
        # 1x1 + (3x3)*2 + 1x1
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i, nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                                                             kernel_size=3, padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))  # 1×H×W

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)

class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatiaGate(gate_channel)

    def forward(self, in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor

class EN_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EN_Block, self).__init__()
        
        self.dw_con1 = Conv_DW(in_channel, in_channel//2)
        self.bam1 = BAM(in_channel//2)
        self.dw_con2 = Conv_DW(in_channel//2, out_channel)
        self.bam2 = BAM(out_channel)
        self.conv = nn.Conv2d(out_channel, out_channel, 1, 1, 0, bias=False)
    def forward(self, x):
        
        x = self.dw_con1(x)
        x = self.bam1(x)
        x = self.dw_con2(x)
        x = self.bam2(x)

        x = self.conv(x)
        
        return x
if __name__== '__main__':
    import pdb
    
    input_t = torch.rand(2,2048, 72, 2)
    bam = EN_Block(2048, 1024)
    out = bam(input_t)
    pdb.set_trace()