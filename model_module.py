import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.append('../')


class TimeBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size = 3, stride = 1, padding = 'valid', dilation = 1):
        super(TimeBlock, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, kernel_size), stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
class ChannelBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size = 3, stride = 1, padding = 'valid', dilation = 1):
        super(ChannelBlock, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(kernel_size, 1), stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
class DE(nn.Module):
    def __init__(self, in_channels, seq_len):
        super(DE, self).__init__()
        self.delta_low = 1
        self.delta_high = 3
        self.theta_low = 4
        self.theta_high = 7
        self.alpha_low = 8
        self.alpha_high = 13
        self.beta_low = 14
        self.beta_high = 30
        self.gamma_low = 31
        self.gamma_high = 50
        self.linear1 = nn.Linear(5, 1)
        self.linear2 = nn.Conv1d(in_channels, in_channels, kernel_size=5)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.seq_len = seq_len
    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.seq_len < 64:
            self.seq_len = 64 #padding
        x = torch.fft.fft(x, n=self.seq_len, dim=1)
        x = torch.abs(x)
        x1 = torch.sum(torch.pow(x[:, self.delta_low: self.delta_high, :], 2), dim=1, keepdim=True)
        x2 = torch.sum(torch.pow(x[:, self.theta_low: self.theta_high, :], 2), dim=1, keepdim=True)
        x3 = torch.sum(torch.pow(x[:, self.alpha_low: self.alpha_high, :], 2), dim=1, keepdim=True)
        x4 = torch.sum(torch.pow(x[:, self.beta_low: self.beta_high, :], 2), dim=1, keepdim=True)
        x5 = torch.sum(torch.pow(x[:, self.gamma_low: self.gamma_high, :], 2), dim=1, keepdim=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = torch.log2(x / self.seq_len)
        x = x.permute(0, 2, 1) 
        x = (self.alpha * F.relu(self.linear1(x)) + self.linear2(x))/2
        return x
class SSF_DST(nn.Module):
    def __init__(self, config):
        super(SSF_DST, self).__init__()
        in_channels, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        d_model = 16
        emb_size = d_model
        self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.BatchNorm1d(emb_size),
                    nn.Dropout(0.8),
                    nn.Linear(emb_size, 16),
                    nn.ReLU(),
                    nn.BatchNorm1d(16),
                    nn.Linear(16, 2))
        self.DE = DE(in_channels, seq_len)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.TimeLayer = nn.ModuleList([TimeBlock(1, d_model, kernel_size=3, stride=1, padding='same', dilation=1),  
                                        TimeBlock(d_model, d_model, kernel_size=3, stride=1, padding='same', dilation=2),
                                        TimeBlock(d_model,  d_model, kernel_size=3, stride=1, padding='same', dilation=2),
                                        TimeBlock(d_model,  d_model, kernel_size=3, stride=1, padding='same', dilation=2)])
        self.ChannelAvg = ChannelBlock(d_model, d_model, kernel_size=in_channels, stride=1, padding='valid')
    def forward(self, x):
        de = self.DE(x)
        x = x * de
        del de
        #B, C, T
        x = x.unsqueeze(1)  
        for layer in self.TimeLayer:
            x = layer(x)
        x = self.ChannelAvg(x)
        x = x.squeeze(2)
        x = self.avg(x)
        x = self.fc(x)
        return x