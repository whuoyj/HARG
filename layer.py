import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
from config import config




class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, act=F.relu, dropout=0.5, bias=True):
        super(GraphConvolution, self).__init__()

        self.alpha = 1.

        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.bias = None

        for w in [self.weight]:
            nn.init.xavier_normal_(w)

    def normalize(self, m):
        rowsum = torch.sum(m, 0)
        r_inv = torch.pow(rowsum, -0.5)
        r_mat_inv = torch.diag(r_inv).float()

        m_norm = torch.mm(r_mat_inv, m)
        m_norm = torch.mm(m_norm, r_mat_inv)

        return m_norm

    def forward(self, adj, x):

        x = self.dropout(x)

        # K-ordered Chebyshev polynomial
        adj_norm = self.normalize(adj)
        sqr_norm = self.normalize(torch.mm(adj, adj))
        m_norm = self.alpha * adj_norm + (1. - self.alpha) * sqr_norm

        x_tmp = torch.einsum('abcd,de->abce', x, self.weight)
        x_out = torch.einsum('ij,abid->abjd', m_norm, x_tmp)
        if self.bias is not None:
            x_out += self.bias

        x_out = self.act(x_out)

        return x_out




class StandConvolution2(nn.Module):
    def __init__(self, dims, num_classes, dropout):
        super(StandConvolution2, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, padding=1),
            nn.InstanceNorm2d(dims[1]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,2), stride=2),
            nn.Conv2d(dims[1], dims[2], kernel_size=3, padding=1),
            nn.InstanceNorm2d(dims[2]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((int(int(config['Max_Time'])/4),int(int(config['Max_Object'])/2)), stride=(int(int(config['Max_Time'])/4),int(int(config['Max_Object'])/2))),

        )

    def forward(self, x):
        x = self.dropout(x.permute(0, 3, 1, 2))
        x_tmp = self.conv(x)
        x_out = x_tmp.view(x.size(0), -1)

        return x_out

class StandConvolution2_RG(nn.Module):
    def __init__(self, dims, num_classes, dropout):
        super(StandConvolution2_RG, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, padding=1),
            nn.InstanceNorm2d(dims[1]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,2), stride=2),
            nn.Conv2d(dims[1], dims[2], kernel_size=3, padding=1),
            nn.InstanceNorm2d(dims[2]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((int(int(config['Max_Time'])/4), int(int(config['Max_Object'])*int(config['Max_Object'])/4)), stride=(int(int(config['Max_Time'])/4),int(int(config['Max_Object'])*int(config['Max_Object'])/4))),

        )

    def forward(self, x):
        x = self.dropout(x.permute(0, 3, 1, 2))
        x_tmp = self.conv(x)
        x_out = x_tmp.view(x.size(0), -1)

        return x_out

class StandConvolution2_HARG(nn.Module):
    def __init__(self, dims, num_classes, dropout):
        super(StandConvolution2_HARG, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, padding=1),
            nn.InstanceNorm2d(dims[1]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,2), stride=2),
            nn.Conv2d(dims[1], dims[2], kernel_size=3, padding=1),
            nn.InstanceNorm2d(dims[2]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1,int(num_classes/2)), stride=(1,int(num_classes/2))),
        )

    def forward(self, x):
        x = self.dropout(x.permute(0, 3, 1, 2))
        x_tmp = self.conv(x)
        x_out = x_tmp.view(x.size(0), -1)

        return x_out


