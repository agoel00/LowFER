# -*- codingL utf-8 -*-

import numpy as np
import torch.nn as nn
import torch
import geotorch

class LowFER(nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(LowFER, self).__init__()
        
        self.E = nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = nn.Embedding(len(d.relations), d2, padding_idx=0)
        k, o = kwargs.get('k', 30), d1
        # self.U = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d1, k * o)),
        #                             dtype=torch.float, device="cuda", requires_grad=True))
        # self.V = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, k * o)),
        #                             dtype=torch.float, device="cuda", requires_grad=True))
        self.U = nn.Linear(d1, k*o, bias=False)
        self.V = nn.Linear(d2, k*o, bias=False)
        self.input_dropout = nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = nn.Dropout(kwargs["hidden_dropout2"])
        self.bn0 = nn.BatchNorm1d(d1)
        self.bn1 = nn.BatchNorm1d(d1)
        self.k = k
        self.o = o
        self.loss = nn.BCELoss()
    
    def init(self):
        print('Init model params...')
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)
        nn.init.uniform_(self.U.weight, -1, 1)
        nn.init.uniform_(self.V.weight, -1, 1)
        geotorch.orthogonal(self.U, 'weight')
        geotorch.orthogonal(self.V, 'weight')
        self.U = self.U.cuda()
        self.V = self.V.cuda()
    
    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)
        r = self.R(r_idx)
        
        ## MFB
        x = torch.mm(e1, self.U.weight.t()) * torch.mm(r, self.V.weight.t())
        x = self.hidden_dropout1(x)
        x = x.view(-1, self.o, self.k)
        x = x.sum(-1)
        x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
        x = nn.functional.normalize(x, p=2, dim=-1)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        
        pred = torch.sigmoid(x)
        return pred
