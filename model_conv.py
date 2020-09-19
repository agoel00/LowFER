# -*- codingL utf-8 -*-

import numpy as np
import torch.nn as nn
import torch

class LowFER(nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(LowFER, self).__init__()
        
        self.E = nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = nn.Embedding(len(d.relations), d2, padding_idx=0)
        k, o = kwargs.get('k', 30), d1
        self.U = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d1, k * o)),
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.V = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, k * o)),
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.conv = nn.Conv2d(1, 200, (2,2), 1, 0, bias=True)
        self.pool = nn.AvgPool1d(100, 10)
        self.fc = nn.Linear(1971, k*o)
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
        # nn.init.uniform_(self.U, -1, 1)
        # nn.init.uniform_(self.V, -1, 1)
        self.U = self.U.cuda()
        self.V = self.V.cuda()
    
    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)
        r = self.R(r_idx)

        c = torch.cat([e1.view(-1, 1, 1, self.o), r.view(-1, 1, 1, self.o)], 2)
        c = self.conv(c)
        c = c.view(c.shape[0], 1, -1)
        c = self.pool(c)
        c = c.view(c.shape[0], -1)
        c = self.fc(c)

        
        ## MFB
        x = torch.mm(e1, self.U) * torch.mm(r, self.V) * c
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
