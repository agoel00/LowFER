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
        # self.U = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d1, k * o)),
                                    # dtype=torch.float, device="cuda", requires_grad=True))
        # self.V = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, k * o)),
                                    # dtype=torch.float, device="cuda", requires_grad=True))
        self.input_dropout = nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = nn.Dropout(kwargs["hidden_dropout2"])
        self.bn0 = nn.BatchNorm1d(d1)
        self.bn1 = nn.BatchNorm1d(d1)
        self.k = k
        self.o = o
        self.loss = nn.BCELoss()

        self.p = d1//2
        self.s = 3
        self.t = 20

        projectionMatrix = np.random.choice(
            (-1,0,1), size=(d1,self.p),
            p=[1./(2*self.s), 1-1./self.s, 1./(2*self.s)]
        )
        self.register_buffer('projMatrix', torch.from_numpy(projectionMatrix))

        self.idx = [torch.randint(0, high=self.p, size=(self.k, self.t, 2))]

        self.factor = 1./torch.sqrt(k*t)
    
    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)

    def _transform(self, batch, h, r):
        y = torch.zeros(batch, self.k)

        for b in range(batch):
            for k in range(self.k):
                for i in range(self.t):
                    r1 = self.projMatrix[self.idx[k, i, 0]]
                    r2 = self.projMatrix[self.idx[k, i, 1]]
                    y[b, k] += torch.dot(h, r1) * torch.dot(r, r2)
                y[b, k] *= self.factor

        return y 


    
    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)
        r = self.R(r_idx)
        batch = e1_idx.shape[0]

        x = self._transform(batch, e1, r)
        
        ## MFB
        # x = torch.mm(e1, self.U) * torch.mm(r, self.V)
        # x = self.hidden_dropout1(x)
        # x = x.view(-1, self.o, self.k)
        # x = x.sum(-1)
        x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
        x = nn.functional.normalize(x, p=2, dim=-1)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        
        pred = torch.sigmoid(x)
        return pred
