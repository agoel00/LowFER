# -*- codingL utf-8 -*-

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F 


class LowFER(nn.Module):
    def __init__(self, d, d1, d2, input_dropout, hidden_dropout1, hidden_dropout2, k, subspace):
        super(LowFER, self).__init__()
        
        self.E = nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = nn.Embedding(len(d.relations), d2, padding_idx=0)
        o = d1
        self.subspace = subspace
        self.U1 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d1, k * o)),
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.V1 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, k * o)),
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout2)
        self.bn0 = nn.BatchNorm1d(d1)
        self.bn1 = nn.BatchNorm1d(d1)
        self.k = k
        self.o = o
        self.loss = nn.BCELoss()

        # self.fc1 = nn.Linear(d1, self.subspace)
        # self.fc2 = nn.Linear(d2, self.subspace)
        self.fc3 = nn.Linear(d1, d2)
        self.fc4 = nn.Linear(d2, d1)
    
    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)
    
    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)
        r = self.R(r_idx)
        
        # e1_bar = F.relu(self.fc1(e1))
        # r_bar = F.relu(self.fc2(r))
        alpha_e1 = torch.sigmoid(self.fc4(r))
        alpha_r = torch.sigmoid(self.fc3(e1))

        # e1_aug = e1_bar * alpha_e1
        # r_aug = r_bar * alpha_r
        ## MFB
        x = (torch.mm(e1, self.U1) * alpha_e1) * (torch.mm(r, self.V1) * alpha_r)
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