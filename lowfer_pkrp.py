# -*- codingL utf-8 -*-

import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm


class LowFER(nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(LowFER, self).__init__()
        
        # self.E = nn.Embedding(len(d.entities), d1, padding_idx=0)
        # self.R = nn.Embedding(len(d.relations), d2, padding_idx=0)
        if kwargs["pretrained_ents"]:
            with open(kwargs["pretrained_ents"], "rb") as handle:
                ents = pickle.load(handle)
            ent_embs = np.array(list(ents.values()))
            assert ent_embs.shape[1] == d1
            self.E = nn.Embedding.from_pretrained(torch.from_numpy(ent_embs).float())
        else:
            self.E = nn.Embedding(len(d.entities), d1, padding_idx=0)
        # print(self.E.weight.dtype)

        if kwargs["pretrained_rels"]:
            with open(kwargs["pretrained_rels"], "rb") as handle:
                rels = pickle.load(handle)
            rel_embs = np.array(list(rels.values()))
            assert rel_embs.shape[1] == d2
            self.R = nn.Embedding.from_pretrained(torch.from_numpy(rel_embs).float())
        else:
            self.R = nn.Embedding(len(d.relations), d2, padding_idx=0)
        # print(self.R.weight.dtype)
        # self.init()
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

        self.p = d1
        self.s = 3
        self.t = 20

        # projectionMatrix = np.random.choice(
        #     (-1,0,1), size=(d1,self.p),
        #     p=[1./(2*self.s), 1-1./self.s, 1./(2*self.s)]
        # )
        projectionMatrix = np.random.choice(
            (-1,0,1), size=(self.k, self.t, 2, d1),
            p=[1./(2*self.s), 1-1./self.s, 1./(2*self.s)]
        )
        proj = torch.from_numpy(projectionMatrix).float().to('cuda:0')
        self.register_buffer('projMatrix', proj)

        # self.idx = torch.randint(0, high=self.p, size=(self.k, self.t, 2))
        # self.idx = np.asarray(self.idx)

        # self.factor = 1./torch.sqrt(torch.Tensor(self.k*self.t))
        self.factor = 1./np.sqrt(float(self.k*self.t))
        self.factor = torch.FloatTensor([self.factor]).to('cuda:0')
    
    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)

    def _transform(self, batch, h, r):
        # y = torch.zeros(batch, self.k).to('cuda:0')

        r1 = self.projMatrix[:, :, 0, :]
        r2 = self.projMatrix[:, :, 1, :]
        tmp1 = torch.matmul(r1, torch.transpose(h, 0, 1))
        tmp2 = torch.matmul(r2, torch.transpose(r, 0, 1))
        tmp3 = torch.sum(tmp1 * tmp2, dim=1)
        y = torch.transpose(tmp3, 0, 1) * self.factor


        # r1 = self.projMatrix[:, :, 0, :]
        # r2 = self.projMatrix[:, :, 1, :]
        # for b in range(batch):
        #     tmp1 = h[b].view(-1) * r1
        #     tmp1 = torch.sum(tmp1, dim=2)
        #     tmp2 = r[b].view(-1) * r2
        #     tmp2 = torch.sum(tmp2, dim=2)
        #     tmp3 = torch.sum(tmp1 * tmp2, dim=1)
        #     y[b, :] += tmp3
        # y = y * self.factor

        # for b in tqdm(range(batch)):
        #     for k in range(self.k):
        #         r1 = self.projMatrix[self.idx[k, :, 0]]
        #         r2 = self.projMatrix[self.idx[k, :, 1]]
        #         tmp1 = h[b].view(-1) * r1
        #         tmp1 = torch.sum(tmp1, dim=1)
        #         tmp2 = r[b].view(-1) * r2
        #         tmp2 = torch.sum(tmp2, dim=1)
        #         y_clone[b, k] += torch.dot(tmp1, tmp2)
        #         y_clone[b, k] = y_clone[b, k] * self.factor
        #         # print(r1.shape)
        #         # print("LALAL")
        #         # print(r2.shape)

        # for b in tqdm(range(batch)):
        #     for k in range(self.k):
        #         for i in range(self.t):
        #             r1 = self.projMatrix[self.idx[k][i][0]]
        #             r2 = self.projMatrix[self.idx[k][i][1]]
        #             y[b, k] += torch.dot(h[b].view(-1), r1.view(-1)) * torch.dot(r[b].view(-1), r2.view(-1))
        #         y[b, k] = y[b,k] * self.factor
        #         # print(y[b,k])
        #         # print(self.factor)
        #         # break
        # print(y)
        # print(y_clone)

        # for i in range(batch):
        #     for j in range(self.k):
        #         if y[i][j].item() != y_clone[i][j].item():
        #             print(i, j, y[i][j].item(), y_clone[i][j].item())
        # print("YEEE")


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
        # x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
        # x = nn.functional.normalize(x, p=2, dim=-1)
        # print(x.shape)
        # print("LALALAL")
        # print(self.E.weight.shape)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        
        pred = torch.sigmoid(x)
        return pred



