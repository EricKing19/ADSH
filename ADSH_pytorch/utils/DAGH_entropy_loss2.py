import torch.nn as nn
import torch
from torch.autograd import Variable


class DAGHLoss(nn.Module):
    '''
    except Tr(BLF)
    '''

    def __init__(self, lambda_1, lambda_2, lambda_3, code_length):
        super(DAGHLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.code_length = code_length

    def forward(self, F_batch, B_batch):
        batch_size = B_batch.size(1)
        # mean_F = F_batch.mean(1).view(self.code_length,1)
        ee = 1e-7
        reg_loss = (B_batch - F_batch) ** 2
        FF = F_batch.mm(F_batch.t())
        F_batch01 = (F_batch + 1) / 2
        L1_loss = - (F_batch01 * torch.log(F_batch01 + ee) + (1 - F_batch01) * torch.log(1 - F_batch01 + ee))

        mean_F = F_batch01.mean(1).view(self.code_length, 1)

        #bla_loss = (mean_F * torch.log(mean_F + ee) + (1 - mean_F) * torch.log(1 - mean_F + ee))
        bla_loss = self.lambda_2 * ((F_batch.sum(1)) ** 2)

        oth_loss = self.lambda_1 * ((FF ** 2).sum() - 2 * batch_size * torch.trace(FF))

        loss = 0.5 * (reg_loss.sum() + bla_loss.sum()) / (batch_size * self.code_length) + self.lambda_3 * L1_loss.sum() / (batch_size * self.code_length) +  0.25 * oth_loss / (batch_size * self.code_length)
        return loss

