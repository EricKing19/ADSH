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
        batch_size = B_batch.size (1)
        reg_loss = (B_batch - F_batch) ** 2
        FF = F_batch.mm (F_batch.t ())
        # FF_oth = (FF - torch.diag (torch.diag (FF)))
        oth_loss = self.lambda_1 * ( (FF ** 2).sum () - 2*batch_size*torch.trace(FF))
        bla_loss = self.lambda_2 * ((F_batch.sum (1)) ** 2)

        loss = 0.5 * (reg_loss.sum () + bla_loss.sum ()) / (batch_size * self.code_length) + 0.5 * oth_loss / (batch_size * self.code_length)
        return loss

