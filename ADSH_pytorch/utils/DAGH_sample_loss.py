import torch.nn as nn
import torch
from torch.autograd import Variable

class DAGH_sample_loss(nn.Module):
    '''
    except Tr(BLF)
    '''
    def __init__(self, lambda_1, lambda_2, lambda_3, code_length):
        super(DAGH_sample_loss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.code_length = code_length

    def forward(self, w_batch, F_batch, B_batch):
        batch_size = B_batch.size (1)
        tr_loss = 0
        for i in range(batch_size):
            for j in range(batch_size):
                tr_loss = tr_loss + w_batch[i,j] *  ((F_batch[:,i] - B_batch[:,j])**2).sum()

        nI_K = Variable (torch.eye (self.code_length, self.code_length).cuda ())
        oth_loss = self.lambda_1 * ((F_batch.mm (F_batch.t ())/batch_size - nI_K) ** 2)
        bla_loss = self.lambda_2 * ((F_batch.sum (1)) ** 2)

        loss = 0.5 * tr_loss / (batch_size*batch_size)*10000 + 0.5 * ( bla_loss.sum ()) / batch_size + 0.5 * oth_loss.sum () / (self.code_length)
        return loss
