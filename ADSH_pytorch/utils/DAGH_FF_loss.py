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

    # def forward(self, F_batch, B_batch):
    #     batch_size = B_batch.size (1)
    #     nI_K = Variable(batch_size * torch.eye (self.code_length, self.code_length).cuda ())
    #     # one_vectors = Variable(torch.ones(self.code_length,batch_size).cuda ())
    #     reg_loss = (B_batch - F_batch) ** 2
    #     oth_loss = self.lambda_1 * ((F_batch.mm(F_batch.t()) - nI_K) ** 2)
    #     bla_loss = self.lambda_2 * ((F_batch.sum(1)) ** 2)
    #     # susb_loss = torch.abs(F_batch) - one_vectors
    #     # L1_loss = self.lambda_3 * torch.abs(susb_loss)
    #     # loss = 0.5*(oth_loss.sum()/batch_size + bla_loss.sum()) / batch_size
    #
    #     loss = 0.5 * ( reg_loss.sum()  + bla_loss.sum()) / batch_size + 0.5 * oth_loss.sum() / (self.code_length ** 2)
    #     # loss = 0.5 * (reg_loss.sum () + oth_loss.sum () + bla_loss.sum () + L1_loss.sum ()) / batch_size
    #     return loss

    def forward(self, F_batch, B_batch):
        batch_size = B_batch.size (1)
        nI_K = Variable (torch.eye (self.code_length, self.code_length).cuda ())
        reg_loss = (B_batch - F_batch) ** 2
        oth_loss = self.lambda_1 * ((F_batch.mm (F_batch.t ())/batch_size - nI_K) ** 2)
        bla_loss = self.lambda_2 * ((F_batch.sum (1)) ** 2)

        loss = 0.5 * (reg_loss.sum () + bla_loss.sum ()) / batch_size + 0.5 * oth_loss.sum () / (self.code_length)
        return loss
