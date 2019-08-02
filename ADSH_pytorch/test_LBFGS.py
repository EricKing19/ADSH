import torch
import torch.optim as optim
from torch.autograd import Variable


X = Variable(torch.ones(1), requires_grad=True)

optimizer_B = optim.LBFGS([X], lr=1)

def closure():
    optimizer_B.zero_grad()

    loss = X**2 + X*4 + 4
    loss.backward()
    return loss
optimizer_B.step(closure)

print X
print X**2 + X*4 + 4
