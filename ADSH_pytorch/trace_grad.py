import torch
from torch.autograd import Variable

x0 = Variable(torch.ones(2, 2), requires_grad=True)
x = 4 * x0
print(x)
y = Variable(torch.ones(2, 2))+2
print(y)
print(x.grad_fn)

z = x.mm(y)
print(z)
out = torch.trace(z)
print(out)
out.norm(1)
gradients = torch.FloatTensor([[3,3],[3,3]])
out.backward(retain_graph =True)
print(x0.grad)
x.backward(gradients)
#out.backward()
print(x0.grad)