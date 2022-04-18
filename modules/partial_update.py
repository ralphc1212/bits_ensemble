import torch
import torch.nn as nn

def get_comp_linear():
    class comp_linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, w1, w2):
            ctx.save_for_backward(input, w1, w2)
            out = torch.mm(input, torch.cat([w1, w2],dim=1))
            return out
        @staticmethod

        def backward(ctx, grad_output):
            input, w1, w2 = ctx.saved_tensors
            grad_input = torch.mm(grad_output, torch.cat([w1, w2],dim=1).t())
            grad_w1 = torch.mm(grad_output[:, :w1.shape[1]], input.expand(w1.shape[1], -1))
            print('gw1', grad_w1.shape)
            print('gi1', grad_input.shape)
            return grad_input, grad_w1, None
    return comp_linear().apply


class MyLinear(torch.nn.Module):
    def __init__(self, in_dim = 3, out_dim = 4):
        super(MyLinear, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(in_dim, int(out_dim/2)), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.randn(in_dim, int(out_dim/2)), requires_grad=False)
        self.comp_l = get_comp_linear()

    def forward(self, input):
        return self.comp_l(input, self.w1, self.w2)

# x = nn.Parameter(torch.rand((1,3)), requires_grad=True)
# l = nn.Linear(3, 2, bias=False)
# print(l.weight)
# o = l(x)
# loss = (o).sum()
# loss.backward()
# print(x.grad)
# print(o.grad)

l = MyLinear()
x = torch.rand((1,3))
y = torch.tensor([1,2,3,4])

optim = torch.optim.SGD(l.parameters(), lr=0.01)

for i in range(100):
    optim.zero_grad()
    out = l(x)
    print(out)
    loss = (out - y)**2
    loss = loss.mean()
    loss.backward()
    optim.step()