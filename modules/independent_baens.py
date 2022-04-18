import torch
import torch.nn as nn

# only weight quantization, without decomposition

TEMPERATURE = 1.
THRES = 0.9

def get_round():
    class round(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            # return torch.round(input)
            return torch.floor(input)

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input
    return round().apply

def df_lt(a, b, temp):
    # if a > b
    return torch.sigmoid((a - b) / temp)

class dense_independent_baens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, quantize=True):
        super(dense_independent_baens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.D = D
        self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):
        # wQ = self.quant(self.U)
        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)

        w = self.U.view(self.N, self.D1, self.D2)

        act = torch.einsum('bnd, ndl -> bnl', x, w)

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act

class Conv2d_independent_baens(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, N=5, quantize=True, first=False):
        super(Conv2d_independent_baens, self).__init__()

        self.first = first
        self.N = N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        D = int(in_channels * out_channels * kernel_size * kernel_size)
        self.D = D
        
        self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):
        if not self.first:
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])

        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)

        w = self.U.view(int(self.out_channels * self.N), int(self.in_channels), self.kernel_size, self.kernel_size)

        # x should be of the size (sub-batch-size , (self.N * in_channels) , kernel_size, kernel_size)
        act = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.N)
        act = act.view(int(act.shape[0] * self.N), int(act.shape[1] / self.N), *act.shape[2:])

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act


# import torch
# from fast_soft_sort.pytorch_ops import soft_rank, soft_sort
# values = torch.tensor([[5.,2.,3.,1.,4.,5.]], requires_grad=True)
# optim = torch.optim.SGD([values], lr=0.1)
# for i in range(100):
#     optim.zero_grad()
#     out = soft_sort(values, regularization_strength=0.1)
#     l = (out - torch.tensor([1.,2.,0.,4.,5.,5.]))**2
#     l.mean().backward()
#     optim.step()
#     print('----------')
#     print(out)
#     print(values)