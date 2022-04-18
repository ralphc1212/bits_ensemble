import torch
import torch.nn as nn

global_nbits = 8

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

class dense_full_ens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2):
        super(dense_full_ens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        # D = int(D1 * D2)

        # # baens
        # self.U = nn.Parameter(torch.normal(0, 1, (1, D1, D2)), requires_grad=True)
        # self.S = nn.Parameter(torch.normal(0, 1, (N, D1, 1)), requires_grad=True)
        # self.R = nn.Parameter(torch.normal(0, 1, (N, 1, D2)), requires_grad=True)

        # torch.nn.init.kaiming_normal_(self.U)
        # torch.nn.init.kaiming_normal_(self.S)
        # torch.nn.init.kaiming_normal_(self.R)
        self.round = get_round()

        # bitensemble pretrain
        self.U = nn.Parameter(torch.normal(0, 1, (N, D1, D2)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):

        s = (torch.max(self.U) - torch.min(self.U)) / (2**global_nbits - 1)
        U = (s * self.round(self.U / s))

        # # baens
        # w = self.S * self.U * self.R

        # act = torch.einsum('bnd, ndl -> bnl', x, w)

        # bitensemble pretrain
        act = torch.einsum('bnd, ndl -> bnl', x, U)

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act

class Conv2d_full_ens(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, N=5, first=False):
        super(Conv2d_full_ens, self).__init__()

        self.first = first
        self.N = N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # D = int(in_channels * out_channels * kernel_size * kernel_size)

        # # baens
        # self.U = nn.Parameter(torch.normal(0, 1, (1, out_channels, in_channels, kernel_size, kernel_size)), requires_grad=True)
        # self.S = nn.Parameter(torch.normal(0, 1, (N, 1, in_channels,  1, 1)), requires_grad=True)
        # self.R = nn.Parameter(torch.normal(0, 1, (N, out_channels, 1, 1, 1)), requires_grad=True)

        # torch.nn.init.kaiming_normal_(self.U)
        # torch.nn.init.kaiming_normal_(self.S)
        # torch.nn.init.kaiming_normal_(self.R)
        self.round = get_round()

        # bitensemble pretrain
        self.U = nn.Parameter(torch.normal(0, 1, (int(N*out_channels), in_channels, kernel_size, kernel_size)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):
        # if not self.first:
            # x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        s = (torch.max(self.U) - torch.min(self.U)) / (2**global_nbits - 1)
        U = (s * self.round(self.U / s))

        # # baens
        # w = self.R * self.U * self.S
        # act = torch.nn.functional.conv2d(x, w.view(w.shape[0] * w.shape[1], *w.shape[2:]),
        #  stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.N)

        act = torch.nn.functional.conv2d(x, U, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.N)

        # act = torch.einsum('bnd, ndl -> bnl', x, w)

        # bitensemble pretrain
        # act = torch.nn.functional.conv2d(x, self.U.view(self.U.shape[0] * self.U.shape[1], *self.U.shape[2:]),
        #  stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.N)

        # act = act.view(int(act.shape[0] * self.N), int(act.shape[1] / self.N), *act.shape[2:])

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act
