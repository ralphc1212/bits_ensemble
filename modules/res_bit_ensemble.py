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

class dense_res_bit_baens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, bw=3, quantize=True):
        super(dense_res_bit_baens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        self.scale = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
        self.bias = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
        self.res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

        self.round = get_round()
        self.maxBWPow = bw

        self.ONE = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.ZERO = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.zp1 = nn.Parameter(torch.tensor([1.2]), requires_grad=False)
        self.zp2 = nn.Parameter(torch.tensor([-0.2]), requires_grad=False)

        self.binary_parameters = nn.Parameter(torch.randn((bw, N, D)), requires_grad=True)
        self.ONES = nn.Parameter(torch.ones_like(self.binary_parameters), requires_grad=False)
        self.ZEROS = nn.Parameter(torch.zeros_like(self.binary_parameters), requires_grad=False)

        self.temperature = TEMPERATURE
        torch.nn.init.kaiming_normal_(self.U)
        torch.nn.init.kaiming_normal_(self.binary_parameters)

    def forward(self, x):
        # wQ = self.quant(self.U)
        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)

        beta = torch.max(self.U)
        alpha = torch.min(self.U)

        # if 1:
        #     u = torch.distributions.uniform.Uniform(self.ZEROS, self.ONES).sample()
        #     g = torch.log(u / (1 - u))
        #     bsamples = torch.sigmoid((g + self.binary_parameters) / self.temperature)
        #     gates = torch.min(self.ONE, torch.max(self.ZERO, bsamples * (self.zp1 - self.zp2) + self.zp2))
        # else:
        #     gates = 1 - torch.sigmoid(self.temperature * torch.log(- self.zp2 / self.zp1) - self.binary_parameters)
        #     gates = gates < THRES

        # keep all
        gates = self.ONES

        order_gates = torch.cumprod(gates > 0, dim=0).detach()
        # order_gates[1, 1:, :] = 0.

        s = None
        vals = None
        for idx, res_deno in enumerate(self.res_denos[:self.maxBWPow]):
            if gates.sum(dim=(1,2 ))[idx] > 0:
                if s is None:
                    # s = (beta - alpha).unsqueeze(1) / res_deno
                    s = (beta - alpha)/ res_deno
                    vals = (s * self.round(self.U / s)).unsqueeze(0)
                else:
                    s = s / res_deno
                    vals = torch.cat([vals, (s * self.round((self.U - vals.sum(0)) / s)).unsqueeze(0)], dim=0)
                    # print(self.U - vals.sum(0))

        w = vals * (gates * order_gates)
        w = w.sum(0).view(self.N, self.D1, self.D2)

        act = torch.einsum('bnd, ndl -> bnl', x, w)

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act



class Conv2d_res_bit_baens(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, N=5, bw=2, quantize=True, first=False):
        super(Conv2d_res_bit_baens, self).__init__()

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

        self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)
        self.scale = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
        self.biasq = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)

        self.res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

        # self.quant = res_quant(maxBWPow=3)
        self.round = get_round()
        self.maxBWPow = bw

        self.ONE = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.ZERO = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.zp1 = nn.Parameter(torch.tensor([1.2]), requires_grad=False)
        self.zp2 = nn.Parameter(torch.tensor([-0.2]), requires_grad=False)

        self.binary_parameters = nn.Parameter(torch.randn((bw, N, D)), requires_grad=True)
        self.ONES = nn.Parameter(torch.ones_like(self.binary_parameters), requires_grad=False)
        self.ZEROS = nn.Parameter(torch.zeros_like(self.binary_parameters), requires_grad=False)

        self.temperature = TEMPERATURE
        torch.nn.init.kaiming_normal_(self.U)
        torch.nn.init.kaiming_normal_(self.binary_parameters)

    def forward(self, x):
        if not self.first:
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])

        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)

        beta = torch.max(self.U)
        alpha = torch.min(self.U)

        # if 1:
        #     u = torch.distributions.uniform.Uniform(self.ZEROS, self.ONES).sample()
        #     g = torch.log(u / (1 - u))
        #     bsamples = torch.sigmoid((g + self.binary_parameters) / self.temperature)
        #     gates = torch.min(self.ONE, torch.max(self.ZERO, bsamples * (self.zp1 - self.zp2) + self.zp2))
        # else:
        #     gates =  1 - torch.sigmoid(self.temperature * torch.log(- self.zp2 / self.zp1) - self.binary_parameters) 
        #     gates = gates < THRES

        # keep all
        gates = self.ONES

        order_gates = torch.cumprod(gates > 0, dim=0).detach()
        # order_gates[1, 1:, :] = 0.

        s = None
        vals = None
        for idx, res_deno in enumerate(self.res_denos[:self.maxBWPow]):
            if gates.sum(dim=(1, 2))[idx] > 0:
                if s is None:
                    # s = (betas - alphas).unsqueeze(1) / res_deno
                    s = (beta - alpha)/ res_deno
                    vals = (s * self.round(self.U / s)).unsqueeze(0)
                else:
                    s = s / res_deno
                    vals = torch.cat([vals, (s * self.round((self.U - vals.sum(0)) / s)).unsqueeze(0)], dim=0)

        w = vals * (gates * order_gates)
        w = w.sum(0).view(int(self.out_channels * self.N), int(self.in_channels), self.kernel_size, self.kernel_size)

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