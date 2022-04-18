import torch
import torch.nn as nn

from torch.autograd import Variable

# class STSign(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input):
#         return torch.sign(input)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.clamp_(-1, 1)

class SigmoidT(torch.autograd.Function):
    """ sigmoid with temperature T for training
        we need the gradients for input and bias
        for customization of function, refer to https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(self, input, T):        
        self.save_for_backward(input)
        self.T = T

        buf = torch.clamp(self.T * input, min=-10.0, max=10.0)
        output = 1. / (1.0 + torch.exp(-buf))
        return output

    @staticmethod
    def backward(self, grad_output):
        # set T = 1 when train binary model in the backward.
        #self.T = 1
        input, = self.saved_tensors
        b_buf = torch.clamp(self.T * input, min=-10.0, max=10.0)
        b_output = 1. / (1.0 + torch.exp(-b_buf))
        temp = b_output * (1 - b_output) * self.T
        grad_input = Variable(temp) * grad_output      
        # corresponding to grad_input
        return grad_input, None

sigmoidT = SigmoidT.apply

TEMPERATURE = 1.

class dense_decompo_nbcp_baens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, bw=2, K=1, quantize=True):
        super(dense_decompo_nbcp_baens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.bw = bw

        if quantize:
            self.U = nn.Parameter(torch.normal(0, 1, (N, D, K)), requires_grad=True)
            self.V = nn.Parameter(torch.normal(0, 1, (bw, D, K)), requires_grad=True)
            self.twopow = nn.Parameter(torch.tensor([2.**i for i in range(self.bw)]), requires_grad=False)
            self.TWO = nn.Parameter(torch.tensor([2.]), requires_grad=False)

            self.scale = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
            self.biasq = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
            torch.nn.init.kaiming_normal_(self.V)

        else:
            self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        self.bias = nn.Parameter(torch.normal(0, 1, (N, D2)), requires_grad=True)

        self.temp = TEMPERATURE
        self.quantize = quantize

        torch.nn.init.kaiming_normal_(self.U)
        torch.nn.init.kaiming_normal_(self.bias)

    def forward(self, x):
        if self.quantize:

            Theta = torch.einsum('ndk, bdk -> nbd', self.U, self.V)

            soft_binary = sigmoidT(Theta, self.temp)
            # soft_binary = sign(Theta)
            # soft_binary = (sign(Theta) + 1.) / 2.

            integer = torch.einsum('uwv, w->uv', soft_binary, self.twopow)

            w = self.scale * (integer - self.biasq - torch.pow(self.TWO, self.bw))

            # w = self.scale * (integer - self.bias - torch.pow(self.TWO, self.bw))

            w = w.view(self.N, self.D1, self.D2)
            # print(w.shape)
            # print(x.shape)
            act = torch.einsum('bnd, ndl -> bnl', x, w)

        if not self.quantize:

            w = self.U.view(self.N, self.D1, self.D2)

            act = torch.einsum('bnd, ndl -> bnl', x, w) 

        act += self.bias

        if torch.sum(torch.isnan(act)) != 0:

            print('act nan')
            print(act)
            exit()

        return act


class Conv2d_decompo_nbcp_baens(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, N=5, bw=2, K=1, quantize=True, first=False):
        super(Conv2d_decompo_nbcp_baens, self).__init__()

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
        self.bw = bw

        if quantize:
            self.U = nn.Parameter(torch.normal(0, 1, (N, D, K)), requires_grad=True)
            self.V = nn.Parameter(torch.normal(0, 1, (bw, D, K)), requires_grad=True)
            self.twopow = nn.Parameter(torch.tensor([2.**i for i in range(self.bw)]), requires_grad=False)
            self.TWO = nn.Parameter(torch.tensor([2.]), requires_grad=False)

            self.scale = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
            self.biasq = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
            torch.nn.init.kaiming_normal_(self.V)

        else:
            self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        self.bias = nn.Parameter(torch.normal(0, 1, (1, int(N * out_channels), 1, 1), requires_grad=True))

        self.temp = TEMPERATURE
        self.quantize = quantize

        torch.nn.init.kaiming_normal_(self.U)
        torch.nn.init.kaiming_normal_(self.bias)

    def forward(self, x):
        if not self.first:
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])

        if self.quantize:

            # Theta = torch.einsum('ur, vr, wr, r->uvw', self.U, self.V, self.W, self.R)
            Theta = torch.einsum('ndk, bdk -> nbd', self.U, self.V)
            
            soft_binary = sigmoidT(Theta, self.temp)

            # sign = STSign.apply
            # # soft_binary = torch.sigmoid(Theta/self.temp)
            # soft_binary = (sign(Theta) + 1.) / 2.

            integer = torch.einsum('uwv, w->uv', soft_binary, self.twopow)

            w = self.scale * (integer - self.biasq - torch.pow(self.TWO, self.bw))

            # w = self.scale * (integer - self.bias - torch.pow(self.TWO, self.bw))

            w = w.view(int(self.out_channels * self.N), int(self.in_channels), self.kernel_size, self.kernel_size)

            # x should be of the size (sub-batch-size , (self.N * in_channels) , kernel_size, kernel_size)
            act = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.N)

        if not self.quantize:

            w = self.U.view(int(self.out_channels * self.N), int(self.in_channels), self.kernel_size, self.kernel_size)

            # x should be of the size (sub-batch-size , (self.N * in_channels) , kernel_size, kernel_size)
            act = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.N)

        act += self.bias

        act = act.view(int(act.shape[0] * self.N), int(act.shape[1] / self.N), *act.shape[2:])

        if torch.sum(torch.isnan(act)) != 0:

            print('act nan')
            print(act)
            exit()

        return act