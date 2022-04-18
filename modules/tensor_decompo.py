import torch
import torch.nn as nn


class dense_decompo_tucker(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, bw=2, KN=3, KD=10, Kbw=1):
        super(dense_decompo_tucker, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.bw = bw
        self.U = nn.Parameter(torch.normal(0, 1, (N, KN)), requires_grad=True)
        self.V = nn.Parameter(torch.normal(0, 1, (D, KD)), requires_grad=True)
        self.W = nn.Parameter(torch.normal(0, 1, (bw, Kbw)), requires_grad=True)
        self.R = nn.Parameter(torch.normal(0, 1, (KN, KD, Kbw)), requires_grad=True)

        self.twopow = nn.Parameter(torch.tensor([2.**i for i in range(self.bw)]), requires_grad=False)
        self.TWO = nn.Parameter(torch.tensor([2.]), requires_grad=False)

        self.scale = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
        self.bias = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
        self.temp = 0.05

        torch.nn.init.kaiming_normal_(self.U)
        torch.nn.init.kaiming_normal_(self.V)
        torch.nn.init.kaiming_normal_(self.W)
        torch.nn.init.kaiming_normal_(self.R)
        # torch.nn.init.kaiming_normal(self.scale)
        # torch.nn.init.kaiming_normal(self.bias)

    def forward(self, x):

        Theta = torch.einsum('ua, vb, wc, abc->uvw', self.U, self.V, self.W, self.R)

        soft_binary = torch.sigmoid(Theta/self.temp)

        integer = torch.einsum('uvw, w->uv', soft_binary, self.twopow)

        w = self.scale * (integer - self.bias - torch.pow(self.TWO, self.bw))

        w = w.view(self.N, self.D1, self.D2)

        act = torch.einsum('ij, bjk->bik', x, w).mean(dim=0)

        if torch.sum(torch.isnan(act)) != 0:

            print('act nan')
            print(act)
            exit()

        return act


class dense_decompo_cp(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, bw=2, K=1):
        super(dense_decompo_cp, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.bw = bw
        self.U = nn.Parameter(torch.normal(0, 1, (N, K)), requires_grad=True)
        self.V = nn.Parameter(torch.normal(0, 1, (D, K)), requires_grad=True)
        self.W = nn.Parameter(torch.normal(0, 1, (bw, K)), requires_grad=True)
        self.R = nn.Parameter(torch.normal(0, 1, (K,)), requires_grad=True)

        self.twopow = nn.Parameter(torch.tensor([2.**i for i in range(self.bw)]), requires_grad=False)
        self.TWO = nn.Parameter(torch.tensor([2.]), requires_grad=False)

        self.scale = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
        self.bias = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
        self.temp = 0.05

        torch.nn.init.kaiming_normal_(self.U)
        torch.nn.init.kaiming_normal_(self.V)
        torch.nn.init.kaiming_normal_(self.W)
        # torch.nn.init.kaiming_normal_(self.R)
        # torch.nn.init.kaiming_normal(self.scale)
        # torch.nn.init.kaiming_normal(self.bias)

    def forward(self, x):

        # Theta = torch.einsum('ur, vr, wr, r->uvw', self.U, self.V, self.W, self.R)
        # soft_binary = torch.sigmoid(Theta / self.temp)
        # integer = torch.einsum('uvw, w->uv', soft_binary, self.twopow)
        #
        # w = self.scale * (integer - self.bias - torch.pow(self.TWO, self.bw))
        # w = w.view(self.N, self.D1, self.D2)
        #
        # votes = torch.einsum('ij, bjk->bik', x, w)
        #
        # f = votes.detach().mean(dim=0)
        # dif = votes - f
        #
        # sum_comb = dif.sum(dim=0) - dif
        #
        # reg_elements = dif * sum_comb
        #
        # reg = reg_elements.sum(dim=0)

        Theta = torch.einsum('ur, vr, wr, r->uvw', self.U, self.V, self.W, self.R)
        soft_binary = torch.sigmoid(Theta/self.temp)
        integer = torch.einsum('uvw, w->uv', soft_binary, self.twopow)

        w = self.scale * (integer - self.bias - torch.pow(self.TWO, self.bw))
        w = w.view(self.N, self.D1, self.D2)

        act = torch.einsum('ij, bjk->bik', x, w).mean(dim=0)

        # votes = torch.einsum('ij, bjk->bik', x.detach(), w)
        #
        # f = votes.detach().mean(dim=0)
        # dif = votes - f
        # sum_comb = dif.sum(dim=0) - dif
        # reg_elements = dif * sum_comb
        #
        # reg = reg_elements.sum(dim=0)

        if torch.sum(torch.isnan(act)) != 0:

            print('act nan')
            print(act)
            exit()

        return act

class dense_decompo_cp_baens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, bw=2, K=1, quantize=True):
        super(dense_decompo_cp_baens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.bw = bw
        self.U = nn.Parameter(torch.normal(0, 1, (N, K)), requires_grad=True)
        self.V = nn.Parameter(torch.normal(0, 1, (D, K)), requires_grad=True)
        self.R = nn.Parameter(torch.normal(0, 1, (K,)), requires_grad=True)

        if quantize:
            self.W = nn.Parameter(torch.normal(0, 1, (bw, K)), requires_grad=True)
            self.twopow = nn.Parameter(torch.tensor([2.**i for i in range(self.bw)]), requires_grad=False)
            self.TWO = nn.Parameter(torch.tensor([2.]), requires_grad=False)

            self.scale = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
            self.bias = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
            torch.nn.init.kaiming_normal_(self.W)

        else:
            self.bias = nn.Parameter(torch.normal(0, 1, (D2,)), requires_grad=True)

        self.temp = 0.05
        self.quantize = quantize

        torch.nn.init.kaiming_normal_(self.U)
        torch.nn.init.kaiming_normal_(self.V)

    def forward(self, x):
        if self.quantize:

            Theta = torch.einsum('ur, vr, wr, r->uvw', self.U, self.V, self.W, self.R)

            soft_binary = torch.sigmoid(Theta/self.temp)
            # soft_binary = torch.sign(Theta)

            integer = torch.einsum('uvw, w->uv', soft_binary, self.twopow)

            w = self.scale * (integer - self.bias - torch.pow(self.TWO, self.bw))

            # w = self.scale * (integer - self.bias - torch.pow(self.TWO, self.bw))

            w = w.view(self.N, self.D1, self.D2)
            # print(w.shape)
            # print(x.shape)
            act = torch.einsum('bnd, ndl -> bnl', x, w)

        if not self.quantize:

            Theta = torch.einsum('ur, vr, r->uv', self.U, self.V, self.R)
            # print(Theta.view(self.N, self.D1, self.D2).shape)
            w = Theta.view(self.N, self.D1, self.D2)

            act = torch.einsum('bnd, ndl -> bnl', x, w) - self.bias

        if torch.sum(torch.isnan(act)) != 0:

            print('act nan')
            print(act)
            exit()

        return act

# torch.nn.Conv2d.weight


class Conv2d_decompo_cp_baens(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, N=5, bw=2, K=1, quantize=True):
        super(Conv2d_decompo_cp_baens, self).__init__()

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
        self.U = nn.Parameter(torch.normal(0, 1, (N, K)), requires_grad=True)
        self.V = nn.Parameter(torch.normal(0, 1, (D, K)), requires_grad=True)
        self.R = nn.Parameter(torch.normal(0, 1, (K,)), requires_grad=True)

        if quantize:
            self.W = nn.Parameter(torch.normal(0, 1, (bw, K)), requires_grad=True)
            self.twopow = nn.Parameter(torch.tensor([2.**i for i in range(self.bw)]), requires_grad=False)
            self.TWO = nn.Parameter(torch.tensor([2.]), requires_grad=False)

            self.scale = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
            self.biasq = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
            torch.nn.init.kaiming_normal_(self.W)

        else:
            if bias:
                self.bias = nn.Parameter(torch.normal(0, 1, (out_channels,)), requires_grad=True)

        self.temp = 0.05
        self.quantize = quantize

        torch.nn.init.kaiming_normal_(self.U)
        torch.nn.init.kaiming_normal_(self.V)

    def forward(self, x):
        if self.quantize:

            Theta = torch.einsum('ur, vr, wr, r->uvw', self.U, self.V, self.W, self.R)

            soft_binary = torch.sigmoid(Theta/self.temp)
            # soft_binary = torch.sign(Theta)

            integer = torch.einsum('uvw, w->uv', soft_binary, self.twopow)

            w = self.scale * (integer - self.biasq - torch.pow(self.TWO, self.bw))

            # w = self.scale * (integer - self.bias - torch.pow(self.TWO, self.bw))

            w = w.view(int(self.out_channels * self.N), int(self.in_channels), self.kernel_size, self.kernel_size)

            # x should be of the size (sub-batch-size , (self.N * in_channels) , kernel_size, kernel_size)
            act = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.N)

        if not self.quantize:

            Theta = torch.einsum('ur, vr, r->uv', self.U, self.V, self.R)
            w = Theta.view(self.N, self.D1, self.D2)

            w = w.view(int(self.out_channels * self.N), int(self.in_channels * self.N), self.kernel_size, self.kernel_size)

            # x should be of the size (sub-batch-size , (self.N * in_channels) , kernel_size, kernel_size)
            act = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.N)


        if torch.sum(torch.isnan(act)) != 0:

            print('act nan')
            print(act)
            exit()

        return act


