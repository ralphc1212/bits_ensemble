import torch
import torch.nn as nn

# only weight quantization, without decomposition

TEMPERATURE = 0.01
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

class dense_res_bit_tree_meanvar_baens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, bw=3, quantize=True):
        super(dense_res_bit_tree_meanvar_baens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.D = D
        self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        self.res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

        self.round = get_round()
        self.maxBWPow = bw

        self.thres_mean = nn.Parameter(torch.tensor([-0.405]), requires_grad=True)
        self.thres_var  = nn.Parameter(torch.tensor([-2.197]), requires_grad=True)

        self.temperature = TEMPERATURE
        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):
        # wQ = self.quant(self.U)
        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)
        self.npartitions = [0] * (self.maxBWPow - 1)

        beta = torch.max(self.U)
        alpha = torch.min(self.U)
        buf = torch.zeros(self.D)
        # group_counts = [0] * (maxBWPow - 1)

        s = None
        for idx, deno in enumerate(self.res_denos[:self.maxBWPow]):
            if s is None:
                s = (beta - alpha) / deno
                vals = (s * self.round(self.U / s)).unsqueeze(0)
            else:
                s = s / deno
                res_errer = self.U - vals.sum(0)

                # aggregated_res_error = res_errer.sum(1)
                sorted_err, idx_maps = torch.sort(res_errer, dim=0)
                transfer_matrix = torch.nn.functional.one_hot(idx_maps, num_classes=self.N).float()

                delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
                delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
                mean_split_point = df_lt(delta_sorted_err.mean(dim=1), torch.sigmoid(self.thres_mean), 0.01)
                # var_split_point = df_lt(torch.sigmoid(self.thres_var), delta_sorted_err.var(dim=1), 0.01)
                # print('mean_split: \n', mean_split_point)
                # print('vari_split: \n', var_split_point)
                # print('index_maps: \n', idx_maps)

                # delta_sorted_err = delta_sorted_err / (torch.max(delta_sorted_err).detach())
                # split_point = mean_split_point * var_split_point
                split_point = mean_split_point

                # group_counts[idx-1] += torch.round(split_point.sum()).item() 
                self.npartitions[idx - 1] += torch.round(split_point).sum().item()

                buf = 0.
                local_cnt = 0
                grad_ = 1.
                t_l = []
                for iidx, idx_map in enumerate(idx_maps):
                    buf += sorted_err[iidx]
                    local_cnt += 1
                    if iidx == len(idx_maps) - 1:
                        t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                    elif torch.round(split_point[iidx]):
                        grad_ = grad_ * split_point[iidx]
                        t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                        buf = 0.
                        local_cnt = 0
                        grad_ = 1.
                    else:
                        grad_ = grad_ * (1 - split_point[iidx])

                grouped = torch.einsum('cb, cba -> ab', torch.cat(t_l, dim=0), transfer_matrix)
                quantized = s * self.round(grouped / s)

                vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

        w = vals.sum(0).view(self.N, self.D1, self.D2)

        act = torch.einsum('bnd, ndl -> bnl', x, w)

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act

class Conv2d_res_bit_tree_meanvar_baens(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, N=5, bw=2, quantize=True, first=False):
        super(Conv2d_res_bit_tree_meanvar_baens, self).__init__()

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

        self.res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

        # self.quant = res_quant(maxBWPow=3)
        self.round = get_round()
        self.maxBWPow = bw

        self.thres_mean = nn.Parameter(torch.tensor([-0.405]), requires_grad=True)
        self.thres_var  = nn.Parameter(torch.tensor([-2.197]), requires_grad=True)

        self.temperature = TEMPERATURE
        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):
        if not self.first:
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])

        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)
        self.npartitions = [0] * (self.maxBWPow - 1)

        beta = torch.max(self.U)
        alpha = torch.min(self.U)
        buf = torch.zeros(self.D)

        s = None
        for idx, deno in enumerate(self.res_denos[:self.maxBWPow]):
            if s is None:
                s = (beta - alpha) / deno
                vals = (s * self.round(self.U / s)).unsqueeze(0)
            else:
                s = s / deno
                res_errer = self.U - vals.sum(0)

                sorted_err, idx_maps = torch.sort(res_errer, dim=0)
                transfer_matrix = torch.nn.functional.one_hot(idx_maps, num_classes=self.N).float()

                delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
                delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
                mean_split_point = df_lt(delta_sorted_err.mean(dim=1), torch.sigmoid(self.thres_mean), 0.01)
                # var_split_point = df_lt(torch.sigmoid(self.thres_var), delta_sorted_err.var(dim=1), 0.01)
                # print('mean_split: \n', mean_split_point)
                # print('vari_split: \n', var_split_point)
                # print('index_maps: \n', idx_maps)

                # delta_sorted_err = delta_sorted_err / (torch.max(delta_sorted_err).detach())
                split_point = mean_split_point

                # split_point = mean_split_point * var_split_point

                # group_counts[idx-1] += torch.round(split_point.sum()).item()
                self.npartitions[idx - 1] += torch.round(split_point).sum().item()

                buf = 0.
                local_cnt = 0
                grad_ = 1.
                t_l = []
                for iidx, idx_map in enumerate(idx_maps):
                    buf += sorted_err[iidx]
                    # print(idx_map)
                    local_cnt += 1
                    if iidx == len(idx_maps) - 1:
                        t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                    elif torch.round(split_point[iidx]):
                        grad_ = grad_ * split_point[iidx]
                        t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                        buf = 0.
                        local_cnt = 0
                        grad_ = 1.
                    else:
                        grad_ = grad_ * (1 - split_point[iidx])

                grouped = torch.einsum('cb, cba -> ab', torch.cat(t_l, dim=0), transfer_matrix)
                quantized = s * self.round(grouped / s)

                vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

        w = vals.sum(0).view(int(self.out_channels * self.N), int(self.in_channels), self.kernel_size, self.kernel_size)

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