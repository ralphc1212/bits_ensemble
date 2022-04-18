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

class dense_res_bit_tree_meanvar_freeze_fine_partition_baens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, bw=3, quantize=True):
        super(dense_res_bit_tree_meanvar_freeze_fine_partition_baens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.D = D
        self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        self.res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

        self.round = get_round()
        self.maxBWPow = bw

        self.zeroD = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.arangeD = nn.Parameter(torch.arange(self.D).float(), requires_grad=False)
        self.thres_mean = nn.Parameter(torch.tensor([-1.3] * D2), requires_grad=True)
        self.thres_var  = nn.Parameter(torch.tensor([-2.197]), requires_grad=True)
        self.transfer_matrix = None
        self.split_point = None
        self.update_partition = True
        self.num_possible_partitions = 5
        self.clustered_indices = None
        self.s_indices = None
        self.inner_transfer_matrix = None

        self.temperature = TEMPERATURE
        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):
        # wQ = self.quant(self.U)
        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)
        self.npartitions = [0] * (self.maxBWPow - 1)

        beta = torch.max(self.U)
        alpha = torch.min(self.U)
        # group_counts = [0] * (maxBWPow - 1)

        s = None
        for idx, deno in enumerate(self.res_denos[:self.maxBWPow]):
            if s is None:
                s = (beta - alpha) / deno
                vals = (s * self.round(self.U / s)).unsqueeze(0)
            else:
                s = s / deno
                res_errer = self.U - vals.sum(0)

                if self.update_partition:
                    sorted_err, idx_maps = torch.sort(res_errer, dim=0)
                    self.transfer_matrix = torch.nn.functional.one_hot(idx_maps, num_classes=self.N).float().detach()

                    # transfered_error = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
                    delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
                    mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(self.thres_mean.repeat_interleave(self.D1)), 0.01)
                    ### May need let the partitioning shared in a channel.
                    self.split_point = mean_split_point.detach()

                    round_split = torch.round(mean_split_point)
                    values, indices, counts = torch.unique(round_split, return_inverse=True, return_counts=True, dim=1)

                    s_values, s_indices = torch.sort(counts, descending=True)

                    # find the close center
                    clustered_indices = indices.clone().detach()

                    for s_idx_less in s_indices[self.num_possible_partitions:]:
                        intersection = values[:, s_idx_less].unsqueeze(1) * values[:, s_indices[:self.num_possible_partitions]]
                        _, selected_center = torch.max(intersection.sum(dim=0), dim=0)
                        clustered_indices[clustered_indices == s_idx_less] = s_indices[:self.num_possible_partitions][selected_center]

                    self.clustered_indices = clustered_indices.detach()
                    self.s_indices = s_indices.detach()

                    pieces = []
                    inner_idxs = []
                    for s_idx in s_indices[:self.num_possible_partitions]:
                        inner_idxs.append((clustered_indices == s_idx).nonzero())

                        piece = sorted_err[:, clustered_indices == s_idx]
                        split_scheme = round_split[:, s_idx]
                        gradients = mean_split_point[:, clustered_indices == s_idx]
                        self.npartitions[idx - 1] += torch.round(split_scheme).sum().item()

                        buf = 0.
                        local_cnt = 0
                        grad_ = 1.
                        t_l = []
                        for iidx in range(sorted_err.shape[0]):

                            buf += piece[iidx]

                            local_cnt += 1
                            if iidx == len(idx_maps) - 1:
                                t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))

                            # the 1st way for the gradients
                            elif torch.round(split_scheme[iidx]):
                                grad_ = grad_ * mean_split_point[iidx, s_idx]
                                # print('gradients: ', gradients[iidx])
                                t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                                buf = 0.
                                local_cnt = 0
                                grad_ = 1.
                            else:
                                grad_ = grad_ * (1 - mean_split_point[iidx, s_idx])
                                # print('gradients: ', gradients[iidx])

                            # # the 2nd way for the gradients
                            # else:
                            #     pos = self.round(gradients[iidx]) * gradients[iidx]
                            #     neg = (1 - self.round(gradients[iidx])) * (1 - gradients[iidx])
                            #     grad_ = grad_ * (pos + neg)

                            #     if torch.round(split_scheme[iidx]):
                            #         t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                            #         buf = 0.
                            #         local_cnt = 0
                            #         grad_ = 1.

                        pieces.append(torch.cat(t_l, dim=0))

                    # self.inner_transfer_matrix = torch.nn.functional.one_hot(torch.cat(inner_idxs).squeeze(dim=1), num_classes=self.D).float().detach()
                    # inner_grouped = torch.cat(pieces, dim=1).mm(self.inner_transfer_matrix)
                    inner_idxs = torch.cat(inner_idxs).squeeze(dim=1)
                    self.inner_transfer_x = self.zeroD.clone().index_put(indices=[inner_idxs], values=self.arangeD.clone()).long().detach()
                    inner_grouped = torch.index_select(torch.cat(pieces, dim=1), 1, self.inner_transfer_x)

                    grouped = torch.einsum('cb, cba -> ab', inner_grouped, self.transfer_matrix)
                    quantized = s * self.round(grouped / s)
                else:
                    transfered_errors = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    round_split = torch.round(self.split_point)
                    pieces = []
                    for s_idx in self.s_indices[:self.num_possible_partitions]:
                        piece = transfered_errors[:, self.clustered_indices == s_idx]
                        split_scheme = round_split[:, s_idx]
                        gradients = self.split_point[:, self.clustered_indices == s_idx]
                        self.npartitions[idx - 1] += torch.round(split_scheme).sum().item()

                        buf = 0.
                        local_cnt = 0
                        t_l = []

                        for iidx in range(transfered_errors.shape[0]):
                            buf += piece[iidx]

                            local_cnt += 1
                            if iidx == transfered_errors.shape[0] - 1:
                                t_l.append((buf / local_cnt).unsqueeze(0).expand(local_cnt, -1))
                            elif torch.round(split_scheme[iidx]):
                                t_l.append((buf / local_cnt).unsqueeze(0).expand(local_cnt, -1))
                                buf = 0.
                                local_cnt = 0

                        pieces.append(torch.cat(t_l, dim=0))

                    # inner_grouped = torch.cat(pieces, dim=1).mm(self.inner_transfer_matrix)
                    inner_grouped = torch.index_select(torch.cat(pieces, dim=1), 1, self.inner_transfer_x)
                    grouped = torch.einsum('cb, cba -> ab', inner_grouped, self.transfer_matrix)
                    quantized = s * self.round(grouped / s)

                vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

        w = vals.sum(0).view(self.N, self.D1, self.D2)

        act = torch.einsum('bnd, ndl -> bnl', x, w)

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act


class Conv2d_res_bit_tree_meanvar_freeze_fine_partition_baens(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, N=5, bw=2, quantize=True, first=False):
        super(Conv2d_res_bit_tree_meanvar_freeze_fine_partition_baens, self).__init__()

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

        self.zeroD = nn.Parameter(torch.zeros(1, self.D), requires_grad=False)
        self.arangeD = nn.Parameter(torch.arange(self.D).float(), requires_grad=False)
        self.thres_mean = nn.Parameter(torch.tensor([-1.3] * out_channels), requires_grad=True)
        self.thres_var  = nn.Parameter(torch.tensor([-2.197]), requires_grad=True)
        self.transfer_matrix = None
        self.split_point = None
        self.update_partition = True
        self.clustered_indices = None
        self.s_indices = None
        self.inner_transfer_matrix = None
        self.num_possible_partitions = 5

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

        s = None
        for idx, deno in enumerate(self.res_denos[:self.maxBWPow]):
            if s is None:
                s = (beta - alpha) / deno
                vals = (s * self.round(self.U / s)).unsqueeze(0)
            else:
                s = s / deno
                res_errer = self.U - vals.sum(0)

                if self.update_partition:
                    sorted_err, idx_maps = torch.sort(res_errer, dim=0)
                    self.transfer_matrix = torch.nn.functional.one_hot(idx_maps, num_classes=self.N).float().detach()

                    # transfered_error = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
                    delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
                    mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(self.thres_mean.repeat_interleave(int(self.in_channels * self.kernel_size * self.kernel_size))), 0.01)
                    ### May need let the partitioning shared in a channel.
                    self.split_point = mean_split_point.detach()

                    round_split = torch.round(mean_split_point)
                    values, indices, counts = torch.unique(round_split, return_inverse=True, return_counts=True, dim=1)

                    s_values, s_indices = torch.sort(counts, descending=True)

                    # find the close center
                    clustered_indices = indices.clone().detach()

                    for s_idx_less in s_indices[self.num_possible_partitions:]:
                        intersection = values[:, s_idx_less].unsqueeze(1) * values[:, s_indices[:self.num_possible_partitions]]
                        _, selected_center = torch.max(intersection.sum(dim=0), dim=0)
                        clustered_indices[clustered_indices == s_idx_less] = s_indices[:self.num_possible_partitions][selected_center]

                    self.clustered_indices = clustered_indices.detach()
                    self.s_indices = s_indices.detach()

                    pieces = []
                    inner_idxs = []
                    for s_idx in s_indices[:self.num_possible_partitions]:
                        inner_idxs.append((clustered_indices == s_idx).nonzero())

                        piece = sorted_err[:, clustered_indices == s_idx]
                        split_scheme = round_split[:, s_idx]
                        gradients = mean_split_point[:, clustered_indices == s_idx]
                        self.npartitions[idx - 1] += torch.round(split_scheme).sum().item()

                        buf = 0.
                        local_cnt = 0
                        grad_ = 1.
                        t_l = []
                        for iidx in range(sorted_err.shape[0]):

                            buf += piece[iidx]

                            local_cnt += 1
                            if iidx == len(idx_maps) - 1:
                                t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))

                            # the 1st way for the gradients
                            elif torch.round(split_scheme[iidx]):
                                grad_ = grad_ * mean_split_point[iidx, s_idx]
                                # print('gradients: ', gradients[iidx])
                                t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                                buf = 0.
                                local_cnt = 0
                                grad_ = 1.
                            else:
                                grad_ = grad_ * (1 - mean_split_point[iidx, s_idx])
                                # print('gradients: ', gradients[iidx])

                            # # the 2nd way for the gradients
                            # else:
                            #     pos = self.round(gradients[iidx]) * gradients[iidx]
                            #     neg = (1 - self.round(gradients[iidx])) * (1 - gradients[iidx])
                            #     grad_ = grad_ * (pos + neg)

                            #     if torch.round(split_scheme[iidx]):
                            #         t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                            #         buf = 0.
                            #         local_cnt = 0
                            #         grad_ = 1.

                        pieces.append(torch.cat(t_l, dim=0))

                    inner_idxs = torch.cat(inner_idxs).squeeze(dim=1)
                    self.inner_transfer_x = self.zeroD.clone().index_put(indices=[inner_idxs], values=self.arangeD.clone()).long().detach()
                    inner_grouped = torch.index_select(torch.cat(pieces, dim=1), 1, self.inner_transfer_x)

                    grouped = torch.einsum('cb, cba -> ab', inner_grouped, self.transfer_matrix)
                    quantized = s * self.round(grouped / s)
                else:
                    transfered_errors = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    round_split = torch.round(self.split_point)
                    pieces = []
                    for s_idx in self.s_indices[:self.num_possible_partitions]:
                        piece = transfered_errors[:, self.clustered_indices == s_idx]
                        split_scheme = round_split[:, s_idx]
                        gradients = self.split_point[:, self.clustered_indices == s_idx]
                        self.npartitions[idx - 1] += torch.round(split_scheme).sum().item()

                        buf = 0.
                        local_cnt = 0
                        t_l = []

                        for iidx in range(transfered_errors.shape[0]):
                            buf += piece[iidx]

                            local_cnt += 1
                            if iidx == transfered_errors.shape[0] - 1:
                                t_l.append((buf / local_cnt).unsqueeze(0).expand(local_cnt, -1))
                            elif torch.round(split_scheme[iidx]):
                                t_l.append((buf / local_cnt).unsqueeze(0).expand(local_cnt, -1))
                                buf = 0.
                                local_cnt = 0

                        pieces.append(torch.cat(t_l, dim=0))

                    # inner_grouped = torch.cat(pieces, dim=1).mm(self.inner_transfer_matrix)
                    inner_grouped = torch.index_select(torch.cat(pieces, dim=1), 1, self.inner_transfer_x)
                    grouped = torch.einsum('cb, cba -> ab', inner_grouped, self.transfer_matrix)
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