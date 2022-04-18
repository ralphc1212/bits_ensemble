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
            return torch.round(input)
            # return torch.floor(input)

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input
    return round().apply

def df_lt(a, b, temp):
    # if a > b
    return torch.sigmoid((a - b) / temp)


res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)
thres_mean = nn.Parameter(torch.tensor([-1.2]*3), requires_grad=True)
thres_var  = nn.Parameter(torch.tensor([-3.2]), requires_grad=True)

N = 4
D1 = 2
D2 = 3
D = int(D1*D2)

U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

beta = torch.max(U)
alpha = torch.min(U)
buf = torch.zeros(D)

ROUND = get_round()

maxBWPow = 3

s = None
for idx, deno in enumerate(res_denos[:maxBWPow]):
    if s is None:
        s = (beta - alpha) / deno
        vals = (s * ROUND(U / s)).unsqueeze(0)
    else:
        s = s / deno
        res_errer = U - vals.sum(0)
        print('------------------')
        print('residual err: \n', res_errer)
        sorted_err, idx_maps = torch.sort(res_errer, dim=0)
        print('sorted err: \n', sorted_err)
        print('index map: \n', idx_maps)

        transfered_error = torch.einsum('ab, cba -> cb', res_errer, torch.nn.functional.one_hot(idx_maps, num_classes=4).float())
        print('transfered error: \n', transfered_error)

        delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
        delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
        print(delta_sorted_err.shape)
        # print(thres_mean.repeat_interleave(2).shape)
        mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(thres_mean.repeat_interleave(2)), 0.01)
        ### May need let the partitioning shared in a channel.

        # var_split_point = df_lt(torch.sigmoid(thres_var), delta_sorted_err.var(dim=1), 0.01)
        print(mean_split_point)
        print(torch.round(mean_split_point))
        round_split = torch.round(mean_split_point)
        values, indices, counts = torch.unique(round_split, return_inverse=True, return_counts=True, dim=1)

        print('*values:\n{}'.format(values))
        print('*indices:\n{}'.format(indices))
        print('*counts:\n{}'.format(counts))
        s_values, s_indices = torch.sort(counts, descending=True)
        print('*sorted indices:\n{}'.format(s_indices))
        print('*sorted counts:\n{}'.format(s_values))

        # find the close center
        clustered_indices = indices.clone().detach()

        for s_idx_less in s_indices[2:]:
            print('******')
            print(s_idx_less)
            print(values[:, s_idx_less])
            print(values[:, s_indices[:2]])
            intersection = values[:, s_idx_less].unsqueeze(1) * values[:, s_indices[:2]]
            print(intersection)
            print(intersection.sum(dim=0))
            _, selected_center = torch.max(intersection.sum(dim=0), dim=0)
            print(selected_center)
            print(s_indices[:2][selected_center])
            clustered_indices[clustered_indices == s_idx_less] = s_indices[:2][selected_center]

        print('*clustered indices:\n{}'.format(clustered_indices))

        pieces = []
        inner_idxs = []
        for s_idx in s_indices[:2]:
            # for gradient
            # print('-----')
            # print(mean_split_point[:, clustered_indices == s_idx])
            # print(round_split[:, clustered_indices == s_idx])
            # print(sorted_err[:, clustered_indices == s_idx])
            # print(round_split[:, s_idx])

            # print(clustered_indices == s_idx)
            # print((clustered_indices == s_idx).nonzero())
            # exit()

            inner_idxs.append((clustered_indices == s_idx).nonzero())

            piece = sorted_err[:, clustered_indices == s_idx]
            split_scheme = round_split[:, s_idx]
            print('SPLIT_SCHEME: ', split_scheme)
            gradients = mean_split_point[:, clustered_indices == s_idx]
            # print(piece)

            buf = 0.
            local_cnt = 0
            grad_ = 1.
            t_l = []
            for iidx in range(sorted_err.shape[0]):

                buf += piece[iidx]

                local_cnt += 1
                if iidx == len(idx_maps) - 1:
                    t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))

                # # the 1st way for the gradients
                # elif torch.round(split_scheme[iidx]):
                #     grad_ = grad_ * mean_split_point[iidx, s_idx]
                #     # print('gradients: ', gradients[iidx])
                #     t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                #     buf = 0.
                #     local_cnt = 0
                #     grad_ = 1.
                # else:
                #     grad_ = grad_ * (1 - mean_split_point[iidx, s_idx])
                #     # print('gradients: ', gradients[iidx])

                # the 2nd way for the gradients
                else:
                    pos = ROUND(gradients[iidx]) * gradients[iidx]
                    neg = (1 - ROUND(gradients[iidx])) * (1 - gradients[iidx])
                    grad_ = grad_ * (pos + neg)

                    if torch.round(split_scheme[iidx]):
                        t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                        buf = 0.
                        local_cnt = 0
                        grad_ = 1.

            pieces.append(torch.cat(t_l, dim=0))

        # print(inner_idxs)
        # print(torch.cat(inner_idxs))
        # print(torch.nn.functional.one_hot(torch.cat(inner_idxs).squeeze(dim=1), num_classes=D))
        # print(pieces)
        # print(sorted_err)
        # print(sorted_err.mm(torch.nn.functional.one_hot(torch.cat(inner_idxs).squeeze(dim=1), num_classes=D).float().t()))
        # print(torch.cat(pieces, dim=1))
        # exit()

        inner_grouped = torch.cat(pieces, dim=1).mm(torch.nn.functional.one_hot(torch.cat(inner_idxs).squeeze(dim=1), num_classes=D).float())

        print(torch.cat(pieces, dim=1))
        print(torch.cat(inner_idxs).squeeze(dim=1))
        inner_idxs = torch.cat(inner_idxs).squeeze(dim=1)
        x = torch.zeros(inner_idxs.shape[0]).index_put(indices=[inner_idxs], values=torch.arange(inner_idxs.shape[0]).float()).long()
        print(x)
        print(torch.index_select(torch.cat(pieces, dim=1), 1, x))
        print('inner_grouped\n', inner_grouped)
        print('sorted_err\n', sorted_err)

        grouped = torch.einsum('cb, cba -> ab', inner_grouped, torch.nn.functional.one_hot(idx_maps, num_classes=4).float())
        print('output: \n', grouped)
        quantized = s * ROUND(grouped / s)

        vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

w = vals.sum(0)

print(vals)

print(U)

print(w)
