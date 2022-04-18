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


res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)
thres_mean = nn.Parameter(torch.tensor([-1.2]), requires_grad=True)
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

        # print('one hot index map: \n', torch.nn.functional.one_hot(idx_maps, num_classes=4))
        # print('one hot index map: \n', torch.nn.functional.one_hot(idx_maps, num_classes=4).shape)
        # print(res_errer.shape)
        # t_er = torch.einsum('ab, cba -> cb', res_errer, torch.nn.functional.one_hot(idx_maps, num_classes=4).float())
        # print(t_er)
        # o_er = torch.einsum('cb, cba -> ab', sorted_err, torch.nn.functional.one_hot(idx_maps, num_classes=4).float())
        # print(o_er)

        delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
        delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
        mean_split_point = df_lt(delta_sorted_err.mean(dim=1), torch.sigmoid(thres_mean), 0.01)
        # var_split_point = df_lt(torch.sigmoid(thres_var), delta_sorted_err.var(dim=1), 0.01)

        split_point = mean_split_point

        # split_point = mean_split_point * var_split_point
        print('split point: \n', split_point)
        print('split point: \n', torch.round(split_point))

        print(split_point)
        exit()

        buf = 0.
        local_cnt = 0
        grad_ = 1.
        t_l = []
        for iidx, idx_map in enumerate(idx_maps):

            buf += sorted_err[iidx]

            local_cnt += 1
            if iidx == len(idx_maps) - 1:
                # t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                t_l.append((buf / local_cnt).unsqueeze(0).expand(local_cnt, -1))
            elif torch.round(split_point[iidx]):
                grad_ = grad_ * split_point[iidx]
                # t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
                t_l.append((buf / local_cnt).unsqueeze(0).expand(local_cnt, -1))
                buf = 0.
                local_cnt = 0
                grad_ = 1.
            else:
                grad_ = grad_ * (1 - split_point[iidx])

        grouped = torch.einsum('cb, cba -> ab', torch.cat(t_l, dim=0), torch.nn.functional.one_hot(idx_maps, num_classes=4).float())
        print('output: \n', grouped)
        quantized = s * ROUND(grouped / s)

        vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

w = vals.sum(0)

print(vals)

print(U)

print(w)
