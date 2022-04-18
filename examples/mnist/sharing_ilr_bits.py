import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.append('../')
from models import ECELoss, set_temp_tcp_baens
from models import ILR

ece = ECELoss()
device = torch.device("cpu")


def load_model(member_index):
    ckpt_path_name = './checkpoint/logistic-regression/' + suffix + '/' + str(member_index) + '-' + ckpt_file

    model = ILR(input_dim=784, output_dim=10)

    model.load_state_dict(torch.load(ckpt_path_name, map_location=device).state_dict())

    model = model.to(device)

    model.eval()

    return model

def df_lt(a, b, temp):
    # if a > b
    return torch.sigmoid((a - b) / temp)

def rbf_wmeans(x, means, sigma, eps):
    return torch.exp(- torch.abs(x - means + eps) / (2 * sigma ** 2))

def vanilla_bits_sharing(weight, bit_levels=3):

    N = weight.shape[0]
    D1 = weight.shape[1]
    D2 = weight.shape[2]

    v = weight.view(weight.shape[0], -1)
    # torch.nn.init.kaiming_normal_(v)

    beta = torch.max(v)
    alpha = torch.min(v)

    thres_means = nn.Parameter(torch.tensor([[-3.] * int(D1 * D2), 
        [-1.] * int(D1 * D2), 
        [0.] * int(D1 * D2), 
        [0.] * int(D1 * D2)]), requires_grad=True).to(device)

    res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False).to(device)
    zeros = torch.zeros(1, v.shape[1]).to(device)

    rbf_mu = nn.Parameter(torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]).unsqueeze(dim=0), requires_grad=False).to(device)
    rbf_sigma = nn.Parameter(torch.tensor([0.6]), requires_grad=False).to(device)
    eps = nn.Parameter(torch.tensor([10 ** (-16)]), requires_grad=False).to(device)

    transfer_matrix = [None] * (bit_levels - 1)
    split_point = [None] * (bit_levels - 1)
    s = None

    for idx, deno in enumerate(res_denos[:bit_levels]):
        if s is None:
            s = (beta - alpha) / deno
            vals = (s * torch.round(v / s)).unsqueeze(0)
        else:
            s = s / deno
            res_errer = v - vals.sum(0)

            # print('--------'+str(idx))
            # print(res_errer.mean(0)[0].mean())
            # print(torch.log(res_errer.mean(0)[0].mean()/(1-res_errer.mean(0)[0].mean())))
            # print(torch.median(res_errer, dim=0)[0].mean())
            # print(torch.log(torch.median(res_errer, dim=0)[0].mean()/torch.median(res_errer, dim=0)[0].mean()))

            if idx == 1:
                sorted_err, idx_maps = torch.sort(res_errer, dim=0)

                transfer_matrix[idx - 1] = torch.nn.functional.one_hot(idx_maps, num_classes=N).float().detach()
                delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
                delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
                # 64 is the k
                # need to be changed if for other layers

                mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(thres_means[idx - 1]), 0.01)
                split_point[idx - 1] = mean_split_point.detach()
                round_split = torch.round(torch.cat([zeros, mean_split_point]))

                # print(round_split)
                # print(round_split.shape)
                print(round_split.sum(dim=0).mean())
                clustering_vec = rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), rbf_mu, rbf_sigma, eps)
                # print(clustering_vec)
                clustering_vec = torch.round(clustering_vec)
                inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0)+eps))
                grouped = torch.einsum('cb, cba -> ab', inner_grouped, transfer_matrix[idx - 1])
            else:
                grouped = res_errer

            if idx == 2:
                quantized = s * torch.round(grouped / s).clamp(-8, 8)
            elif idx == 1:
                quantized = s * torch.round(grouped / s).clamp(-2, 2)
            elif idx == 3:
                quantized = s * torch.round(grouped / s).clamp(-256, 256)
            elif idx == 4:
                quantized = s * torch.round(grouped / s).clamp(-65536, 65536)
            # quantized = s * torch.round(grouped / s)
            # print(grouped)
            # print(res_errer)
            vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

    vals = vals.sum(0)
    return vals.view(N, D1, D2)

mode = 'fp'
nmembers = 16
suffix = '-'.join(['nmember', str(nmembers)])

ckpt_file = 'ckpt.pytorch'

ensemble = []

for i in range(nmembers):
    ensemble.append(load_model(i))

print(ensemble[0].state_dict().keys())

# put the weights together

weights_l1 = []
weights_l2 = []

for i in range(nmembers):
    weights_l1.append(ensemble[i].state_dict()['linear1.weight'].permute(1, 0))
    weights_l2.append(ensemble[i].state_dict()['linear2.weight'].permute(1, 0))

weights_l1 = torch.stack(weights_l1, dim=0).to(device)
weights_l2 = torch.stack(weights_l2, dim=0).to(device)
print(ensemble[0].state_dict()['linear2.weight'].mean())
print(ensemble[0].state_dict()['linear2.bias'].mean())


##############################

# # test 0. 8-bit quantization 
# # thres mean: -3.3594, -7.4433
# grouped_l1 = vanilla_bits_sharing(weights_l1, 5)

# # thres mean: -3.5006, -7.1668
# grouped_l2 = vanilla_bits_sharing(weights_l2, 5)

# for i in range(nmembers):
#     # ensemble[i].state_dict()['linear1.weight'] = grouped_l1[i].permute(1, 0)

#     # ensemble[i].state_dict()['linear2.weight'] = grouped_l2[i].permute(1, 0)

#     new_dict = {'linear1.weight': grouped_l1[i].permute(1, 0), 'linear2.weight': grouped_l2[i].permute(1, 0)}
#     ensemble[i].load_state_dict(new_dict, strict=False)
#     print(ensemble[0].state_dict()['linear2.weight'].mean())
#     print(ensemble[0].state_dict()['linear2.bias'].mean())

#     # exit()
#     dir_path = './checkpoint/logistic-regression/' + suffix + '/test0/'
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)

#     save_path = './checkpoint/logistic-regression/' + suffix + '/test0/' + str(i) + '-' + ckpt_file

#     torch.save(ensemble[i].state_dict(), save_path)

##############################

# # test 1. only group epsilon 32
# # thres mean: -3.3594, -7.4433
# grouped_l1 = vanilla_bits_sharing(weights_l1, 5)

# # thres mean: -3.5006, -7.1668
# grouped_l2 = vanilla_bits_sharing(weights_l2, 5)

# for i in range(nmembers):
#     # ensemble[i].state_dict()['linear1.weight'] = grouped_l1[i].permute(1, 0)

#     # ensemble[i].state_dict()['linear2.weight'] = grouped_l2[i].permute(1, 0)

#     new_dict = {'linear1.weight': grouped_l1[i].permute(1, 0), 'linear2.weight': grouped_l2[i].permute(1, 0)}
#     ensemble[i].load_state_dict(new_dict, strict=False)
#     # print(ensemble[0].state_dict()['linear2.weight'].mean())
#     # print(ensemble[0].state_dict()['linear2.bias'].mean())

#     # exit()
#     dir_path = './checkpoint/logistic-regression/' + suffix + '/test1/'
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)

#     save_path = './checkpoint/logistic-regression/' + suffix + '/test1/' + str(i) + '-' + ckpt_file

#     torch.save(ensemble[i].state_dict(), save_path)

# averaged paritioning
# 0.0637
# 0.2094

# ##############################

# # test 2. only group epsilon 16
# # thres mean: -3.3594, -7.4433
# grouped_l1 = vanilla_bits_sharing(weights_l1, 4)

# # thres mean: -3.5006, -7.1668
# grouped_l2 = vanilla_bits_sharing(weights_l2, 4)

# for i in range(nmembers):
#     # ensemble[i].state_dict()['linear1.weight'] = grouped_l1[i].permute(1, 0)

#     # ensemble[i].state_dict()['linear2.weight'] = grouped_l2[i].permute(1, 0)

#     # print(ensemble[0].state_dict()['linear2.weight'].mean())
#     # print(ensemble[0].state_dict()['linear2.bias'].mean())
#     new_dict = {'linear1.weight': grouped_l1[i].permute(1, 0), 'linear2.weight': grouped_l2[i].permute(1, 0)}
#     ensemble[i].load_state_dict(new_dict, strict=False)
#     # print(ensemble[0].state_dict()['linear2.weight'].mean())
#     # print(ensemble[0].state_dict()['linear2.bias'].mean())

#     # exit()
#     dir_path = './checkpoint/logistic-regression/' + suffix + '/test2/'
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)

#     save_path = './checkpoint/logistic-regression/' + suffix + '/test2/' + str(i) + '-' + ckpt_file

#     torch.save(ensemble[i].state_dict(), save_path)

# averaged paritioning
# 0.0411
# 0.2039


# ##############################

# # # test 3. only group epsilon 8
# # # thres mean: -3.3594, -7.4433
# grouped_l1 = vanilla_bits_sharing(weights_l1, 3)

# # thres mean: -3.5006, -7.1668
# grouped_l2 = vanilla_bits_sharing(weights_l2, 3)

# for i in range(nmembers):
#     # ensemble[i].state_dict()['linear1.weight'] = grouped_l1[i].permute(1, 0)

#     # ensemble[i].state_dict()['linear2.weight'] = grouped_l2[i].permute(1, 0)

#     # print(ensemble[0].state_dict()['linear2.weight'].mean())
#     # print(ensemble[0].state_dict()['linear2.bias'].mean())
#     new_dict = {'linear1.weight': grouped_l1[i].permute(1, 0), 'linear2.weight': grouped_l2[i].permute(1, 0)}
#     ensemble[i].load_state_dict(new_dict, strict=False)
#     # print(ensemble[0].state_dict()['linear2.weight'].mean())
#     # print(ensemble[0].state_dict()['linear2.bias'].mean())

#     # exit()
#     dir_path = './checkpoint/logistic-regression/' + suffix + '/test3/'
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)

#     save_path = './checkpoint/logistic-regression/' + suffix + '/test3/' + str(i) + '-' + ckpt_file

#     torch.save(ensemble[i].state_dict(), save_path)

# # averaged paritioning
# # 0.0632
# # 0.2719

# ##############################

# # # test 4. only group epsilon 4
# # # thres mean: -3.3594, -7.4433
# grouped_l1 = vanilla_bits_sharing(weights_l1, 2)

# # thres mean: -3.5006, -7.1668
# grouped_l2 = vanilla_bits_sharing(weights_l2, 2)

# for i in range(nmembers):
#     # ensemble[i].state_dict()['linear1.weight'] = grouped_l1[i].permute(1, 0)

#     # ensemble[i].state_dict()['linear2.weight'] = grouped_l2[i].permute(1, 0)

#     # print(ensemble[0].state_dict()['linear2.weight'].mean())
#     # print(ensemble[0].state_dict()['linear2.bias'].mean())
#     new_dict = {'linear1.weight': grouped_l1[i].permute(1, 0), 'linear2.weight': grouped_l2[i].permute(1, 0)}
#     ensemble[i].load_state_dict(new_dict, strict=False)
#     # print(ensemble[0].state_dict()['linear2.weight'].mean())
#     # print(ensemble[0].state_dict()['linear2.bias'].mean())

#     # exit()
#     dir_path = './checkpoint/logistic-regression/' + suffix + '/test4/'
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)

#     save_path = './checkpoint/logistic-regression/' + suffix + '/test4/' + str(i) + '-' + ckpt_file

#     torch.save(ensemble[i].state_dict(), save_path)

# averaged paritioning
# 0.1283
# 0.1148


# ##############################

# # test 5. group epsilon 4 and epsilon 8
# # thres mean: -3.3594, -7.4433

# grouped_l1 = vanilla_bits_sharing(weights_l1, 3)

# # thres mean: -3.5006, -7.1668
# grouped_l2 = vanilla_bits_sharing(weights_l2, 3)
# # exit()
# for i in range(nmembers):
#     # ensemble[i].state_dict()['linear1.weight'] = grouped_l1[i].permute(1, 0)

#     # ensemble[i].state_dict()['linear2.weight'] = grouped_l2[i].permute(1, 0)

#     # print(ensemble[0].state_dict()['linear2.weight'].mean())
#     # print(ensemble[0].state_dict()['linear2.bias'].mean())
#     new_dict = {'linear1.weight': grouped_l1[i].permute(1, 0), 'linear2.weight': grouped_l2[i].permute(1, 0)}
#     ensemble[i].load_state_dict(new_dict, strict=False)
#     # print(ensemble[0].state_dict()['linear2.weight'].mean())
#     # print(ensemble[0].state_dict()['linear2.bias'].mean())

#     # exit()
#     dir_path = './checkpoint/logistic-regression/' + suffix + '/test5/'
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)

#     save_path = './checkpoint/logistic-regression/' + suffix + '/test5/' + str(i) + '-' + ckpt_file

#     torch.save(ensemble[i].state_dict(), save_path)

# averaged paritioning
# eps 4
# 3.7298
# 1.1958
# eps 8
# 5.0891
# 2.5352


# ##############################

# # # test 6. only group epsilon 4 // thres -3
# # # thres mean: -3.3594, -7.4433
grouped_l1 = vanilla_bits_sharing(weights_l1, 2)

# thres mean: -3.5006, -7.1668
grouped_l2 = vanilla_bits_sharing(weights_l2, 2)

for i in range(nmembers):
    # ensemble[i].state_dict()['linear1.weight'] = grouped_l1[i].permute(1, 0)

    # ensemble[i].state_dict()['linear2.weight'] = grouped_l2[i].permute(1, 0)

    # print(ensemble[0].state_dict()['linear2.weight'].mean())
    # print(ensemble[0].state_dict()['linear2.bias'].mean())
    new_dict = {'linear1.weight': grouped_l1[i].permute(1, 0), 'linear2.weight': grouped_l2[i].permute(1, 0)}
    ensemble[i].load_state_dict(new_dict, strict=False)
    # print(ensemble[0].state_dict()['linear2.weight'].mean())
    # print(ensemble[0].state_dict()['linear2.bias'].mean())

    # exit()
    dir_path = './checkpoint/logistic-regression/' + suffix + '/test6/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    save_path = './checkpoint/logistic-regression/' + suffix + '/test6/' + str(i) + '-' + ckpt_file

    torch.save(ensemble[i].state_dict(), save_path)

# averaged paritioning
# 0.9276
# 1.2125

