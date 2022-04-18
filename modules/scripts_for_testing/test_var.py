import torch
import time
import torch.nn.functional as F

# a = torch.sigmoid(torch.rand((1000,)))
# print(a)
# print(a)
# print(a.var())


a = [[2,5,1],[5,1,0],[4,6,2],[14,16,12]]
a = torch.tensor(a)
print(len(a))
print(a)
s, i = torch.sort(a,dim=0)
print(s)
print(i)

# print(a[i[0]])

for idx, imap in enumerate(i):
	print(idx, imap)
	print(zip(imap, [0,1,2]))
	exit()
	print(a[zip(imap, [0,1,2])])
