import torch
torch.manual_seed(7)
x = torch.rand(1, 2, 3, 10, device='cuda')
print(x.shape)
print(torch.squeeze(x).shape)
