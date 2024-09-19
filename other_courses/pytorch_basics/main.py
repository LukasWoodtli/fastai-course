import torch
import torch.nn as nn
import torch.optim as optim

tnsr = torch.Tensor([[[3,2],[1,5]], [[6,4],[9,6]], [[2,4],[0,5]]])
print(tnsr)

print(tnsr.device)

# device can be changed to gpu or CUDA if available
tnsr.to(device="cpu")

print(tnsr.shape)

# indexing
print(tnsr[0])

# special tensors
torch.ones_like(tnsr)
torch.zeros_like(tnsr)

torch.randn_like(tnsr)


# Neural Network
linear = nn.Linear(10, 2)
input = torch.randn(3, 10)
output = linear(input)
print(output)

relu = nn.ReLU()
relu_output = relu(output)


# Optimizers
mlp_layer = nn.Sequential(nn.Linear(5, 2), nn.BatchNorm1d(2), nn.ReLU())
input = torch.randn(5, 5) + 1
output = mlp_layer(input)
print(output)

adam_opt = optim.Adam(mlp_layer.parameters(), lr=1e-1)
print(adam_opt)


# Training example
train_ex = torch.randn(100, 5) + 1
# zeroing gradients
adam_opt.zero_grad()
curr_loss = torch.abs(1 - mlp_layer(train_ex)).mean()
curr_loss.backward()
adam_opt.step()
print(curr_loss)
