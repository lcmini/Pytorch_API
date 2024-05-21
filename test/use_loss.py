import torch

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)

targets = torch.tensor([1, 6, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 3, 1))
targets = torch.reshape(targets, (1, 1, 3, 1))

print(inputs)
print(targets)

# L1loss
loss = torch.nn.L1Loss(reduction='sum')
result = loss(inputs, targets)
print(result)

# MSEloss
loss_mse = torch.nn.MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)
