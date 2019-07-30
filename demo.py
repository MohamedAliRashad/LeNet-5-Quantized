import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from LeNet import LeNet5, quantize_arr

NUM_Rows = 5
NUM_Coloums = 5
NUM_Filters = 6

matrix = []
Biases = []
Weights = []

# for n in range(NUM_Filters):
#     print("Filter {}:".format(n))
#     for i in range(NUM_Rows):
#         rows = input()
#         matrix.append([float(x) for x in rows.split()[:NUM_Coloums]])
#     Biases.append(float(input("Bias: ")))
#     Weights.append(torch.from_numpy(np.array(matrix)).unsqueeze(0).numpy())
#     matrix.clear()

# print("Filter {}:".format(0))
# for i in range(NUM_Rows):
#     rows = input()
#     matrix.append([float(x) for x in rows.split()[:NUM_Coloums]])

# print(torch.from_numpy(np.array(Weights)).shape)
# print(torch.from_numpy(np.array(Biases)).shape)
model = LeNet5()
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].data)

with open("inputs.txt", 'r') as f:
    Weights = f.read().split()

Weights = [float(x) for x in Weights]
Weights = np.expand_dims(np.expand_dims(np.reshape(np.array(Weights), (5, 5)), axis=0), axis=0)
Weights = np.repeat(Weights, 6, axis=0)

Biases = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)

# model.conv1.weight.data = torch.from_numpy(np.array(Weights))
# model.conv1.bias.data = torch.from_numpy(np.array(Biases))

model.conv1.weight.data = torch.from_numpy(Weights).float()
model.conv1.bias.data = torch.from_numpy(Biases).float()

print(model.conv1.weight.data)
print(model.conv1.bias.data)

transform = [transforms.Pad(2), transforms.ToTensor()]

test_data = datasets.MNIST(root='data',
                           train=False,
                           download=True,
                           transform=transforms.Compose(transform))

test_loader = DataLoader(test_data,
                         batch_size=1,
                         num_workers=0)
# Input image for analysis
input_img = next(iter(test_loader))[0][0].squeeze(0)
plt.imshow(input_img)

# Modifying the input image for analysis
input_img = input_img.unsqueeze(0).unsqueeze(0)

plt.figure(figsize=(10, 10))
row = 2
columns = 3
for i in range(6):
    output, min_val, max_val = quantize_arr(
        F.relu(model.conv1.forward(input_img))[0][i].detach().numpy())
    plt.subplot(6 / columns + 1, columns, i + 1)
    plt.imshow(output)

plt.show()