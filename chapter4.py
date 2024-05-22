import torch
import torch.nn.functional as F


# Fig. 4.1
input_data = torch.tensor([[[[1, 1, 0, 2, 0], [1, 2, 2, 3, 1], [1, 2, 2, 1, 0], [0, 1, 1, 3, 2], [0, 0, 0, 1, 1]]]])
weight = torch.tensor([[[[-1, 2, 2], [0, -2, 3], [0, -1, -2]]]])
output_data = F.conv2d(input_data, weight, stride=1, padding=1)
print(output_data)

# Fig. 4.2
input_data = torch.tensor([[[[1, 1, 0, 2, 0], [1, 2, 2, 3, 1], [1, 2, 2, 1, 0], [0, 1, 1, 3, 2], [0, 0, 0, 1, 1]]]])
weight = torch.tensor([[[[-1, 2, 2], [0, -2, 3], [0, -1, -2]]]])
output_data = F.conv2d(input_data, weight, stride=1, padding=1)
print(output_data)

input_data = torch.tensor([[[[2, 1, 1, 2, 3], [3, 2, 1, 2, 1], [1, 0, 1, 1, 0], [0, 1, 2, 2, 3], [1, 0, 1, 3, 1]]]])
weight = torch.tensor([[[[0, 1, -2], [3, -2, 2], [0, 0, 1]]]])
output_data = F.conv2d(input_data, weight, stride=1, padding=1)
print(output_data)

input_data = torch.tensor([[[[3, 1, 0, 1, 0], [2, 1, 2, 1, 3], [2, 2, 2, 3, 0], [0, 3, 2, 1, 0], [2, 0, 0, 1, 3]]]])
weight = torch.tensor([[[[-2, -2, 0], [1, 0, 3], [5, -1, 0]]]])
output_data = F.conv2d(input_data, weight, stride=1, padding=1)
print(output_data)

input_data = torch.tensor([[[[1, 1, 0, 2, 0], [1, 2, 2, 3, 1], [1, 2, 2, 1, 0], [0, 1, 1, 3, 2], [0, 0, 0, 1, 1]],
                            [[2, 1, 1, 2, 3], [3, 2, 1, 2, 1], [1, 0, 1, 1, 0], [0, 1, 2, 2, 3], [1, 0, 1, 3, 1]],
                            [[3, 1, 0, 1, 0], [2, 1, 2, 1, 3], [2, 2, 2, 3, 0], [0, 3, 2, 1, 0], [2, 0, 0, 1, 3]]]])
weight = torch.tensor([[[[-1, 2, 2], [0, -2, 3], [0, -1, -2]],
                        [[0, 1, -2], [3, -2, 2], [0, 0, 1]],
                        [[-2, -2, 0], [1, 0, 3], [5, -1, 0]]]])
output_data = F.conv2d(input_data, weight, stride=1, padding=1)
print(output_data)

# Fig. 4.3
input_data = torch.tensor([[[[1., 1., 0., 2., 0.], [1., 2., 2., 3., 1.], [1., 2., 2., 1., 0.], [0., 1., 1., 3., 2.], [0., 0., 0., 1., 1.]]]])
output_data = F.max_pool2d(input_data, kernel_size=2, stride=2)
print(output_data)

# Fig. 4.4
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'

x = torch.arange(-1, 1.1, 0.1)
y = torch.nn.functional.relu(x)
plt.figure(constrained_layout=True, figsize=(6.5, 5), dpi=600)
plt.plot(x, y)
plt.xlabel('$x$', fontsize=12)
plt.ylabel(r'ReLU($x$)', fontsize=12, fontname='times new roman')
plt.grid(color='lightgray', alpha=0.3)
plt.savefig('Fig. 44.pdf')
plt.savefig('Fig. 44.png')