# Import modules
import os
import time
import torch
import random
import argparse
import torchvision
import numpy as np
from PIL import Image
from attOfSVNet import Network

# Settings
parser = argparse.ArgumentParser(description='Model evaluation')
parser.add_argument('--dataset', type=str, default='CNSig', help='dataset name')
parser.add_argument('--epochs', type=int, default=100, help='training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_channels', type=int, default=1, help='number of image channels')
parser.add_argument('--image_size', type=tuple, default=(128, 256), help='image size')
args = parser.parse_args()
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'models/%s/best.pth'%args.dataset


# Modeling
model = Network(args.num_channels).to(device)
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))


# Model predict
model.eval()
t1 = time.time()
img1 = Image.open('datasets/CNSig/pred/1_11.png').convert('L') #真实签名
img2 = Image.open('datasets/CNSig/pred/1_20.png').convert('L') #待认证签名（伪造签名）
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize(args.image_size)])
img1 = 1 - transform(img1)
img2 = 1 - transform(img2)
x1 = torch.unsqueeze(img1, 0)
x2 = torch.unsqueeze(img2, 0)
x1 = x1.to(device)
x2 = x2.to(device)
output = model(x1, x2)
# 计算准确率
pred = torch.where(output >= 0.5, 1, 0)
if pred == 0:
    print('待认证签名为伪造签名')
else:
    print('待认证签名为真实签名')
t2 = time.time()
times = t2 - t1
print('Time taken per sample: %.2f seconds'%times)

t1 = time.time()
img1 = Image.open('datasets/CNSig/pred/2_11.png').convert('L') #真实签名
img2 = Image.open('datasets/CNSig/pred/2_21.png').convert('L') #待认证签名（真实签名）
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize(args.image_size)])
img1 = 1 - transform(img1)
img2 = 1 - transform(img2)
x1 = torch.unsqueeze(img1, 0)
x2 = torch.unsqueeze(img2, 0)
x1 = x1.to(device)
x2 = x2.to(device)
output = model(x1, x2)
# 计算准确率
pred = torch.where(output >= 0.5, 1, 0)
if pred == 0:
    print('待认证签名为伪造签名')
else:
    print('待认证签名为真实签名')
t2 = time.time()
times = t2 - t1
print('Time taken per sample: %.2f seconds'%times)