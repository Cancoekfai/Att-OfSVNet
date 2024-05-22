# Import modules
import os
import time
import torch
import random
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from load_data_vis import Dataset
from sklearn.metrics import roc_curve
from attOfSVNet_ablation import Network
from sklearn.metrics import confusion_matrix

plt.rcParams['font.family'] = 'Times New Roman'


# Settings
parser = argparse.ArgumentParser(description='Model evaluation')
parser.add_argument('--train_dataset', type=str, default='CNSig', help='dataset name, options: [CNSig, CEDAR, BHSig-B, BHSig-H]')
parser.add_argument('--test_dataset', type=str, default='BHSig-H', help='dataset name, options: [CNSig, CEDAR, BHSig-B, BHSig-H]')
parser.add_argument('--epochs', type=int, default=100, help='training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_channels', type=int, default=1, help='number of image channels')
parser.add_argument('--image_size', type=tuple, default=(128, 256), help='image size')
parser.add_argument('--inv', type=bool, default=True, help='whether to inversion')
parser.add_argument('--att', type=bool, default=True, help='whether to 2D Attention')
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
test_data_path = 'datasets/%s/test.csv'%args.test_dataset
model_path = 'models_%s_%s_%s/best.pth'%(args.train_dataset, args.inv, args.att)


# load data
test_dataset = Dataset(test_data_path, args, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                       torchvision.transforms.Resize(args.image_size)]))
test_db = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
print('test  data: images:', (len(test_dataset), args.num_channels, args.image_size[0], args.image_size[1]), 'labels:', len(test_dataset))


# Modeling
model = Network(args).to(device)
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))


# Model test
total = 0
correct = 0
Y = np.array([], dtype=np.int32)
outputs = np.array([], dtype=np.int32)
prediction = np.array([], dtype=np.int32)
model.eval()
t1 = time.time()
with tqdm(total=len(test_db)) as pbar:
    for step, (x1, x2, y) in enumerate(test_db):
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        Y = np.append(Y, y.cpu())
        output = model(x1, x2)
        outputs = np.append(outputs, output.cpu().detach().numpy())
        # 计算准确率
        pred = torch.where(output >= 0.5, 1, 0)
        prediction = np.append(prediction, pred.cpu())
        total += y.shape[0]
        correct += int((pred == y).sum())
        pbar.set_postfix({'accuracy': '%.2f'%((correct/total) * 100) + '%'})
        pbar.update(1)
t2 = time.time()
times = t2 - t1
print('Time taken per sample: %.2f seconds'%(times/total))

TN, FP, FN, TP = confusion_matrix(Y, prediction).ravel()
FRR = FN / (TP + FN)
FAR = FP / (FP + TN)
FPR, TPR, thresholds = roc_curve(Y, outputs)
eer_threshold = thresholds[np.nanargmin(np.abs(FPR - (1 - TPR)))]
EER = FPR[np.nanargmin(np.abs(FPR - (1 - TPR)))]
print('False Rejection Rate (FRR): %.2f'%(FRR * 100))
print('False Acceptance Rate (FAR): %.2f'%(FAR * 100))
print('Accuracy (ACC): %.2f'%((correct/total) * 100))
print('Equal Error Rate (EER): %.2f'%(EER * 100))