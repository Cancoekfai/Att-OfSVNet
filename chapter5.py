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
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_channels', type=int, default=1, help='number of image channels')
parser.add_argument('--image_size', type=tuple, default=(128, 256), help='image size')
args = parser.parse_args()
args.dataset = 'CNSig'
args.inv = True
args.att = True
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_data_path = 'datasets/%s/test.csv'%args.dataset
model_path = 'models_%s_%s_%s/best.pth'%(args.dataset, args.inv, args.att)


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
fig, ax1 = plt.subplots(constrained_layout=True, figsize=(6.5, 5), dpi=600)
ax1.set_xlabel('Thresholds', fontsize=12)
ax1.set_ylabel('FAR', fontsize=12)
ax1.plot(thresholds, FPR)
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
ax2.set_ylabel('FRR', fontsize=12)
ax2.plot(thresholds, 1 - TPR)
ax2.tick_params(axis='y')
ax2.scatter(thresholds[np.nanargmin(np.abs(FPR - (1 - TPR)))],
            EER, color='r', s=25, zorder=2)
ax2.annotate('',
             xy=(thresholds[np.nanargmin(np.abs(FPR - (1 - TPR)))], EER + 0.02),
             xytext=(thresholds[np.nanargmin(np.abs(FPR - (1 - TPR)))], EER + 0.15),
             arrowprops=dict(facecolor='r', edgecolor='r', width=0.25, headwidth=5, headlength=5))
ax2.text(thresholds[np.nanargmin(np.abs(FPR - (1 - TPR)))] - 0.03, EER + 0.17, 'EER', color='r', fontsize=12)
ax1.grid(color='lightgray', alpha=0.3)
ax2.grid(color='lightgray', alpha=0.3)
plt.savefig('Fig. 51.pdf')
plt.savefig('Fig. 51.png')
plt.show()
#%%
# Import modules
import os
import time
import torch
import random
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from load_data_vis import Dataset
from sklearn.metrics import roc_curve
from attOfSVNet_ablation import Network
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='Model evaluation')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_channels', type=int, default=1, help='number of image channels')
parser.add_argument('--image_size', type=tuple, default=(128, 256), help='image size')
args = parser.parse_args()

for dataset in ['CEDAR', 'BHSig-B', 'BHSig-H', 'CNSig']:
    for att in [False, True]:
        for inv in [False, True]:
            args.dataset = dataset
            args.inv = inv
            args.att = att
            
            # Settings
            seed = 42
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            torch.backends.cudnn.deterministic = True
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            test_data_path = 'datasets/%s/test.csv'%args.dataset
            model_path = 'models_%s_%s_%s/best.pth'%(args.dataset, args.inv, args.att)
            
            
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
            np.savez('%s_%s_%s.npz'%(args.dataset, args.inv, args.att), FPR=FPR, TPR=TPR)
#%%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

for dataset in ['CEDAR', 'BHSig-B', 'BHSig-H', 'CNSig']:
    fig, ax_main = plt.subplots(constrained_layout=True, figsize=(6.5, 5), dpi=600)
    left, bottom, width, height = [0.71, 0.41, 0.25, 0.25]
    ax_sub = fig.add_axes([left, bottom, width, height])
    methods = ['original', 'black-and-white filp', '2D Attention', 'black-and-white filp and 2D Attention']
    i = 0
    for att in [False, True]:
        for inv in [False, True]:
            metircs = np.load('%s_%s_%s.npz'%(dataset, inv, att))
            FPR = metircs['FPR']
            TPR = metircs['TPR']
            ax_main.plot(FPR, TPR, zorder=2, label=methods[i])
            ax_sub.plot(FPR, TPR, zorder=2, label=methods[i])
            i += 1
    ax_main.plot([0, 1], [0, 1], color='darkgoldenrod', alpha=0.6, linestyle='--', zorder=1, label='whole')
    ax_main.set_xlim([0.0, 1.0])
    ax_main.set_ylim([0.0, 1.05])
    ax_main.tick_params(labelsize=13)
    ax_main.set_xlabel('FPR', fontsize=15)
    ax_main.set_ylabel('TPR', fontsize=15)
    ax_main.grid(color='lightgray', alpha=0.3)
    ax_sub.plot([0, 1], [0, 1], color='darkgoldenrod', alpha=0.6, linestyle='--', zorder=1, label='whole')
    if dataset == 'CEDAR':
        ax_sub.set_xlim([-0.05, 0.15])
        ax_sub.set_ylim([0.85, 1.05])
    elif dataset == 'BHSig-B':
        ax_sub.set_xlim([-0.025, 0.25])
        ax_sub.set_ylim([0.6, 1.05])
    elif dataset == 'BHSig-H':
        ax_sub.set_xlim([0.0, 0.2])
        ax_sub.set_ylim([0.55, 0.95])
    elif dataset == 'CNSig':
        ax_sub.set_xlim([-0.005, 0.075])
        ax_sub.set_ylim([0.8, 1.01])
    ax_sub.set_xlabel('FPR')
    ax_sub.set_ylabel('TPR')
    ax_sub.grid(color='lightgray', alpha=0.3)
    ax_main.legend()
    plt.savefig('Fig. ROC_%s.pdf'%dataset)
    plt.savefig('Fig. ROC_%s.png'%dataset)
    plt.show()