# Import modules
import os
import torch
import random
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from load_data_vis import Dataset
from attOfSVNet_vis import Network

# Settings
parser = argparse.ArgumentParser(description='Model evaluation')
parser.add_argument('--dataset', type=str, default='BHSig-H', help='dataset name, options: [CNSig, CEDAR, BHSig-B, BHSig-H]')
parser.add_argument('--epochs', type=int, default=100, help='training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_channels', type=int, default=1, help='number of image channels')
parser.add_argument('--image_size', type=tuple, default=(128, 256), help='image size')
parser.add_argument('--inv', type=bool, default=False, help='whether to inversion')
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
vis_data_path = 'vis/%s/inputs/vis.csv'%args.dataset
model_path = 'models_%s_%s_%s/best.pth'%(args.dataset, args.inv, args.att)


# load data
vis_dataset = Dataset(vis_data_path, args, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                     torchvision.transforms.Resize(args.image_size)]))
vis_db = torch.utils.data.DataLoader(dataset=vis_dataset, batch_size=args.batch_size, shuffle=False)
print('test  data: images:', (len(vis_dataset), args.num_channels, args.image_size[0], args.image_size[1]), 'labels:', len(vis_dataset))


# Modeling
model = Network(args).to(device)
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))


# Model visualization
model.eval()
with tqdm(total=len(vis_db)) as pbar:
    for step, (x1, x2, y) in enumerate(vis_db):
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        output = model(x1, x2)
        pbar.update(1)