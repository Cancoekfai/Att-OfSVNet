# Att-OfSVNet

Code to run Offline Signature Verification from my research Offline Signature Verification Based on Att-OfSVNet on a Large Chinese Signature Dataset.


# Installation

## Basic

- `Python` >= 3.6

## Modules

- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `torch` >= 1.8.1
- `torchvision` >= 0.9.1
- `tqdm`

```shell
pip install -r requirements.txt
```


# Data Preparation

The samples were paired using the following script. Many of the datasets were collected by other researchers. Please cite their papers if you use the data.

```shell
python prepare_data.py --dataset dataset
```

- `CEDAR`: 1320 images genuine signature images and 1320 forged signature images from the [CEDAR dataset](https://cedar.buffalo.edu/NIJ/data/) [[Citation](https://github.com/Cancoekfai/Att-OfSVNet/blob/main/datasets/bibtex/CEDAR.tex)].
- `BHSig260`: [BHSig260 dataset](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E/edit?resourcekey=0-MUNnTzBi4h_VE0J84NDF3Q) [[Citation](https://github.com/Cancoekfai/Att-OfSVNet/blob/main/datasets/bibtex/BHSig.tex)] contains two sub-datasets, BHSig-B and BHSig-H. The BHSig-B dataset has 2400 images genuine signature images and 3000 forged signature images. The BHSig-H dataset has 3840 images genuine signature images and 4800 forged signature images.
- `CNSig`: 8360 images genuine signature images and 8360 forged signature images from the [CNSig dataset](https://drive.google.com/file/d/1Co6eQi42FA1Nwa2L3_4lp1xaVR-Yb1nw/view?usp=drive_link).
