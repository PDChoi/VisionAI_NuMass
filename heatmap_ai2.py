import os
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.ndimage import gaussian_filter
from colorspacious import cspace_converter
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import warnings

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import *
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torchvision
import numpy as np
import torchvision.models as models

import timm
from PIL import Image
import maxvit
import huggingface_hub
warnings.filterwarnings('ignore')

model = timm.create_model(
    'maxvit_tiny_pm_256',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

print(model)

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

readfiles = 1

path1 = '../crit_quijote_z_0_mnv0.0_full/'
path2 = '../crit_quijote_z_0_mnv0.1_full/'
path3 = '../crit_quijote_z_0_mnv0.4_full/'

outputdir = 'HMfiles/'

bins = 256
sigma = 5

train_size = 0.8

k_list = list(range(1,180,2))
kfold = KFold(n_splits = 5, shuffle=True, random_state=0)
skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
cv = skf

def myplot(x, y, s, bins=bins):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1000], [0, 1000]])
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

filelist1 = os.listdir(path1)
filelist1.sort()
filelist2 = os.listdir(path2)
filelist2.sort()
filelist3 = os.listdir(path3)
filelist3.sort()


if readfiles == 1:
    for i in range(100):
        
        df1 = pd.read_csv(path1 + filelist1[i])
        df1 = df1[df1.kind != 3]
        img1, extent = myplot(df1["x"], df1["y"], sigma)
        np.save("%sHM_xy_nu0.0_%d.npy" %(outputdir, i), img1)
        img1, extent = myplot(df1["y"], df1["z"], sigma)
        np.save("%sHM_yz_nu0.0_%d.npy" %(outputdir, i), img1)
        img1, extent = myplot(df1["z"], df1["x"], sigma)
        np.save("%sHM_zx_nu0.0_%d.npy" %(outputdir, i), img1)
        img1, extent = myplot(df1["y"], df1["x"], sigma)
        np.save("%sHM_xy_nu0.0_%d.npy" %(outputdir, i), img1)
        img1, extent = myplot(df1["z"], df1["y"], sigma)
        np.save("%sHM_yz_nu0.0_%d.npy" %(outputdir, i), img1)
        img1, extent = myplot(df1["x"], df1["z"], sigma)
        np.save("%sHM_zx_nu0.0_%d.npy" %(outputdir, i), img1)
        
        df2 = pd.read_csv(path2 + filelist2[i])
        df2 = df2[df2.kind != 3]
        img2, extent = myplot(df2["x"], df2["y"], sigma)
        np.save("%sHM_xy_nu0.1_%d.npy" %(outputdir, i), img2)
        img2, extent = myplot(df2["y"], df2["z"], sigma)
        np.save("%sHM_yz_nu0.1_%d.npy" %(outputdir, i), img2)
        img2, extent = myplot(df2["z"], df2["x"], sigma)
        np.save("%sHM_zx_nu0.1_%d.npy" %(outputdir, i), img2)
        img2, extent = myplot(df2["y"], df2["x"], sigma)
        np.save("%sHM_xy_nu0.1_%d.npy" %(outputdir, i), img2)
        img2, extent = myplot(df2["z"], df2["y"], sigma)
        np.save("%sHM_yz_nu0.1_%d.npy" %(outputdir, i), img2)
        img2, extent = myplot(df2["x"], df2["z"], sigma)
        np.save("%sHM_zx_nu0.1_%d.npy" %(outputdir, i), img2)

        df3 = pd.read_csv(path3 + filelist3[i])
        df3 = df3[df3.kind != 3]
        img3, extent = myplot(df3["x"], df3["y"], sigma)
        np.save("%sHM_xy_nu0.4_%d.npy" %(outputdir, i), img3)
        img3, extent = myplot(df3["y"], df3["z"], sigma)
        np.save("%sHM_yz_nu0.4_%d.npy" %(outputdir, i), img3)
        img3, extent = myplot(df3["z"], df3["x"], sigma)
        np.save("%sHM_zx_nu0.4_%d.npy" %(outputdir, i), img3)
        img3, extent = myplot(df3["y"], df3["x"], sigma)
        np.save("%sHM_xy_nu0.4_%d.npy" %(outputdir, i), img3)
        img3, extent = myplot(df3["z"], df3["y"], sigma)
        np.save("%sHM_yz_nu0.4_%d.npy" %(outputdir, i), img3)
        img3, extent = myplot(df3["x"], df3["z"], sigma)
        np.save("%sHM_zx_nu0.4_%d.npy" %(outputdir, i), img3)

img_array = []


for i in range(100):
    img1 = np.load("%sHM_xy_nu0.0_%d.npy" %(outputdir, i))
    img1_flt = img1.flatten()
    img1_set = [img1_flt, '0.0']
    img_array.append(img1_set)
    img1 = np.load("%sHM_yz_nu0.0_%d.npy" %(outputdir, i))
    img1_flt = img1.flatten()
    img1_set = [img1_flt, '0.0']
    img_array.append(img1_set)
    img1 = np.load("%sHM_zx_nu0.0_%d.npy" %(outputdir, i))
    img1_flt = img1.flatten()
    img1_set = [img1_flt, '0.0']
    img_array.append(img1_set)
 
    img2 = np.load("%sHM_xy_nu0.1_%d.npy" %(outputdir, i))
    img2_flt = img2.flatten()
    img2_set = [img2_flt, '0.1']
    img_array.append(img2_set)
    img2 = np.load("%sHM_yz_nu0.1_%d.npy" %(outputdir, i))
    img2_flt = img2.flatten()
    img2_set = [img2_flt, '0.1']
    img_array.append(img2_set)
    img2 = np.load("%sHM_zx_nu0.1_%d.npy" %(outputdir, i))
    img2_flt = img2.flatten()
    img2_set = [img2_flt, '0.1']
    img_array.append(img2_set)

    img3 = np.load("%sHM_xy_nu0.4_%d.npy" %(outputdir, i))
    img3_flt = img3.flatten()
    img3_set = [img3_flt, '0.4']
    img_array.append(img3_set)
    img3 = np.load("%sHM_yz_nu0.4_%d.npy" %(outputdir, i))
    img3_flt = img3.flatten()
    img3_set = [img3_flt, '0.4']
    img_array.append(img3_set)
    img3 = np.load("%sHM_zx_nu0.4_%d.npy" %(outputdir, i))
    img3_flt = img3.flatten()
    img3_set = [img3_flt, '0.4']
    img_array.append(img3_set)


train, test = train_test_split(img_array, train_size = train_size, test_size = 1 - train_size)

xtr = []
ytr = []
xts = []
yts = []
'''
for i in range(len(train)):
    xtr.append(train[i][0].reshape(-1, 1).flatten())
    ytr.append(train[i][1])


for i in range(len(test)):
    xts.append(test[i][0].reshape(-1, 1).flatten())
    yts.append(test[i][1])
'''

for i in range(len(train)):
    xtr.append(train[i][0])
    ytr.append(train[i][1])


for i in range(len(test)):
    xts.append(test[i][0])
    yts.append(test[i][1])

ytr = list(map(float, ytr))
yts = list(map(float, yts))

xtr = torch.tensor(xtr)
ytr = torch.tensor(ytr)
xts = torch.tensor(xts)
yts = torch.tensor(yts)









cross_validation_scores = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, xtr, ytr, scoring='accuracy', cv=cv)
    print(k, scores.mean())
    cross_validation_scores.append(scores.mean())

optimal_k = k_list[cross_validation_scores.index(max(cross_validation_scores))]
print(optimal_k)
knn = KNeighborsClassifier(n_neighbors = optimal_k)
knn.fit(xtr, ytr)
predict_knn = knn.predict(xts)
print('')
print('Accuracy(kNN):', accuracy_score(yts, predict_knn))
print(yts)
print(predict_knn)


