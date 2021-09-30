# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:15:12 2021

@author: fayya
"""


import torch
from DiffJPEG import DiffJPEG
import numpy as np
from skimage import data
import matplotlib.pyplot as plt
original = data.astronaut()/255.0
jpeg = DiffJPEG(height=512, width=512, differentiable=True, quality=5)
x0 = torch.from_numpy(original).float()
x0 = torch.unsqueeze(x0,dim=0)
x = x0.permute((0,3,1,2))
xp = jpeg(x)
xp0 = xp.permute((0,2,3,1))
xp0n = xp0.detach().numpy()[0]
x0n = x0.detach().numpy()[0]
print(np.linalg.norm(x0n-xp0n))
f, axarr = plt.subplots(1,2)
axarr[0].imshow(x0n)
axarr[1].imshow(xp0n)
