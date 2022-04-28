import numpy as np
import torch

fname = './features_1.out'
features_filename = fname

features = np.loadtxt(features_filename)
features = np.swapaxes(features, 0, 1) 
features = np.reshape(features, (1, 512, 512, 8))

direction = np.reshape(features[:, :, :, 3], (1, 512, 512, 1))
data = features[:, :, :, 3]
# print(data.shape)
dist = np.reshape(features[:, :, :, 6], (1, 512, 512, 1))
torch.tensor(direction).cuda().float(), torch.tensor(dist).cuda().float()