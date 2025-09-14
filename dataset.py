# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class SpectrumDataset(Dataset):
    def __init__(self, npz_file, split='train', train_frac=0.8, val_frac=0.1):
        data = np.load(npz_file)
        X = data['X']  # (N,F,T)
        Y = data['Y']  # (N,4)
        N = X.shape[0]
        # split indices
        train_end = int(N * train_frac)
        val_end = train_end + int(N * val_frac)
        if split == 'train':
            idx = np.arange(0, train_end)
        elif split == 'val':
            idx = np.arange(train_end, val_end)
        else:
            idx = np.arange(val_end, N)
        self.X = X[idx]
        self.Y = Y[idx]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        y = self.Y[i].astype(np.float32)  # for BCEWithLogitsLoss expect float
        # return channel-first tensor for conv2d: (C=1,F,T)
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y)
