import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

# This is important to exploit the GPU if it is available
use_cuda = torch.cuda.is_available()

# Seed for the random number generator
torch.manual_seed(1)

device = torch.device("cuda" if use_cuda else "cpu")


class NetworkTrafficDataset(Dataset):
    """
    This class is designed to load the Bot-IoT Dataset, by selecting the 10 most important features
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.df = self.df[['attack','seq', 'stddev', 'N_IN_Conn_P_SrcIP','min', 'state_number', 'mean', 'N_IN_Conn_P_DstIP', 'drate', 'srate', 'max']] # This must be more dynamic, like using some parameters
        self.label = self.df.iloc[:,-1]
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.df.iloc[idx, 1:]
        data = np.array(data)
        label = self.df.iloc[idx, 0]
        label = np.array([label])

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
        
        return data, label

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # print(sample.shape)
        x = torch.from_numpy(sample)
        x = x.to(device).float()
        # print(x.size())
        return x

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        x_max = torch.max(sample)
        x_min = torch.min(sample)
        return (sample - (x_max + x_min)/2)/((x_max + x_min)/2)

class ToTensorLong(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # print(sample.shape)
        x = torch.from_numpy(sample)
        x = x.long().to(device)
        print(x)
        return x