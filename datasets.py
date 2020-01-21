import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
# Add standardization of the data
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
        # Preprocessing phase
        data_sc = MinMaxScaler() # Normalization to [0,1] range
        self.data = data_sc.fit_transform(self.df.values[:, 1:])
        # print(self.data)
        self.targets = self.df.iloc[:,0]
        # self.targets = data_sc.fit_transform([self.df.values[:,0]])
        # print(self.labels)
        self.transform = transform
        # Maybe we need also the standardization this is due to the fact that my data has input values with differing scales
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        data = self.data[idx]
        data = np.array(data)
        # label = self.targets[0][idx]
        label = self.targets[idx]
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
        # print(sample)
        x = x.to(device).float()
        # print(x.size())
        return x

class Normalize(object):
    """Normalization of the Tensors"""

    def __call__(self, sample):
        # print("Before transposing: " + sample)
        # print("After transposing: " + torch.transpose(sample, 0, 1))
        x_max = torch.max(sample)
        # print(x_max) # So, in this case the maximum is the one of each row, so we have to reshape the matrix before to do this operation
        x_min = torch.min(sample)
        return (sample - (x_max + x_min)/2)/((x_max + x_min)/2)

class ToTensorLong(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # print(sample.shape)
        x = torch.from_numpy(sample)
        x = x.long().to(device)
        return x