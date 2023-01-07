from torch.utils.data import Dataset
import torch
import numpy as np
import torch.utils.data

class EvaluateDataset(Dataset):
    def __init__(self, data, rank):
        self.data = data
        self.rank = rank
        self.type = type
        self.generate_data()

    def generate_data(self):
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.rank = torch.tensor(self.rank, dtype=torch.float32)
        dataset = []
        for i in range(0, len(self.rank)):
            dataset.append({'data':self.data[i],'label':self.rank[i]})
        self.dataset=dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
