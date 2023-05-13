import torch
from torch.utils.data import Dataset

class CutSelectionDataset(Dataset):
    def __init__(self, node_features, cut_features, best_cut_indices, cut_values):
        self.node_features = node_features
        self.cut_features = cut_features
        self.best_cut_indices = best_cut_indices
        self.cut_values = cut_values

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        node_feature = self.node_features[idx]
        cut_feature = self.cut_features[idx]
        best_cut_index = self.best_cut_indices[idx]
        cut_value = self.cut_values[idx]

        return node_feature, cut_feature, best_cut_index, cut_value