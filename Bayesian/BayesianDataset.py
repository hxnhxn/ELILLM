import torch
import math
import numpy as np
from torch.utils.data import Dataset


# 定义 BayesianDataset 数据集类
class BayesianDataset(Dataset):

    def __init__(self, smiles=None, embeddings=None, scores=None):
        super(BayesianDataset, self).__init__()
        if embeddings is None:
            self.embeddings = []
        else:
            self.embeddings = embeddings
        if scores is None:
            self.scores = []
        else:
            self.scores = scores
        if smiles is None:
            self.smiles = []
        else:
            self.smiles = smiles

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.scores[idx]

    def sort(self):
        sort_indices = np.argsort(self.scores)[::-1]
        sorted_embeddings = [self.embeddings[i] for i in sort_indices]
        sorted_scores = [self.scores[i] for i in sort_indices]
        sorted_smiles = [self.smiles[i] for i in sort_indices]
        self.embeddings = sorted_embeddings
        self.scores = sorted_scores
        self.smiles = sorted_smiles

    def add_item(self, smiles, embedding, score):
        self.smiles.append(smiles)
        self.embeddings.append(embedding)
        self.scores.append(score)


    def add_items(self, smiles, embeddings, scores):
        self.smiles.extend(smiles)
        self.embeddings.extend(embeddings)
        self.scores.extend(scores)

    def maxy(self):
        if len(self.scores) == 0:
            return -1000.0
        else:
            return max(self.scores)

class ClassificationDataset(Dataset):
    def __init__(self, embeddings=None, labels=None):
        super(ClassificationDataset, self).__init__()
        self.embeddings = embeddings
        self.labels = labels
        if embeddings is None:
            self.embeddings = []
        if labels is None:
            self.labels = []
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    def add_item(self, embedding, label):
        self.embeddings.append(embedding)
        self.labels.append(label)
    def add_items(self, embeddings, labels):
        self.embeddings.extend(embeddings)
        self.labels.extend(labels)

class PositionEmbedder:
    def __init__(self, max_len=2048, d_model=3072):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加 batch_size 维度
        self.pe = pe

    def get_position_embedding(self, embeddings):
        return self.pe[:,:embeddings.shape[1]]
