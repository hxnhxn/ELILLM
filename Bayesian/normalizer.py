from Bayesian.BayesianDataset import BayesianDataset
import numpy as np

class Normalizer:

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def update(self, mean, std):
        self.mean = mean
        self.std = std

    def normalize(self, dataset:BayesianDataset):
        dataset.scores = [(i - self.mean) / self.std for i in dataset.scores]


    def denormalize(self, dataset:BayesianDataset):
        dataset.scores  = [i*self.std + self.mean for i in dataset.scores]


    def normalize_one(self, y):
        if self.std is None:
            return y
        return (y-self.mean)/self.std


    def denormalize_one(self, y):
        if self.std is None:
            return y
        return y*self.std + self.mean

