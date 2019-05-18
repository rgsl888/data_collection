from torch.utils.data import Dataset
import pickle as pkl
from PIL import Image
import os
import numpy as np
import torch
import cv2 as cv
import random


class TrainDataset(Dataset):
    def __init__(self, data_path, csv_path, transform=None):
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.data = pkl.load(open(csv_path, "rb"))
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def get_weights(self):
        w = [0]*7
        weights = []
        s = 0
        for i in self.data:
            if i[1] == 0:
                w[0] += 1
            elif i[1] == 1:
                w[1] += 1
            elif i[1] == 2:
                w[2] += 1
            elif i[1] == 3:
                w[3] += 1
            elif i[1] == 4:
                w[4] += 1
            elif i[1] == 5:
                w[5] += 1
            elif i[1] == 6:
                w[6] += 1

        for i in range(7):
            num = w[i]
            s += 1/num
            weights.append(1/num)

        for i in range(7):
            weights[i] = weights[i]/s

        return weights

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_path,self.data[idx][0]))
        label = self.data[idx][1]
        if self.transform:
            image = self.transform(image)

        return image, label

class ValDataset(Dataset):
    def __init__(self, data_path, csv_path, transform=None):
        super(ValDataset, self).__init__()
        self.data = pkl.load(open(csv_path, "rb"))
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_path, self.data[idx][0]))
        label = self.data[idx][1]
        if self.transform:
            image = self.transform(image)

        return image, label



if __name__ == "__main__":
    dataset = TrainDataset("/home/palparmar/research/fer/data/manual_cropped",
    "/home/palparmar/research/fer/data/manual_cropped.csv")

    image, label = dataset[0]

    print("Expression is %d with valence and arousal are %0.2f and %0.2f"%(label, valence, arousal))
