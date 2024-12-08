import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, Grayscale, Resize, RandomCrop, ToTensor


class LogisticRegression_():
    def __init__(self,c,learning_rate,epoch,X,Y):
        self.input = X
        self.target = Y
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.C = c

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self,X,Y,w,h):
        return -(1/len(X)) * ((np.dot(Y.T,np.log(h)) + np.dot((1-Y).T,np.log(1-h)))) + (self.C * np.sum(w))

    def train(self):
        m = len(self.input)

        self.weights = np.zeros(self.input.shape[1])
        self.bias = 0

        # Gradient Descent
        for epoch in range(self.epochs):
            z = np.dot(self.input,self.weights) + self.bias
            h = self.sigmoid(z)

            loss = self.compute_loss(self.input,self.target,self.weights,h)

            self.weights = self.weights - (self.learning_rate / m) * (np.dot(self.input.T,(h - self.target)))
            self.bias = self.bias - (self.learning_rate / m) * np.sum(h - self.target)

            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}: Loss {loss}")

        print("Train end")

    def evaluate(self,X,Y):
        y_pred = self.sigmoid(np.dot(X,self.weights))
        y_pred = np.where(y_pred > 0.5, 1, 0)
        accuracy = np.sum(y_pred == Y) / len(Y)
        print(f"Accuracy: {accuracy}")






























