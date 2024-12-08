import pandas as pd
import numpy as np
import sklearn

class LogisticRegression_():
    def __init__(self, c, learning_rate, epoch, X, Y):
        self.input = X
        self.target = Y
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.C = c

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, Y, w, h):
        m = len(X)
        # Binary cross-entropy
        cross_entropy = -(1 / m) * (np.dot(Y.T, np.log(h)) + np.dot((1 - Y).T, np.log(1 - h)))

        # L2 regularization term
        l2_reg = (self.C / (2 * m)) * np.sum(np.square(w))
        return cross_entropy + l2_reg


    def train(self,x_test,y_test):
        m = len(self.input)

        self.weights = np.zeros(self.input.shape[1])
        self.bias = 0
        self.train_loss = []
        self.validation_loss = []

        # Gradient Descent
        for epoch in range(self.epochs):
            z = np.dot(self.input, self.weights) + self.bias
            h = self.sigmoid(z)

            # compute train loss
            t_loss = self.compute_loss(self.input, self.target, self.weights, h)
            self.train_loss.append(t_loss)

            #compute validation loss
            v_loss = self.compute_loss(x_test, y_test, self.weights, self.sigmoid(np.dot(x_test, self.weights) + self.bias))
            self.validation_loss.append(v_loss)

            #compute validation loss
            z_val = np.dot(self.input, self.weights) + self.bias

            # Compute gradients
            dw = (1 / m) * np.dot(self.input.T, (h - self.target)) + (self.C / m) * self.weights
            db = (1 / m) * np.sum(h - self.target)

            # Update parameters
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            if epoch % 100 == 0:
                print(f"Epoch: {epoch + 1}: train Loss: {t_loss} validation Loss: {v_loss}")

        print("Train end")

    def evaluate(self, X, Y):
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        accuracy = np.sum(y_pred == Y) / len(Y)
        print(f"Accuracy: {accuracy}")








