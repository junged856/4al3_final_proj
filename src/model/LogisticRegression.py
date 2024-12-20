import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LogisticRegression_():
    def __init__(self, c, learning_rate, epoch, X, Y):
        self.input = X
        self.target = Y
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.C = c
    
    def set_input(self, new_input):
        self.input = new_input

    def set_target(self, new_target):
        self.target = new_target

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, Y, w, h):
        m = len(X)
        # Binary cross-entropy
        cross_entropy = -(1 / m) * (np.dot(Y.T, np.log(h)) + np.dot((1 - Y).T, np.log(1 - h)))

        # L2 regularization term
        l2_reg = (self.C / (2 * m)) * np.sum(np.square(w))
        return cross_entropy + l2_reg


    def train(self):
        m = len(self.input)

        self.weights = np.zeros(self.input.shape[1])
        self.bias = 0
        self.train_loss = []
        self.validation_loss = []

        X_train, X_val, y_train, y_val = train_test_split(self.input, self.target, test_size=0.2, random_state=42)

        # Gradient Descent
        for epoch in range(self.epochs):
            z = np.dot(X_train, self.weights) + self.bias
            h = self.sigmoid(z)

            # compute train loss
            t_loss = self.compute_loss(X_train, y_train, self.weights, h)
            self.train_loss.append(t_loss)

            #compute validation loss
            v_loss = self.compute_loss(X_val, y_val, self.weights, self.sigmoid(np.dot(X_val, self.weights) + self.bias))
            self.validation_loss.append(v_loss)

            #compute validation loss
            z_val = np.dot(X_train, self.weights) + self.bias

            # Compute gradients
            dw = (1 / m) * np.dot(X_train.T, (h - y_train)) + (self.C / m) * self.weights
            db = (1 / m) * np.sum(h - y_train)

            # Update parameters
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            if epoch % 100 == 0:
                print(f"Epoch: {epoch + 1}: train Loss: {t_loss}, Validation Loss: {v_loss}")

        print("Train end")
        self.plot_losses()

    def evaluate(self, X, Y):
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        accuracy = np.sum(y_pred == Y) / len(Y)
        print(f"Accuracy: {accuracy}")

        return y_pred

    def plot_losses(self):
        plt.plot(range(1, self.epochs + 1), self.train_loss, label="Train Losses") 
        plt.plot(range(1, self.epochs + 1), self.validation_loss, label="Validation Losses") 

        plot_title = "Train loss vs. Validation loss for LogisticRegression"
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(plot_title)
        plt.legend()

        plt.show()
