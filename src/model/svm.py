from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class svm_():
    def __init__(self,learning_rate,epoch,C_value,X,Y, batch_size=32):
        self.rff = RandomFourierFeatures(gamma=1.0, D=100)
        #initialize the variables
        self.input = X
        self.target = Y
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.C = C_value
        self.bias = 0
        self.batch_size = batch_size

        self.threshold = 1e-4
     
        self.weights = np.zeros(X.shape[1])
    
    def _compute_loss(self, X, y):
        hinge_loss = np.maximum(0, 1 - y * (np.dot(X, self.weights) + self.bias))
        return 0.5 * np.dot(self.weights, self.weights) + self.C * np.sum(hinge_loss)
    
    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.rff.fit(X_train)

        X_train_transformed = self.rff.transform(X_train)
        X_val_transformed = self.rff.transform(X_val)

        # Initialize weights
        n_samples, n_features = X_train_transformed.shape
        self.weights = np.zeros(n_features)

        # Training loop
        train_losses = []
        val_losses = []
        # stopped = False

        for epoch in range(self.epochs):
            X_train_transformed, y_train = shuffle(X_train_transformed, y_train, random_state=42)

            # mini-batch GD
            for i in range(0, len(X_train_transformed), self.batch_size):
                batch_X = X_train_transformed[i:i + self.batch_size]
                batch_y = y_train[i:i + self.batch_size]

                # Compute margin
                margin = batch_y * (np.dot(batch_X, self.weights) + self.bias)

                mask = margin < 1

                # samples incorrectly classified
                violating_X = batch_X[mask]
                violating_y = batch_y[mask]

                if len(violating_X) > 0:
                    grad_w = self.weights - self.C * np.dot(violating_X.T, violating_y)
                    grad_b = -self.C * np.sum(violating_y)
                else:
                    grad_w = self.weights
                    grad_b = 0

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            train_loss = self._compute_loss(X_train_transformed, y_train)
            val_loss = self._compute_loss(X_val_transformed, y_val)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if epoch % (self.epochs // 10) == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        self.plot_losses(train_losses, val_losses)
        return train_losses, val_losses

    def evaluate(self, X, Y):
        X_transformed = self.rff.transform(X)
        y_pred = np.sign(np.dot(X_transformed, self.weights) + self.bias)
        accuracy= accuracy_score(Y, y_pred)

        print("Accuracy on test dataset: {}".format(accuracy))
        return y_pred
    
    def plot_losses(self, train_losses, val_losses):
        plt.plot(range(1, self.epochs + 1), train_losses, label="Train Losses") 
        plt.plot(range(1, self.epochs + 1), val_losses, label="Validation Losses") 

        plot_title = "Train loss vs. Validation loss for SVM"
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(plot_title)
        plt.legend()

        plt.show()

class RandomFourierFeatures:
    def __init__(self, gamma=1.0, D=100):
        self.gamma = gamma
        self.D = D
        self.W = None
        self.b = 0.0

    def fit(self, X):
        n_features = X.shape[1]
        self.W = np.sqrt(2 * self.gamma) * np.random.randn(n_features, self.D)
        self.b = np.random.uniform(0, 2 * np.pi, self.D)

    def transform(self, X):
        projection = np.dot(X, self.W) + self.b
        return np.sqrt(2 / self.D) * np.cos(projection)