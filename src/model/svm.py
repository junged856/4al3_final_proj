from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

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

    def compute_loss(self,X,Y):
        # hinge loss
        loss=0

        for i in range(len(X)):
            loss += max(0, 1 - Y[i] * np.dot(self.weights,X[i]))

        regularization_term = 0.5 * np.linalg.norm(self.weights) ** 2
        total_loss = (self.C * loss) + regularization_term 
        return total_loss.item()
    
    def _compute_loss(self, X, y):
        """
        Compute the hinge loss + regularization term.
        """
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

            # if (len(val_losses) > 1 and abs(val_loss - val_losses[-2]) < self.threshold and stopped == False):
            #     print(f"Stopping early at epoch: {epoch} due to model converging below threshold with loss: {val_loss}")
            #     stopped = True

            if epoch % (self.epochs // 10) == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        return train_losses, val_losses

    def evaluate(self, X, Y):
        """
        Predict labels for input data.
        Parameters:
        - X: Input data of shape (n_samples, n_features).
        Returns:
        - Predicted labels of shape (n_samples,).
        """
        X_transformed = self.rff.transform(X)
        y_pred = np.sign(np.dot(X_transformed, self.weights) + self.bias)
        accuracy= accuracy_score(Y, y_pred)
        print("Accuracy on test dataset: {}".format(accuracy))
        
    # def predict(self,X_test,Y_test):
    #         predicted_values = [np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])]
            
    #         accuracy= accuracy_score(Y_test, predicted_values)
    #         print("Accuracy on test dataset: {}".format(accuracy))

    #         precision = precision_score(Y_test, predicted_values, average='macro', zero_division=1)
    #         print("Precision on test dataset: {}".format(precision))

    #         recall = recall_score(Y_test, predicted_values, average='macro', zero_division=1)
    #         print("Recall on test dataset: {}".format(recall))

    #         return accuracy, precision, recall

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