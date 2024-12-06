from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

class svm_():
    def __init__(self,learning_rate,epoch,C_value,X,Y):

        #initialize the variables
        self.input = X
        self.target = Y
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.C = C_value

        self.threshold = 1e-4
     
        self.weights = np.zeros(X.shape[1])
    
    # the function return gradient for 1 instance
    def compute_gradient(self,X,Y):
        X_ = np.array([X]).flatten()

        hinge_distance = 1 - (Y* np.dot(X_,self.weights))
        total_distance = np.zeros(len(self.weights))
        if hinge_distance[0] > 0:
            total_distance += self.weights - (self.C * Y * X_)
        else:
            total_distance += self.weights

        return total_distance

    def compute_loss(self,X,Y):
        # calculate hinge loss
        loss=0

        for i in range(len(X)):
            loss += max(0, 1 - Y[i] * np.dot(self.weights,X[i]))

        regularization_term = 0.5 * np.linalg.norm(self.weights) ** 2
        total_loss = (self.C * loss) + regularization_term 
        return total_loss.item()
    
    def mini_batch_gradient_descent(self,X,Y,batch_size):
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
        train_losses = []
        validation_losses = []
        early_stop = 0
        stopped = False

        for epoch in range(self.epoch):
            features, output = shuffle(X_train, Y_train, random_state=42)
            features_val, output_val = shuffle(X_val, Y_val, random_state=42)

            # mini-batches
            for i in range(0, len(features), batch_size):
                batch_X = features[i:i + batch_size]
                batch_Y = output[i:i + batch_size]

                gradients = np.zeros_like(self.weights)
                for j, feature in enumerate(batch_X):
                    gradients += self.compute_gradient(feature, batch_Y[j])

                gradients /= batch_size

                self.weights -= self.learning_rate * gradients

            train_loss = np.mean(self.compute_loss(features, output))
            validation_loss = np.mean(self.compute_loss(features_val, output_val))

            if epoch % (self.epoch // 10) == 0:
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)
                print(f"Epoch: {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}")


            if (len(validation_losses) > 1 and abs(validation_loss - validation_losses[-2]) < self.threshold and stopped == False):
                print(f"Stopping early at epoch: {epoch} due to model converging below threshold with loss: {validation_loss}")
                early_stop = epoch   
                stopped = True

        print("Training ended...")
        # print("weights are: {}".format(self.weights))

        return train_losses, validation_losses, early_stop

    def predict(self,X_test,Y_test):
            predicted_values = [np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])]
            
            accuracy= accuracy_score(Y_test, predicted_values)
            print("Accuracy on test dataset: {}".format(accuracy))

            precision = precision_score(Y_test, predicted_values, average='macro', zero_division=1)
            print("Precision on test dataset: {}".format(precision))

            recall = recall_score(Y_test, predicted_values, average='macro', zero_division=1)
            print("Recall on test dataset: {}".format(recall))

            return accuracy, precision, recall