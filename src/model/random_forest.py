from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

class random_forest():
    def __init__(self, forest_size):
        self.forest_size = forest_size
        self.forest = None
        
    # create forest
    def fit(self, X, Y):
        print("fitting model...")
        
        sample_size = len(X) 
        self.forest = [None] * self.forest_size
        
        for t_i in range(self.forest_size):
            # bootstrap sample
            X_sample, Y_sample = self.sampling_strat(X, Y, sample_size)
                
            clf = DecisionTreeClassifier()
            clf.fit(X_sample, Y_sample)
            self.forest[t_i] = clf
        return
    
    # sampling with replacement
    def sampling_strat(self, X, Y, n_samples):
        X_sample, Y_sample = resample(X,Y,replace=True,n_samples=n_samples,random_state=42)
        return X_sample, Y_sample

    def predict(self, X):
        y_pred = np.array([])
        class_probabilities = []
        
        for i in range(len(X)):
            pos_predictions = 0
            neg_predictions = 0
            
            for tree in self.forest:
                t_pred = tree.predict([X[i]])
                t_pred = t_pred[0]
                
                if t_pred == 1: 
                    pos_predictions += 1 
                else: 
                    neg_predictions += 1
            
            # aggregate class predictions
            forest_pred = 0
            if pos_predictions >= neg_predictions: # TO-DO: what happens when values are equal
                forest_pred = 1
            else:
                forest_pred = -1
            y_pred = np.append(arr=y_pred, values=forest_pred)
            class_probabilities.append([pos_predictions / 100, neg_predictions / 100])
            
            if i % 1000 == 0:
                print("aggregated predictions for ", i, " samples.")
        
        # turn to col vector
        y_pred = np.c_[y_pred]
        
        return y_pred, class_probabilities
    
    def evaluate(self, Y, Y_pred, y_pred_prob):
        accuracy = accuracy_score(Y, Y_pred)
        
        class_map = {
            1: 0,
            -1: 1
        }
        y_true = [class_map[x] for x in Y]
        entropy_loss = log_loss(y_true, y_pred_prob)
        
        return accuracy, entropy_loss