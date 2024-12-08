from sklearn.metrics import accuracy_score, recall_score, precision_score
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
            
            if i % 1000 == 0:
                print("aggregated predictions for ", i, " samples.")
        
        # turn to col vector
        y_pred = np.c_[y_pred]
        
        return y_pred
    
    def evaluate(self, Y, Y_pred):
        accuracy = accuracy_score(Y, Y_pred)
        return accuracy