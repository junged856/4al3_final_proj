from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import tree
        

class random_forest():
    def __init__(self, forest_size, max_tree_depth):
        self.forest_size = forest_size
        self.forest = None
        self.max_tree_depth = max_tree_depth
        
    # create forest
    def fit(self, X, Y):
        print("fitting model...")
        
        sample_size = len(X) 
        self.forest = [None] * self.forest_size
        
        for t_i in range(self.forest_size):
            # bootstrap sample
            X_sample, Y_sample = self.sampling_strat(X, Y, sample_size)
            
            clf = tree.DecisionTreeClassifier(max_depth=self.max_tree_depth)
            clf.fit(X_sample, Y_sample)
            self.forest[t_i] = clf
        return
    
    # sampling with replacement
    def sampling_strat(self, X, Y, n_samples):
        X_sample, Y_sample = resample(X,Y,replace=True,n_samples=n_samples,random_state=42)
        return X_sample, Y_sample

    def predict(self, X):
        y_pred = np.array([])
        
        class_probabilities = self.predict_proba(X)
        
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
            
            if i % 3000 == 0:
                print("aggregated predictions for ", i, " samples.")
        
        # turn to col vector
        y_pred = np.c_[y_pred]
        
        return y_pred, class_probabilities
    
    def evaluate(self, X, y):
        
        class_map = {
            1: 1,
            -1: 0
        }
        y_true = [class_map[x] for x in y.values]
        X_npy = X.values

        print("Performing KFold Cross Validation on Random Forest Classifier...")

        K = 5

        # Initialize KFold object
        kf = KFold(n_splits=K, shuffle=True, random_state=42)

        # Store cross-validation scores
        cv_scores = []
        entropy_losses = []

        # Iterate through the K folds
        for train_index, val_index in kf.split(X_npy):
            X_train, X_val = X_npy[train_index], X_npy[val_index]
            y_train, y_val = y_true[train_index], y_true[val_index]
                        
            # Train the model
            self.fit(X_train, y_train)
            
            # Predict on the validation set
            y_pred, y_pred_probabilities = self.predict(X_val)
            
            accuracy_score = accuracy_score(y_val, y_pred)
            entropy_loss = entropy_loss = log_loss(y_true, y_pred_probabilities)
            
            # Append the score to the list
            cv_scores.append(accuracy_score)
            entropy_losses.append(entropy_loss)

        # Compute the average score across all folds
        average_score = np.mean(cv_scores)
        average_loss = np.mean(entropy_losses)

        print(f'Average accuracy score across {K} folds: {average_score:.4f}')

        print(f"Average entropy loss across {K} folds: {average_loss}")

        return 
        
    def predict_proba(self, X):
        # Get probabilities from all trees
        tree_probs = [tree.predict_proba(X) for tree in self.forest]

        # Average probabilities of likelihood to belong to class 0 (income below 50k)
        avg_probs = []
        for i in range(len(X)):
            sum = 0
            for probs in tree_probs:
                sum += probs[i][0] 
            # Normalize
            avg_probs.append([sum / len(self.forest), 1 - sum / len(self.forest)])

        return avg_probs
    
    
    
    
