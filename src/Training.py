# from twisted.conch.scripts.tkconch import frame
from ucimlrepo import fetch_ucirepo
import ssl

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import tree

from pickle import dump



# giving an SSLCertVerificationError when trying to fetch UCI repo
ssl._create_default_https_context = ssl._create_unverified_context





###################################################################
####################### Data Preprocessing ########################
###################################################################

def processing():

    # fetch dataset
    adult = fetch_ucirepo(id=2)

    features = pd.DataFrame(adult.data.features)
    target = pd.DataFrame(adult.data.targets)

    # drop the education column as it is already represented in the education_num column
    features = features.drop(columns=['education'])
    data = pd.concat([features, target], axis=1)

    print("before drop missing values: ", data.shape)
    # drop missing values
    data = data.dropna()

    print("after drop missing values: ", data.shape)

    before = {'workclass': data['workclass'].unique(),
              'marital-status': data['marital-status'].unique(),
              'occupation': data['occupation'].unique(),
              'relationship': data['relationship'].unique(),
              'race': data['race'].unique(),
              'native-country': data['native-country'].unique(),
              'income': data['income'].unique()
              }

    # Categorical Data Preprocessing

    cate_colname = ['workclass', 'marital-status', 'occupation', 'relationship', 'native-country','race']

    from sklearn.preprocessing import LabelEncoder

    labelEncoder = LabelEncoder()

    for i in cate_colname:
        data[i] = labelEncoder.fit_transform(data[i])

    data.head()

    after = {'workclass': data['workclass'].unique(),
              'marital-status': data['marital-status'].unique(),
              'occupation': data['occupation'].unique(),
              'relationship': data['relationship'].unique(),
              'race': data['race'].unique(),
              'native-country': data['native-country'].unique(),
              'income': data['income'].unique()
              }

    # check before and after by comparing the unique values
    for i in before.keys():
        print(f"col_name: {i} before: {len(before[i])} after: {len(after[i])}")

    #   Handling binary data
    sex_map = {
        'Male': 0,
        'Female': 1
    }

    income_map = {
        '<=50K': 0,
        '>50K': 1,
        '<=50K.': 0,
        '>50K.': 1
    }

    # replace the values in the column
    data['sex'] = data['sex'].replace(sex_map)
    data['income'] = data['income'].replace(income_map)


    continues_colname = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

    from sklearn.preprocessing import StandardScaler

    #normalize the continues data
    scaler = StandardScaler()
    data[continues_colname] = scaler.fit_transform(data[continues_colname])


    # split the data into features and target
    #split the data into training and testing data
    from sklearn.model_selection import train_test_split

    X = data.drop(columns=['income'])
    y = data['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #store the data into pickle files
    X_train.to_pickle("X_train.pkl")
    X_test.to_pickle("X_test.pkl")
    y_train.to_pickle("y_train.pkl")
    y_test.to_pickle("y_test.pkl")
    X.to_pickle("X.pkl")
    y.to_pickle("y.pkl")


    return X_train, X_test, y_train, y_test

###################################################################
####################### svm Model Training ########################
###################################################################
class svm_():
    def __init__(self, learning_rate, epoch, C_value, X, Y, batch_size=32):
        self.rff = RandomFourierFeatures(gamma=1.0, D=100)
        # initialize the variables
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
        accuracy = accuracy_score(Y, y_pred)

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

            # compute validation loss
            v_loss = self.compute_loss(X_val, y_val, self.weights,
                                       self.sigmoid(np.dot(X_val, self.weights) + self.bias))
            self.validation_loss.append(v_loss)

            # compute validation loss
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
            if pos_predictions >= neg_predictions: 
                forest_pred = 1
            else:
                forest_pred = 0
            y_pred = np.append(arr=y_pred, values=forest_pred)
            
            if i % 3000 == 0:
                print("aggregated predictions for ", i, " samples.")
        
        # turn to col vector
        y_pred = np.c_[y_pred]
        
        return y_pred, class_probabilities
    
    def evaluate(self, X, y):        
        y_true = y.values
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
            
            accuracy = accuracy_score(y_val, y_pred)
            entropy_loss = log_loss(y_val, y_pred_probabilities)
            
            # Append the score to the list
            cv_scores.append(accuracy)
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


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = processing()

    ###################################################################
    ####################### svm Model Training ########################
    ###################################################################
    C = 0.001
    learning_rate = 0.001
    epoch = 100

    my_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_train, Y=y_train)
    # X_train = X_train.to_numpy()
    # y_train = y_train.to_numpy().reshape(-1, 1)

    # train model
    # ensuring y is in the set {-1, 1}
    y_train_preprocessed = 2 * y_train - 1
    print("Training SVM...")
    training_losses, validation_losses = my_svm.train(X_train, y_train_preprocessed)
    print("Training complete.")

    with open("svm.pkl", "wb") as f:
        dump(my_svm, f, protocol=5)

    ###################################################################
    ################ Logistic Regression Model Training ###############
    ###################################################################
    lr_learning_rate = 0.001
    lr_epoch = 1500
    c = 0.001

    lr = LogisticRegression_(c, lr_learning_rate, lr_epoch, X_train, y_train)

    print("Training Logistic Regression Model")
    lr.train()

    with open("LogisticRegression.pkl", "wb") as f:
        dump(lr, f, protocol=5)


    ###################################################################
    ################ Random Forest Model Training #####################
    ###################################################################
    rf = random_forest(forest_size=100, max_tree_depth=7)
    print("Training Random Forest Classifier...")
    rf.fit(X_train.values, y_train)
    print("Training ended")

    with open("random_forest.pkl", "wb") as f:
        dump(rf, f, protocol=5)




