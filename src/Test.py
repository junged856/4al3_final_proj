from pickle import load
import numpy as np
import pandas as pd
from Training import svm_,LogisticRegression_,RandomFourierFeatures,random_forest


# Load the model

with open("svm.pkl", "rb") as f1:
    my_svm = load(f1)

with open("LogisticRegression.pkl", "rb") as f2:
    lr = load(f2)

with open("random_forest.pkl", "rb") as f3:
    rf = load(f3)

# Load the data
X_train = pd.read_pickle("X_train.pkl")
y_train = pd.read_pickle("y_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_test = pd.read_pickle("y_test.pkl")



#######################################################
################### Evaluating SVM #####################
#######################################################

# testing the model
print("Evaluating SVM...")
y_test_preprocessed = 2 * y_test -1
y_pred = my_svm.evaluate(X_test,y_test_preprocessed)


#######################################################
########### Evaluating Logistic Regression ############
#######################################################
print("Evaluating LogisticRegression ...")
y_pred = lr.evaluate(X_test, y_test)




#######################################################
############## Bias Fairness Strategy #################
#######################################################
from sklearn.metrics import confusion_matrix

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
y_pred = pd.Series(y_pred).reset_index(drop=True)

male_mask = X_test['sex'] == 0
female_mask = X_test['sex'] == 1

y_test_male = y_test[male_mask]
y_pred_male = y_pred[male_mask]

cm = confusion_matrix(y_test_male, y_pred_male)

tn, fp, fn, tp = cm.ravel()

male_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
male_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
print("Male TPR:", male_tpr)
print("Male FPR:", male_fpr)


def measure_race_bias(y_pred, y_test, X_test):
    unique_races = X_test['race'].unique()

    race_map = {
        1: 'White',
        0: 'Black',
        4: 'Amer-Indian-Eskimo',
        2: 'Asian-Pac-Islander',
        3: 'Other'
    }

    for race in unique_races:
        race_mask = X_test['race'] == race

        y_test_race = y_test[race_mask]
        y_pred_race = y_pred[race_mask]

        cm = confusion_matrix(y_test_race, y_pred_race)

        tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Race {race_map[race]}: TPR = {tpr:.2f}")


measure_race_bias(y_pred, y_test, X_test)

# Balance the data and then refeed the new mitigated X and Y into the logisitc regression model to retrain
from sklearn.utils import resample

def balance_data(X_train, y_train, sensitive_column):
    train_data = pd.concat([X_train, y_train], axis=1)

    balanced_data = []
    for group in train_data[sensitive_column].unique():
        group_data = train_data[train_data[sensitive_column] == group]
        class_0 = group_data[group_data['income'] == 0]
        class_1 = group_data[group_data['income'] == 1]

        min_size = min(len(class_0), len(class_1))
        balanced_group = pd.concat([
            resample(class_0, replace=True, n_samples=min_size, random_state=42),
            resample(class_1, replace=True, n_samples=min_size, random_state=42)
        ])
        balanced_data.append(balanced_group)

    balanced_data = pd.concat(balanced_data)
    return balanced_data.drop('income', axis=1), balanced_data['income']

X_train_mitigated, y_train_mitigated = balance_data(X_train, y_train, 'race')
lr.set_input(X_train_mitigated)
lr.set_target(y_train_mitigated)

lr.train()

# Then gather the new predicted y values and remeasure the Bias
y_pred_mitigated = lr.evaluate(X_test, y_test)
measure_race_bias(y_pred_mitigated, y_test, X_test)


#######################################################
############### Evaluating Random forest ##############
#######################################################

# Load the data, train-test splits are done in the rf.evaluate method
X = pd.read_pickle("X.pkl")
y = pd.read_pickle("y.pkl")

rf.evaluate(X, y) # performs k-fold cross validation + prints out the results to console