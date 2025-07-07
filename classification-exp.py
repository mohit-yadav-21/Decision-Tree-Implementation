import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Write the code for Q2 a) and b) below. Show your results.

# Splitting
data_length = X.shape[0]
X_train = pd.DataFrame(X[: int(0.7 * data_length)])
y_train = pd.Series(y[: int(0.7 * data_length)])
X_test = pd.DataFrame(X[int(0.7 * data_length) :])
y_test = pd.Series(y[int(0.7 * data_length) :])

# 2a

print("\n..........2a...........\n")

# List of criteria to evaluate
given_criterions = ["information_gain", "gini_index"]

# Loop through each criterion
for criterion in given_criterions:
    print("Criterion:", criterion)

    decision_tree = DecisionTree(criterion=criterion)
    decision_tree.fit(X_train, y_train)

    prediction = decision_tree.predict(X_test)
    print(" |----Test Accuracy:", accuracy(prediction, y_test))
    
    # Print precision and recall for each class
    for cls in y_test.unique():
        print("     |----Precision for class", cls, ":", precision(prediction, y_test, cls))
        print("     |----Recall for class", cls, ":", "{:.2f}".format(recall(prediction, y_test, cls)))

print("\n..........2b...........\n")

#2b

k = 5
fold_size = data_length // k
depths = [5]
max_depth = 10

for iteration in range(k):
    print("Fold:", iteration)

    # Define the boundaries of the validation fold
    test_start = iteration * fold_size
    test_end = iteration * fold_size + fold_size

    # Create validation and training splits
    X_test = pd.DataFrame(X[test_start : test_end])
    y_test = pd.Series(y[test_start : test_end])
    X_train = pd.DataFrame(np.concatenate([X[ : test_start], X[test_end : ]]))
    y_train = pd.Series(np.concatenate([y[ : test_start], y[test_end : ]]))

    # Train and evaluate a new model on each split
    for criterion in given_criterions:
        print("    |----Criterion:", criterion)
        for depth in depths:
            print("        |----Depth:", depth)
            decision_tree = DecisionTree(criterion=criterion, max_depth=depth)
            decision_tree.fit(X_train, y_train)
            prediction = decision_tree.predict(X_test)
            print("             |----Accuracy: ", accuracy(prediction, y_test))

            # Calculate precision and recall for each class
            for cls in y_test.unique():
                print("             |----Precision for class", cls, ":", precision(prediction, y_test, cls))
                print("             |----Recall for class", cls, ":", "{:.2f}".format(recall(prediction, y_test, cls)))

# Nested cross-validation:
outFolds = 5
inFolds = 4
outer_fold_size = data_length // outFolds

hyperparameters = {}
hyperparameters["criterion"] = given_criterions
hyperparameters["max_depth"] = range(1, max_depth + 1)

outer_df = pd.DataFrame(columns=["Outer_fold", "Depth", "Criterion", "Accuracy"])
from itertools import product
for outer in range(outFolds):

    # Define the boundaries for the outer validation set
    test_start  = outer * outer_fold_size
    test_end    = outer * outer_fold_size + outer_fold_size

    # Create outer validation and training splits
    X_test = pd.DataFrame(X[test_start : test_end]).reset_index(drop=True)
    y_test = pd.Series(y[test_start : test_end]).reset_index(drop=True)
    X_outer_train = pd.DataFrame(np.concatenate([X[ : test_start], X[test_end : ]])).reset_index(drop=True) 
    y_outer_train = pd.Series(np.concatenate([y[ : test_start], y[test_end : ]])).reset_index(drop=True)

    inner_fold_size = len(X_outer_train) // inFolds
    # Store the validation accuracy for each hyperparameter combination
    df = pd.DataFrame(columns=["Depth", "Criterion", "Accuracy"])

    for inner in range(inFolds):
        for depth, criterion in product(hyperparameters["max_depth"], hyperparameters["criterion"]):
            # Define the boundaries for the inner validation set
            validation_start = inner * inner_fold_size
            validation_end   = (inner + 1) * inner_fold_size

            # Create inner validation and training splits
            X_validation  = X_outer_train[validation_start : validation_end].reset_index(drop=True)
            y_validation  = y_outer_train[validation_start : validation_end].reset_index(drop=True)
            X_inner_train = pd.DataFrame(pd.concat([X_outer_train[ : validation_start], X_outer_train[validation_end : ]])).reset_index(drop=True)
            y_inner_train = pd.Series(pd.concat([y_outer_train[ : validation_start], y_outer_train[validation_end : ]])).reset_index(drop=True)

            # Train and evaluate the model on the inner validation set
            decision_tree = DecisionTree(criterion=criterion, max_depth=depth)
            decision_tree.fit(X_inner_train, y_inner_train)
            prediction = decision_tree.predict(X_validation)

            # Record the accuracy for each hyperparameter combination
            df.loc[len(df)] = [depth, criterion, accuracy(prediction, y_validation)]
                
    # Determine the best hyperparameter combination
    df = df.groupby(["Depth", "Criterion"]).mean()['Accuracy']
    opt_parameters = df.idxmax()

    # Train the model on the outer training set using the optimal hyperparameters and evaluate on the outer validation set
    decision_tree = DecisionTree(criterion=opt_parameters[1], max_depth=opt_parameters[0])
    decision_tree.fit(X_outer_train, y_outer_train)
    y_hat = decision_tree.predict(X_test)

    # Record the accuracy for the best hyperparameters
    outer_df.loc[len(outer_df)] = [outer, opt_parameters[0], opt_parameters[1], accuracy(y_hat, y_test)]

print(outer_df)
print("Mean accuracy:", outer_df["Accuracy"].mean())
print(" ")