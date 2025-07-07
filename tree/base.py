"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class TreeNode:
    def __init__(self, output=None, feature=None, threshold=None, depth=None):
        self.output = output         # The value stored at this node (e.g., average for continuous, mode for categorical)
        self.feature = feature       # The feature used to split at this node
        self.children = {}           # A dictionary of child nodes; keys are feature values (for categorical) or "Yes"/"No" (for continuous)
        self.threshold = threshold   # Threshold for splitting a continuous feature; None for categorical
        self.depth = depth           # The depth (level) of this node in the tree

    def is_terminal(self):
        # Returns True if the node is a terminal/leaf node (i.e., no further splits)
        return self.feature is None   

    def predict(self, data: pd.Series, max_depth=np.inf):
        # Predicts the output by traversing the tree based on the given data point
        if self.is_terminal() or self.depth >= max_depth:  # If this is a leaf node or we've reached the maximum depth, return the output
            return self.output

        else:
            if self.threshold is None:  # If the node splits on a categorical feature
                if data[self.feature] in self.children:
                    # If the feature value exists among the children, recurse on the corresponding child node
                    return self.children[data[self.feature]].predict(data, max_depth)

                else:
                    # If the feature value isn't in the child nodes, select one child node randomly
                    random_key = np.random.choice(list(self.children.keys()))

                    try:
                        # Attempt to recurse on the randomly selected child node
                        return self.children[random_key].predict(data, max_depth)
                    except:
                        # If an error occurs, return the current node's output value
                        return self.output

            else:  # If the node splits on a continuous feature
                if data[self.feature] > self.threshold:
                    # If the feature value exceeds the threshold, recurse on the "Yes" child node
                    return self.children["Yes"].predict(data, max_depth)
                else:
                    # If the feature value is less than or equal to the threshold, recurse on the "No" child node
                    return self.children["No"].predict(data, max_depth)



@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None              # Root node of the tree
        self.output_name = None       # Name of the output column
        self.output_type = None       # Type of the output column

    def create_tree(self, X: pd.DataFrame, y: pd.Series, depth=0):
        # Recursive function to create a decision tree
        if len(y.unique()) == 1:
            # If all target values are identical, return a leaf node
            return TreeNode(output=y.iloc[0], depth=depth)

        elif (X.size > 0 and len(X) > 0 and depth <= self.max_depth and np.max(X.nunique()) > 1):
            # Case 1: Maximum depth has not been exceeded
            # Case 2: The features DataFrame is not empty
            # Case 3: There exists at least one feature with more than one unique value

            # Drop features with only one unique value, as they don't contribute to splitting
            for column in X.columns:
                if X[column].nunique() == 1:
                    X = X.drop(columns=[column])

            # Determine the best feature to split on and the optimal split value
            threshold, best_attr = opt_split_attribute(
                X, y, self.criterion, pd.Series(X.columns)
            )

            # Create a new node with the selected feature and split value
            current_node = TreeNode(feature=best_attr, threshold=threshold, depth=depth)

            if threshold is None:  # Categorical feature
                for unique_value in X[best_attr].unique():
                    X1, y1 = split_data(X, y, best_attr, unique_value)
                    current_node.children[unique_value] = self.create_tree(X1, y1, depth=depth + 1)

            else:  # Continuous feature
                X1, X2, y1, y2 = split_data_real(X, y, best_attr, threshold)
                current_node.children["Yes"] = self.create_tree(X1, y1, depth=depth + 1)
                current_node.children["No"] = self.create_tree(X2, y2, depth=depth + 1)

            # Assign the node's output value based on the type of target variable
            if isreal(y):
                current_node.output = y.mean()
            else:
                current_node.output = y.mode()[0]

            return current_node

        else:
            # If further splitting is not possible, return a leaf node with the mean/mode of the target
            if isreal(y):
                return TreeNode(output=y.mean(), depth=depth)
            else:
                return TreeNode(output=y.mode()[0], depth=depth)


 
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 
        self.output_name = y.name
        self.output_type = y.dtype.name
        self.root = self.create_tree(X, y, 0)


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        y_hat = []

        for _, row in X.iterrows():
            predicted_value = self.root.predict(row, self.max_depth)
            y_hat.append(predicted_value)

        # return pd.Series(y_hat, name=self.output_name, dtype=self.output_type)
        return pd.Series(y_hat)

    def plot(self) -> None:
        """
        Method to visualize the decision tree.

        Example Output:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Here, Y stands for Yes and N stands for No.
        """
        def print_node(node, child_name, depth=0, is_last_child=False):
            leaf_prefix = "└───" if is_last_child else "├───"  # Prefix for leaf nodes.
            split_line = "├────" if not is_last_child else "└────"  # Prefix for internal nodes.

            indent = "│    " * (depth - 1) if depth > 0 else ""  # Tree indentation for levels.

            if node.is_terminal():  # Check if the current node is a leaf/terminal.
                val_var = f"{node.output:.2f}" if isinstance(node.output, float) else f"{node.output}"
                # Display the node's value, formatted for float if applicable.
                print(f"{indent}{leaf_prefix * (depth > 0)} {child_name}: " + val_var)

            else:
                split_info = ""  # Variable to hold the split criteria string.
                if node.threshold is None:
                    # Handle discrete attribute splitting.
                    split_info = (
                        f"{indent}{split_line * (depth > 0)}{child_name} -> ?(X[{node.feature}]):")
                else:
                    # Handle continuous attribute splitting.
                    split_var = f"{node.threshold:.2f})" if isinstance(node.threshold, float) else f"{node.threshold})"
                    split_info = (
                        f"{indent}{split_line * (depth > 0)}{child_name} -> ?({node.feature} > " + split_var)

                print(split_info)  # Output the split information.

                # Recursively print child nodes.
                for i, key in enumerate(node.children.keys()):
                    is_last_child = i == len(node.children) - 1
                    print_node(node.children[key], key, depth + 1, is_last_child)

        print("\n<<<<<<<<<<<<<<<- START OF TREE ->>>>>>>>>>>>>>\n")
        print_node(self.root, "root")
        print("\n<<<<<<<<<<<<<<<- END OF TREE ->>>>>>>>>>>>>>\n")

        pass