"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these functions are here to simply help you.
"""
import numpy as np
import pandas as pd

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X, drop_first=True)

def isreal(Y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return Y.dtype.name != 'category'


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    value_counts = Y.value_counts(normalize=True)
    return -sum(value_counts * np.log2(value_counts))
    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    value_counts = Y.value_counts(normalize=True)
    return 1 - sum(value_counts ** 2)
    pass

def variance(x):
    if x.size != 0: return np.var(x)
    else: return 0
    pass

def entropy_info_gain(Y: pd.Series, attr: pd.Series) -> tuple[float,float]:
    """
    Function to calculate the information gain using entropy
    """
    # Creting the dataframe to have the corresponding values corelation
    df = pd.DataFrame({"input": attr, "output": Y})

    match (isreal(df["input"]), isreal(df["output"])):

        case (True, True):  # RIRO
            sorted_df = df.sort_values(by="input").reset_index(drop=True)   # Sort the dataframe according to the input values
            best_split = 0
            max_gain = -np.inf

            for ind in range(1, Y.size):
                split = float(sorted_df.iloc[ind, 0] + sorted_df.iloc[ind - 1, 0]) / 2
                df_right = sorted_df[sorted_df["input"] > split].reset_index(drop=True)
                df_left = sorted_df[sorted_df["input"] <= split].reset_index(drop=True)
                y_val1, y_val2 = df_right["output"], df_left["output"]
                wgt_variance = variance(y_val1) * len(y_val1) + variance(y_val2) * len(y_val2)
                gain = variance(Y) - wgt_variance / len(Y)
                if gain > max_gain:
                    max_gain = gain
                    best_split = split

            return (best_split, max_gain)        

        case (True, False):  # RIDO
            sorted_df = df.sort_values(by="input").reset_index(drop=True)   # Sort the dataframe according to the input values
            best_split = 0       
            max_gain = -np.inf   

            for ind in range(Y.size):    # Iterate over all the values of the input
                if ind == Y.size - 1:
                    continue

                split = float(sorted_df.iloc[ind, 0] + sorted_df.iloc[ind + 1, 0]) / 2   # Calculate the split value, at the mid point of the two consecutive values
                # Split the dataframe into two parts, one with values > split and other with values <= split
                df_right = sorted_df[sorted_df["input"] > split].reset_index(drop=True)   
                df_left = sorted_df[sorted_df["input"] <= split].reset_index(drop=True)
                y_val1, y_val2 = df_right["output"], df_left["output"]    # Get the corresponding output values
                entropy_val = (float(entropy(y_val1) * y_val1.size + entropy(y_val2) * y_val2.size) / Y.size )   # Calculate the weighted entropy
                gain = entropy(Y) - entropy_val    
                if gain > max_gain:
                    max_gain = gain
                    best_split = split

            return (best_split, max_gain)     # Return the best split value and the corresponding gain

        case (False, True):  # DIRO
            unique_attr = df["input"].unique()    # Get the unique attribute values
            wgt_variance = 0             

            for val in unique_attr:                # Iterate over all the unique attribute values
                df_new = df[df["input"] == val]    # Seperate the dataframe according to the attribute value
                wgt_variance += variance(df_new["output"]) * len(df_new["output"])   # Calculate the weighted variance

            gain = variance(Y) - wgt_variance / Y.size 

            return (None, gain)    # Return the (None: as discrete input) and gain
        
        case (False, False):  # DIDO
            base_entropy = entropy(Y)   # Calculate the base entropy
            df_new = pd.crosstab(index=df["output"], columns=df["input"], normalize="columns")   # Create a cross tabulation table, with the relative probability of each ouput value for a given attribute value
            entropy_vals = (df_new.apply(lambda x: -x * np.log2(x + 1e-6), axis=0).sum(axis=0).sort_index(axis=0)) 
            attr_normalized = attr.value_counts(normalize=True).sort_index(axis=0)  # Normalize the attribute values to return the relative probabilities
            gain_value = base_entropy - (entropy_vals * attr_normalized).sum(axis=0)  
            return (None, gain_value)        

        case (_,_):     # Any other case
            print("Error: Invalid input/output type combination.")
            return (None, None)

def gini_info_gain(Y: pd.Series, attr: pd.Series) -> tuple[float, float]:
    """
    Function to calculate the information gain using gini index
    """
    df = pd.DataFrame({"input": attr, "output": Y})

    if isreal(df["output"]):  # In case gini index is used accidentally then we redirect it to entropy info gain.
        return entropy_info_gain(Y, attr)

    if isreal(df["input"]) is False:
        df_new = (df.groupby("input", observed=False).apply(lambda x: gini_index(x)).sort_index(axis=0))
        probability = df["input"].value_counts(normalize=True).sort_index(axis=0)
        weighted_gini = df_new * probability
        gain = gini_index(Y) - weighted_gini.sum()
        return (None, gain)

    else:
        sorted_df = df.sort_values(by="input").reset_index(drop=True)
        best_split = 0
        max_gain = -np.inf

        for ind in range(1, Y.size):
            split = float(sorted_df.iloc[ind, 0] + sorted_df.iloc[ind - 1, 0]) / 2
            df_right = sorted_df[sorted_df["input"] > split].reset_index(drop=True)
            df_left = sorted_df[sorted_df["input"] <= split].reset_index(drop=True)
            y_val1, y_val2 = df_right["output"], df_left["output"]
            weighted_gini = (
                gini_index(y_val1) * len(y_val1) + gini_index(y_val2) * len(y_val2)
            ) / len(Y)
            gain = gini_index(Y) - weighted_gini
            if gain > max_gain:
                max_gain = gain
                best_split = split

        return (best_split, max_gain)

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # Finding the best split attribute based on the features and criterion

    best_attribute = None       # Best attribute to split upon
    best_split_val = None  # in case of real input

    if criterion == "entropy":
        best_gain = -np.inf     # Best gain set to -inf to begin
        for attribute in features:   # Iterate over all the features/attributes, and find the one with the maximum gain
            split_value, gain = entropy_info_gain(y, X[attribute])
            if not(gain <= best_gain):
                best_gain = gain
                best_split_val, best_attribute = split_value, attribute

    else:   # gini index
        best_gain = -np.inf 
        for attribute in features:  
            split_value, gain = gini_info_gain(y, X[attribute])
            if not(gain <= best_gain):
                best_gain = gain
                best_split_val, best_attribute = split_value, attribute

    return (best_split_val, best_attribute)


def real_variance(X: pd.DataFrame, y: pd.Series, value: np.float64 , attribute):
    """
    Function to calculate the weighted variance
    """

    mask = (X[attribute] <= value)
    var_left = np.var(y[mask]) * len(y[mask])
    var_right = np.var(y[~mask]) * len(y[~mask])
    return var_left + var_right


def opt_split_value(X: pd.DataFrame, y: pd.Series, attribute):
    """
    Function to find the optimal split value for a given attribute.

    X: Input features (DataFrame)
    y: Output values (Series)
    attribute: Attribute to split upon
    criterion: Splitting criterion ('information_gain' for discrete output, 'mse' for real output)

    return: Optimal split value
    """

    X = X.sort_values(by=[attribute])
    check_values = [(X[attribute].iloc[i] + X[attribute].iloc[i+1]) / 2 for i in range(X.shape[0]-1)]

    y = y if isreal(y) else y.cat.codes
    min_var = float('inf')
    optimal_value = None

    for value in check_values:
        var = real_variance(X, y, value, attribute)
        if var < min_var:
            min_var = var
            optimal_value = value

    return optimal_value


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value) -> tuple[pd.DataFrame, pd.Series]:
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """
    # Splits the data based on a particular value of a particular attribute.
    prev_name = y.name
    if y.name == None:
        y.name = "#"

    df = X.join(y)    # Join the input and output dataframes
    col_name = y.name   

    df = df[df[attribute] == value].drop(columns=attribute).reset_index(drop=True)   # Filter based on the value and drop the attribute column and reset the index
    y_new = df[col_name]
    X_new = df.drop(columns=col_name)
    y_new.name = prev_name
    return (X_new, y_new)

def split_data_real(X: pd.DataFrame, y: pd.Series, attribute, value) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Function to split the dataframe into two parts, one with values > value and other with values <= value
    Provides both the dataframes and the corresponding output values
    '''
    prev_name = y.name
    if y.name == None:
        y.name = "#"

    df = X.join(y)
    col_name = y.name

    df1 = df[df[attribute] > value].reset_index(drop=True)    
    y_new1 = df1[col_name]
    X_new1 = df1.drop(columns=col_name)

    df2 = df[df[attribute] <= value].reset_index(drop=True)
    y_new2 = df2[col_name]
    X_new2 = df2.drop(columns=col_name)

    y_new1.name = prev_name
    y_new2.name = prev_name

    return (X_new1, X_new2, y_new1, y_new2) 