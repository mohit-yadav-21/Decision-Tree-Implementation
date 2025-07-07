from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size, "Size of True values and the predicted values must be the same"
    assert isinstance(y_hat, pd.Series), "y_hat must be a Pandas Series"
    assert isinstance(y, pd.Series), "y must be Pandas Series"
    assert y_hat.size>0, "Predicted values (input) must be no empty"
    
    correct_prediction=(y_hat==y).sum()
    acc=correct_prediction/y.size

    return acc

    pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size, "Size of True values and the predicted values must be the same"
    assert isinstance(y_hat, pd.Series), "y_hat must be a Pandas Series"
    assert isinstance(y, pd.Series), "y must be Pandas Series"
    assert y_hat.size>0, "Predicted values (input) must be no empty"
    assert cls in y.unique(), f"Class {cls} not found in true values"
    assert cls in y_hat.unique(), f"Class {cls} not found in predicted values"

    TP=((y_hat==cls)&(y==cls)).sum()
    FP=((y_hat==cls)&(y!=cls)).sum()

    if TP+FP==0:
       return 0.0
    
    Precision=TP/(TP+FP)

    return Precision

    pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size, "Size of True values and the predicted values must be the same"
    assert isinstance(y_hat, pd.Series), "y_hat must be a Pandas Series"
    assert isinstance(y, pd.Series), "y must be Pandas Series"
    assert y_hat.size>0, "Predicted values (input) must be no empty"
    assert cls in y.unique(), f"Class {cls} not found in true values"
    assert cls in y_hat.unique(), f"Class {cls} not found in predicted values"

    TP=((y_hat==cls)&(y==cls)).sum()
    FN=((y_hat!=cls)&(y==cls)).sum()

    if TP+FN==0:
       return 0.0
    
    Recall=TP/(TP+FN)

    return Recall
    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size, "Size of True values and the predicted values must be the same"
    assert isinstance(y_hat, pd.Series), "y_hat must be a Pandas Series"
    assert isinstance(y, pd.Series), "y must be Pandas Series"
    assert y_hat.size>0, "Predicted values (input) must be no empty"

    RMSE=(((y_hat-y)**2).mean())**0.5
    print(RMSE)

    pass


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size, "Size of True values and the predicted values must be the same"
    assert isinstance(y_hat, pd.Series), "y_hat must be a Pandas Series"
    assert isinstance(y, pd.Series), "y must be Pandas Series"
    assert y_hat.size>0, "Predicted values (input) must be no empty"

    MAE=(abs(y_hat-y)).mean()
    print(MAE)

    pass