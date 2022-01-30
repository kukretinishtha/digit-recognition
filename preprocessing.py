import pandas as pd
import numpy as np
import os
import errno
import keras
from sklearn.model_selection import train_test_split

def read_dataset(dataset_filepath):
    if os.path.exists(dataset_filepath):
        dataset = pd.read_csv(dataset_filepath)
        return dataset
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_filepath)

def splitting_dataset_into_testing_and_validation(dataframe, label, number_of_features):
    dataframe_features = dataframe.iloc[:, label+1:number_of_features]
    dataframe_label = dataframe.iloc[:, label]
    X_train, X_validation, y_train, y_validation = train_test_split(dataframe_features, dataframe_label, test_size = 0.2, random_state = 1212)
    return X_train, X_validation, y_train, y_validation

def reshape_matrix(dataframe, number_of_rows, number_of_features):
    dataframe = dataframe.values.reshape(number_of_rows, number_of_features)
    return dataframe