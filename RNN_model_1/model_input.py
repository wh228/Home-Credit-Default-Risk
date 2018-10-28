import numpy as np
import pandas as pd
import gc
from data_process import Number_of_RNN
from data_transform import data_transform
from data_transform import Numeric_Column, Categorical_Column
from keras import utils
from scalar import Scalar

Duplication_Rate = 0.1
Duplication_Factor = 8

def duplicate_portion(data):
    app = np.repeat(data[-int(data.shape[0]*Duplication_Rate):], Duplication_Factor, axis=0)
    return np.concatenate((data, app), axis=0)

def get_model_input(train_file, test_file):

    train_numeric_ft, train_categorical_ft, train_target = data_transform(train_file)
    test_numeric_ft, test_categorical_ft, test_target = data_transform(test_file)

    train_numeric = train_numeric_ft.values.reshape(-1, Number_of_RNN, len(Numeric_Column))
    # print train_numeric.shape
    train_version = train_categorical_ft[[Categorical_Column[0]]].values.reshape(-1, Number_of_RNN, 1)
    train_missing = train_categorical_ft[[Categorical_Column[1]]].values.reshape(-1, Number_of_RNN, 1)
    y_train = train_target.values[::Number_of_RNN]

    test_numeric = test_numeric_ft.values.reshape(-1, Number_of_RNN, len(Numeric_Column))
    test_version = test_categorical_ft[[Categorical_Column[0]]].values.reshape(-1, Number_of_RNN, 1)
    test_missing = test_categorical_ft[[Categorical_Column[1]]].values.reshape(-1, Number_of_RNN, 1)
    y_test = test_target.values[::Number_of_RNN]

    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)

    '''
    Add a function to dupliate the label == 1 observations in trainning set
    '''
    train_numeric = duplicate_portion(train_numeric)
    train_version = duplicate_portion(train_version)
    train_missing = duplicate_portion(train_missing)
    y_train = duplicate_portion(y_train)

    x_train = [train_numeric, train_version, train_missing]
    x_test = [test_numeric, test_version, test_missing]

    return x_train, y_train, x_test, y_test

def get_model_input_submission(test_file):

    test_numeric_ft, test_categorical_ft, test_target = data_transform(test_file)

    test_numeric = test_numeric_ft.values.reshape(-1, Number_of_RNN, len(Numeric_Column))
    test_version = test_categorical_ft[[Categorical_Column[0]]].values.reshape(-1, Number_of_RNN, 1)
    test_missing = test_categorical_ft[[Categorical_Column[1]]].values.reshape(-1, Number_of_RNN, 1)

    x_test = [test_numeric, test_version, test_missing]

    return x_test


if __name__ == '__main__':

    train_file = '../intermediate_data/TRAIN_ft_processed_20.csv'
    test_file = '../intermediate_data/VAL_ft_processed_20.csv'


    x_train, y_train, x_test, y_test = get_model_input(train_file, test_file)

    print x_train[0].shape
    print x_train[1].shape
    print x_train[2].shape
    print y_train.shape
    print x_test[0].shape
    print x_test[1].shape
    print x_test[2].shape
    print y_test.shape