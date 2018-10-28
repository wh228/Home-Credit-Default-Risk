import numpy as np
import pandas as pd
import gc
from data_process import Number_of_RNN
from scalar import Scalar

Numeric_Column = ['NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT',
                  'EARLY_PAYMENT', 'EXTRA_PAYMENT']
Categorical_Column = ['NUM_INSTALMENT_VERSION', 'MISSING_VALUE']


def data_transform(filename):

    df = pd.read_csv(filename)
    # df = df[100060:100100] # for test purpose only!
    df.reset_index(drop=True, inplace=True)

    target = df[['TARGET']]

    numeric_column = Numeric_Column
    categorical_column = Categorical_Column

    for column in numeric_column:
        df[column] = Scalar().transform(df[column], column)

    '''
    To support various length RNN, make sure all timesteps for masked are 0
    '''
    for column in categorical_column:
        df.loc[:, column] = df[column] + 1
    # avoid 0 for categorical variables

    # print df.head()

    for column in numeric_column + ['NUM_INSTALMENT_VERSION']:
        df.loc[df.MISSING_VALUE == 3, column] = 0
    df.loc[df.MISSING_VALUE == 3, 'MISSING_VALUE'] = 0

    numeric_ft = df[numeric_column]
    categorical_ft = df[categorical_column]

    # print len(df[df.MISSING_VALUE == 0])

    return numeric_ft, categorical_ft, target


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)

    filename = '../intermediate_data/VAL_ft_processed_50.csv'
    numeric_ft, categorical_ft, target = data_transform(filename)
    print numeric_ft.head()
    print categorical_ft.head()
    print target.head()
