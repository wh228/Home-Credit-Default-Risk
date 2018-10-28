import pandas as pd
import numpy as np
from data_process import Number_of_RNN

Validation_Ratio = 0.2

def add_label():
    ft = pd.read_csv('../intermediate_data/ft_processed_50.csv')
    label = pd.read_csv('../intermediate_data/train_chopped.csv')
    ft = ft.astype(int)
    label = label.astype(int)
    ft = ft.merge(label, on=['SK_ID_CURR'], how='left')
    return ft

def split_test():
    ft = add_label()
    ft_test = ft[ft.TARGET.isnull()]
    ft_train = ft[ft.TARGET.notnull()]


    ft_test.to_csv('../intermediate_data/ft_processed_50_test.csv', index=False)
    ft_train.to_csv('../intermediate_data/ft_processed_50_train.csv', index=False)

def split_train():
    ft_train = pd.read_csv('../intermediate_data/ft_processed_50_train.csv')

    ft_train_0 = ft_train[ft_train.TARGET == 0]
    ft_train_0.reset_index(drop=True, inplace=True)
    ft_train_0_train = ft_train_0[:int(len(ft_train_0)//Number_of_RNN * (1 - Validation_Ratio)) * Number_of_RNN]
    ft_train_0_val = ft_train_0[int(len(ft_train_0) // Number_of_RNN * (1 - Validation_Ratio)) * Number_of_RNN:]


    ft_train_1 = ft_train[ft_train.TARGET == 1]
    ft_train_1.reset_index(drop=True, inplace=True)
    ft_train_1_train = ft_train_1[:int(len(ft_train_1) // Number_of_RNN * (1 - Validation_Ratio)) * Number_of_RNN]
    ft_train_1_val = ft_train_1[int(len(ft_train_1) // Number_of_RNN * (1 - Validation_Ratio)) * Number_of_RNN:]

    ft_train_train = pd.concat([ft_train_0_train, ft_train_1_train])
    ft_train_val = pd.concat([ft_train_0_val, ft_train_1_val])
    print len(ft_train_train)
    print len(ft_train_val)

    ft_train_train.to_csv('../intermediate_data/TRAIN_ft_processed_50.csv', index=False)
    ft_train_val.to_csv('../intermediate_data/VAL_ft_processed_50.csv', index=False)


if __name__ == '__main__':
    split_test()
    split_train()