import numpy as np
import pandas as pd


Number_of_RNN = 50

def check_duplicates(filename):
    df = pd.read_csv(filename)
    res = df.duplicated().sum()
    print res

def id_with_installments_history():
    installments_payments = pd.read_csv('../raw_data/installments_payments.csv')
    return installments_payments['SK_ID_CURR'].unique()

'''
since we only use the features from installments history
The not used features in train, test set are deleted
The ID with no installments history are also deleted!!
'''
def process_train():
    train = pd.read_csv('../raw_data/application_train.csv')
    train = train[['SK_ID_CURR', 'TARGET']]
    train = train[train['SK_ID_CURR'].isin(id_with_installments_history())]
    train.to_csv('../intermediate_data/train_chopped.csv', index=False)

def process_test():
    test = pd.read_csv('../raw_data/application_test.csv')
    test = test[['SK_ID_CURR']]
    test = test[test['SK_ID_CURR'].isin(id_with_installments_history())]
    test.to_csv('../intermediate_data/test_chopped.csv', index=False)


'''
Due to limit computing resource, only the most recent Number_of_RNN installment payments are considered!
'''
def process_features():
    # DAYS_INSTALMENT, SK_ID_CURR have no missing value
    ft = pd.read_csv('../raw_data/installments_payments.csv')
    ft.sort_values(by=['SK_ID_CURR', 'DAYS_INSTALMENT'], ascending=True, inplace=True)
    # print ft
    ft.drop(columns=['SK_ID_PREV'], inplace=True)
    ft = ft.groupby(['SK_ID_CURR']).tail(Number_of_RNN)  # get the most recent Number_of_RNN payments
    # print ft

    ft['EARLY_PAYMENT'] = ft['DAYS_INSTALMENT'] - ft['DAYS_ENTRY_PAYMENT']
    ft['EXTRA_PAYMENT'] = ft['AMT_PAYMENT'] - ft['AMT_INSTALMENT']
    ft['MISSING_VALUE'] = 0

    ft.loc[ft['AMT_PAYMENT'].isnull(), 'MISSING_VALUE'] = 1    # mark for rows with missing data
    ft.loc[ft['DAYS_ENTRY_PAYMENT'].isnull(), 'MISSING_VALUE'] = 1  # mark for rows with missing data

    print ft[ft['AMT_PAYMENT'].isnull()].count()
    print ft[ft['MISSING_VALUE'] == 1].count()

    ft.to_csv('../intermediate_data/ft_recent_50.csv', index=False)
    return ft

def process_features_expand():
    '''
    function is to fill the ID with less than Number_of_RNN payments, so each ID will have Number_of_RNN rows!
    '''
    ft = pd.read_csv('../intermediate_data/ft_recent_50.csv')
    ft['RANK'] = ft.groupby('SK_ID_CURR')['DAYS_INSTALMENT'].rank(ascending = False, method='first')
    ft['RANK'] = Number_of_RNN - ft['RANK']  # so we could put the interpolated data at the beginning!!
    # print ft

    id_list = ft['SK_ID_CURR'].unique()
    id_list = np.repeat(id_list, Number_of_RNN)
    id_list = id_list.reshape(-1, 1)
    rank_list = np.array(range(Number_of_RNN))
    rank_list = rank_list.reshape(-1, 1)
    rank_list = np.repeat(rank_list, len(id_list) // Number_of_RNN, axis=1)
    rank_list = np.swapaxes(rank_list, 0, 1)
    rank_list = rank_list.reshape(-1, 1)
    # print id_list
    # print rank_list
    rows = np.concatenate((id_list, rank_list), axis=1)
    # print rows
    ft_new = pd.DataFrame(rows, columns=['SK_ID_CURR', 'RANK'])
    ft_new = ft_new.merge(ft, on=['SK_ID_CURR', 'RANK'], how='left')

    ft_new.drop(columns=['RANK'], inplace=True)

    ft_new.to_csv('../intermediate_data/ft_expand_50.csv', index=False)
    return ft_new


def process_feature_interpolation():
    ft = pd.read_csv('../intermediate_data/ft_expand_50.csv')
    ft.loc[ft['AMT_INSTALMENT'].isnull(), 'MISSING_VALUE'] = 2   # mark for purely fake rows

    print len(ft[ft.MISSING_VALUE != 0]) / len(ft)
    ## 0.01221
    ## about 1 percent missing in total time steps

    ft.interpolate(method='linear', limit_direction='forward', inplace=True)
    ft.interpolate(method='linear', limit_direction='backward', inplace=True)
    ft = ft.astype(int)
    # treat everything as int, even the payment amount!

    ft.to_csv('../intermediate_data/ft_processed_50.csv', index=False)
    return ft


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    # process_train()
    # process_test()
    print process_features()
    print process_features_expand()
    print process_feature_interpolation()
    # check_duplicates('../raw_data/installments_payments.csv')