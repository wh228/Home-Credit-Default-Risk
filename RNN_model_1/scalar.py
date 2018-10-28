import numpy as np
import pandas as pd


'''
implement a Mini Max Scalar
with set limits
'''

class Scalar():
    def __init__(self):
        '''
        # upgrdated value according to '../intermediate_data/ft_processed_20.csv'
        '''
        self.map = {'NUM_INSTALMENT_NUMBER': (1, 277),
                    'DAYS_INSTALMENT': (-2922, -1),
                    'DAYS_ENTRY_PAYMENT': (-4921, -1),
                    'AMT_INSTALMENT': (0, 3771487),
                    'AMT_PAYMENT': (0, 3771487),
                    'EARLY_PAYMENT': (-2882, 3189),
                    'EXTRA_PAYMENT': (-2424726, 2383748),
                    }
    def transform(self, data, name):
        # name = name.split('_')[0]
        try:
            min, max = self.map[name]
            return (data - min)/float(max-min)
        except:
            if name not in ('NUM_INSTALMENT_VERSION', 'MISSING_VALUE'):
                raise ValueError('Scaler is not used for what it should be used!')
            return data

    def inverse_transform(self, data, name):
        # name = name.split('_')[0]
        try:
            min, max = self.map[name]
            return data * (max - min) + min
        except:
            if name not in ('NUM_INSTALMENT_VERSION', 'MISSING_VALUE'):
                raise ValueError('Scaler is not found!')
            return data

def check_MiniMax():
    df = pd.read_csv('../intermediate_data/ft_processed_20.csv')
    print df.columns
    print df.min()
    print df.max()

if __name__ == '__main__':
    check_MiniMax()