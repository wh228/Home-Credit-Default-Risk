
from model_input import get_model_input_submission
from model_build import rnn_model
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from os.path import exists
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score
from data_process import Number_of_RNN


def get_data(path, test_file):

    model = load_model(path)
    x_test = get_model_input_submission(test_file)
    prediction = model.predict(x_test)

    test_id = pd.read_csv(test_file)['SK_ID_CURR'][::Number_of_RNN]
    test_id = np.array(test_id).reshape(-1, 1)
    prediction = prediction.reshape(-1, 2)
    outputs = np.concatenate((test_id, prediction), axis=1)
    outputs = pd.DataFrame(outputs, columns=['SK_ID_CURR', '0','TARGET'])

    outputs = outputs[['SK_ID_CURR', 'TARGET']]
    return outputs

if __name__ == '__main__':

    model_path = 'compiled/20181026_rnn3_50_model.h5'

    test_file = '../intermediate_data/ft_processed_50_test.csv'


    outputs = get_data(model_path, test_file)

    submission = pd.read_csv('../raw_data/sample_submission.csv')
    submission.drop(columns=['TARGET'], inplace=True)
    submission = submission.merge(outputs, on=['SK_ID_CURR'], how='left')
    submission.loc[submission.TARGET.isnull(), 'TARGET'] = 0.081

    submission.to_csv('../outputs/20181027_submission_rnn3_50.csv', index=False)




