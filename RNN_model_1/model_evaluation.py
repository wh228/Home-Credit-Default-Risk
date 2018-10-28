
from model_input import get_model_input
from model_build import rnn_model
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from os.path import exists
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score
from data_process import Number_of_RNN


def model_fit_predict(x_train, y_train, x_test, y_test):

    model = rnn_model(x_train, y_train)

    path = 'compiled/20181026_rnn3_50_model.h5'
    if exists(path):
        model = load_model(path)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    model.fit(x_train, y_train, epochs=200, batch_size=512, validation_split=0.1, verbose=1, shuffle=True, callbacks=[earlyStopping])
    model.save(path)

    prediction = model.predict(x_test)

    return prediction


def calculate_metrics(truth, pred):

    ac = accuracy_score(truth, pred)
    cm = confusion_matrix(truth, pred)

    print ac
    print cm

    return

def get_prediction(train_file, test_file):
    x_train, y_train, x_test, y_test = get_model_input(train_file, test_file)
    pred = model_fit_predict(x_train, y_train, x_test, y_test)

    pred = np.argmax(pred, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    calculate_metrics(y_test, pred)

    test_id = pd.read_csv(test_file)['SK_ID_CURR'][::Number_of_RNN]
    outputs = pd.DataFrame({'SK_ID_CURR': test_id, 'truth': y_test, 'prediction': pred},
                          columns=['SK_ID_CURR', 'truth', 'prediction'])
    return outputs



# def calculate_accuracy(output):
#     prediction = output['AQI'].tolist()
#     truth = output['AQI_truth'].tolist()
#     ac = accuracy_score(truth, prediction)
#     cm = confusion_matrix(truth, prediction)
#     plt.imshow(cm, cmap=plt.cm.Blues)
#     for x in range(len(cm)):
#         for y in range(len(cm)):
#             plt.annotate(cm[x, y], xy = (y, x), horizontalalignment = 'center', verticalalignment = 'center')
#     plt.xticks(np.arange(0, 5), np.arange(1, 6))
#     plt.yticks(np.arange(0, 5), np.arange(1, 6))
#     plt.xlabel('Predicted AQI')
#     plt.ylabel('True AQI')
#     return ac, plt

def get_auc_data(train_file, test_file):

    path = 'compiled/20181026_rnn3_50_model.h5'
    model = load_model(path)
    x_train, y_train, x_test, y_test = get_model_input(train_file, test_file)
    prediction = model.predict(x_test)
    y_test = np.argmax(y_test, axis=-1)

    test_id = pd.read_csv(test_file)['SK_ID_CURR'][::Number_of_RNN]
    test_id = np.array(test_id).reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    prediction = prediction.reshape(-1, 2)
    outputs = np.concatenate((test_id, y_test), axis=1)
    outputs = np.concatenate((outputs, prediction), axis=1)
    outputs = pd.DataFrame(outputs, columns=['SK_ID_CURR', 'truth','0','1'])

    outputs.to_csv('../outputs/20181026_rnn3_50_auc.csv', index=False)


if __name__ == '__main__':


    train_file = '../intermediate_data/TRAIN_ft_processed_50.csv'
    test_file = '../intermediate_data/VAL_ft_processed_50.csv'



    # print get_prediction(train_file, test_file)

    # outputs = get_prediction(train_file, test_file)
    # outputs.to_csv('../outputs/20181026_rnn3_50.csv', index=False)

    ###########################################################

    get_auc_data(train_file, test_file)


