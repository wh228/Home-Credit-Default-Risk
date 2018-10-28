from model_input import get_model_input
from keras.layers import Input, Reshape, GRU, RepeatVector, TimeDistributed, Dense, Masking, Lambda
from keras.layers.embeddings import Embedding
from keras import backend as K
import numpy as np
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam

from data_process import Number_of_RNN


def rnn_model(x_train, y_train):

    # Inputs
    num = Input(shape=(x_train[0].shape[1], x_train[0].shape[2]))
    version = Input(shape=(x_train[1].shape[1], x_train[1].shape[2]))
    missing = Input(shape=(x_train[2].shape[1], x_train[2].shape[2]))

    inputs = [num, version, missing]

    # Embedding for categorical variables
    reshape_version = Reshape(target_shape=(-1,))(version)
    embedding_version = Embedding(180, 2, input_length=x_train[1].shape[1] * x_train[1].shape[2], mask_zero=True, name='M_version')(reshape_version)

    reshape_missing = Reshape(target_shape=(-1,))(missing)
    embedding_missing = Embedding(4, 1, input_length=x_train[1].shape[1] * x_train[1].shape[2], mask_zero=True, name='M_missing')(reshape_missing)

    num = Masking(mask_value=0, name='M_num')(num)

    # # # concatenate layer
    merge_ft = concatenate([num, embedding_version, embedding_missing], axis=-1, name='concate')

    # GRU with various length
    '''
    Do not use anymore mask layer, as a new layer will overwrite the mask tensor.
    As long as part of the timestep is masked, then the whole timestep is masked and won't be calculated
    '''

    gru_1 = GRU(128, return_sequences=True, name='gru_1')(merge_ft)
    gru_2 = GRU(64, return_sequences=True, name='gru_2')(gru_1)
    gru_3 = GRU(64, name='gru_3')(gru_2)
    ##############################################
    # gru_3 = GRU(64)(merge_ft)

    dense_ft = Dense(2, name='dense_ft')(gru_3)
    outputs = Lambda(lambda x: K.tf.nn.softmax(x), name='outputs')(dense_ft)

    model = Model(inputs=inputs, outputs=outputs)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam)

    return model


# if __name__ == '__main__':
#     train_file = '../intermediate_data/TRAIN_ft_processed_20.csv'
#     test_file = '../intermediate_data/VAL_ft_processed_20.csv'
#     x_train, y_train, x_test, y_test = get_model_input(train_file, test_file)
#     model = rnn_model(x_train, y_train)
#     print model.summary()

############################################

if __name__ == '__main__':
    # for test mask
    # fake num with size 1*5*3
    num = [[[0,0,0],
           [1,2,3],
           [0,0,0],
           [1,2,3],
           [0,0,0]]]
    num = np.array(num)

    c1 = [[[0],
           [1],
           [0],
           [1],
           [0]]]
    c1 = np.array(c1)

    c2 = [[[0],
           [1],
           [0],
           [1],
           [0]]]
    c2 = np.array(c2)

    y = [[0, 1]]
    y = np.array(y)

    x = [num, c1, c2]

    model = rnn_model(x, y)

    layer_name = 'gru_1'

    intermediate_model = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)
    print intermediate_model.predict(x)
