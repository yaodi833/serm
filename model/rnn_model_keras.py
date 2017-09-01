from keras.models import Sequential,Model
from keras.layers import Embedding, Dense, merge, SimpleRNN, Merge, Activation, LSTM, GRU, Dropout,Input,TimeDistributed
from keras import optimizers
from EmbeddingMatrix import EmbeddingMatrix
from keras.utils.np_utils import to_categorical
import config

GRID_COUNT = config.GRID_COUNT

TEXT_K = config.text_k


# def geo_rnn_model(user_dim, place_dim = GRID_COUNT*GRID_COUNT, time_dim=24, pl_d=100, time_k=100,
#                   hidden_neurons=200, learning_rate=0.003):
#     # RNN model construction
#     pl_model = Sequential()
#     pl_model.add(Embedding(place_dim + 1, pl_d, mask_zero=True, ))
#     pl_model.summary()
#     # pl_model.add(Dense(hidden_neurons,use_bias=False))
#     time_model = Sequential()
#     time_model.add(Embedding(time_dim + 1, time_k, mask_zero=True))
#     # time_model.add(Dense(hidden_neurons,use_bias=False))
#     time_model.summary()
#     user_model = Sequential()
#     user_model.add(Embedding(user_dim + 1, place_dim + 1, mask_zero=True))
#     # user_model.add(Embedding(user_dim+1, user_r,mask_zero=True))
#     # user_model.add(Dense(place_dim+1))
#     user_model.summary()
#     rnn_model = Sequential()
#     rnn_model.add(Merge([pl_model, time_model], mode='concat'))
#     rnn_model.add(LSTM(hidden_neurons, return_sequences=True))
#     # rnn_model.add(Dropout(0.2))
#     rnn_model.add(Dense(place_dim + 1))
#     rnn_model.summary()
#     model = Sequential()
#     model.add(Merge([rnn_model, user_model], mode='sum'))
#     model.add(Activation('softmax'))
#
#     # model.load_weights('./model/User_RNN_Seg_Epoch_0.3_rmsprop_300.h5')
#     # Optimization
#     sgd = optimizers.SGD(lr=learning_rate)
#     rmsprop = optimizers.RMSprop(lr=learning_rate)
#     model.compile(optimizer=rmsprop, loss='categorical_crossentropy')
#     model.summary()
#     return model

def geo_lprnn_model(user_dim, len, place_dim = GRID_COUNT*GRID_COUNT, time_dim=config.time_dim, pl_d=config.pl_d,
                    time_k=config.time_k, hidden_neurons=config.hidden_neurons, learning_rate=config.learning_rate):
    # RNN model construction
    pl_input = Input(shape=(len-1,), dtype='int32', name = 'pl_input')
    time_input = Input(shape=(len-1,), dtype='int32', name = 'time_input')
    user_input = Input(shape=(len-1,), dtype='int32', name='user_input')

    pl_embedding = Embedding(input_dim=place_dim + 1, output_dim=pl_d, name ='pl_embedding' ,
                             mask_zero=True)(pl_input)
    time_embedding = Embedding(input_dim=time_dim + 1, output_dim=time_k, name='time_embedding',
                               mask_zero=True)(time_input)
    user_embedding = Embedding(input_dim=user_dim + 1, output_dim=place_dim + 1, name='user_embedding',
                               mask_zero=True)(user_input)

    attrs_latent = merge([pl_embedding,time_embedding],mode='concat')
    lstm_out = LSTM(hidden_neurons, return_sequences=True,name='lstm_layer')(attrs_latent)
    dense = Dense(place_dim + 1, name='dense')(lstm_out)
    out_vec = merge([dense,user_embedding],mode='sum')
    pred = Activation('softmax')(out_vec)
    model = Model([pl_input,time_input,user_input], pred)

    # model.load_weights('./model/User_RNN_Seg_Epoch_0.3_rmsprop_300.h5')
    # Optimization
    sgd = optimizers.SGD(lr=learning_rate)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy')
    model.summary()
    return model


def geo_lprnn_text_model(user_dim, len, place_dim = GRID_COUNT*GRID_COUNT, time_dim=config.time_dim, pl_d=config.pl_d,
                    time_k=config.time_k, hidden_neurons=config.hidden_neurons, learning_rate=config.learning_rate):
    # RNN model construction
    pl_input = Input(shape=(len-1,), dtype='int32', name = 'pl_input')
    time_input = Input(shape=(len-1,), dtype='int32', name = 'time_input')
    user_input = Input(shape=(len-1,), dtype='int32', name='user_input')
    text_input = Input(shape=(len-1, pl_d), dtype='float32', name='text_input')


    pl_embedding = Embedding(input_dim=place_dim + 1, output_dim=pl_d, name ='pl_embedding' ,
                             mask_zero=True)(pl_input)
    time_embedding = Embedding(input_dim=time_dim + 1, output_dim=time_k, name='time_embedding',
                               mask_zero=True)(time_input)
    user_embedding = Embedding(input_dim=user_dim + 1, output_dim=place_dim + 1, name='user_embedding',
                               mask_zero=True)(user_input)
    # text_embedding = Dense(pl_d)(text_input)

    attrs_latent = merge([pl_embedding,time_embedding, text_input],mode='concat')
    # time_dist = TimeDistributed(Dense(50))
    lstm_out = LSTM(hidden_neurons, return_sequences=True,name='lstm_layer')(attrs_latent)
    dense = Dense(place_dim + 1, name='dense')(lstm_out)
    out_vec = merge([dense,user_embedding],mode='sum')
    pred = Activation('softmax')(out_vec)
    model = Model([pl_input,time_input,user_input,text_input], pred)

    # model.load_weights('./model/User_RNN_Seg_Epoch_0.3_rmsprop_300.h5')
    # Optimization
    sgd = optimizers.SGD(lr=learning_rate)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy')
    model.summary()
    return model

def geo_lprnn_trainable_text_model(user_dim, len,word_vec, place_dim = GRID_COUNT*GRID_COUNT, time_dim=config.time_dim,
                            pl_d=config.pl_d, time_k=config.time_k, hidden_neurons=config.hidden_neurons,
                                   learning_rate=config.learning_rate):
    # RNN model construction
    pl_input = Input(shape=(len-1,), dtype='int32', name = 'pl_input')
    time_input = Input(shape=(len-1,), dtype='int32', name = 'time_input')
    user_input = Input(shape=(len-1,), dtype='int32', name='user_input')
    text_input = Input(shape=(len-1, word_vec.shape[0]), dtype='float32', name='text_input')


    pl_embedding = Embedding(input_dim=place_dim + 1, output_dim=pl_d, name ='pl_embedding' ,
                              mask_zero=True)(pl_input)
    time_embedding = Embedding(input_dim=time_dim + 1, output_dim=time_k, name='time_embedding',
                               mask_zero=True)(time_input)
    user_embedding = Embedding(input_dim=user_dim + 1, output_dim=place_dim + 1, name='user_embedding',
                               mask_zero=True)(user_input)

    # text_embedding = Embedding(input_dim=word_vec.shape[0],output_dim= TEXT_K,
    #                           weights=[word_vec],name="text_embeddng")(text_input)
    text_embedding = EmbeddingMatrix(TEXT_K, weights=[word_vec], name="text_embeddng", trainable=True)(text_input)

    attrs_latent = merge([pl_embedding,time_embedding, text_embedding],mode='concat')
    # time_dist = TimeDistributed(Dense(50))
    lstm_out = LSTM(hidden_neurons, return_sequences=True,name='lstm_layer0')(attrs_latent)
    # lstm_out = LSTM(hidden_neurons, return_sequences=True, name='lstm_layer1')(lstm_out)
    # lstm_out = LSTM(hidden_neurons, return_sequences=True, name='lstm_layer2')(lstm_out)
    dense = Dense(place_dim + 1, name='dense')(lstm_out)
    out_vec = merge([dense,user_embedding],mode='sum')
    pred = Activation('softmax')(out_vec)
    model = Model([pl_input,time_input,user_input,text_input], pred)

    # model.load_weights('./model/User_RNN_Seg_Epoch_0.3_rmsprop_300.h5')
    # Optimization
    sgd = optimizers.SGD(lr=learning_rate)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy')
    model.summary()
    return model