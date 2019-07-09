from keras.models import Model
from keras.layers import Embedding, Dense, Activation, LSTM,Input
from keras.layers.merge import add,concatenate
from keras import optimizers
from EmbeddingMatrix import EmbeddingMatrix
import config
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]  = config.GPU
GRID_COUNT = config.GRID_COUNT

TEXT_K = config.text_k


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

    attrs_latent = concatenate([pl_embedding,time_embedding])
    lstm_out = LSTM(hidden_neurons, return_sequences=True,name='lstm_layer')(attrs_latent)
    dense = Dense(place_dim + 1, name='dense')(lstm_out)
    out_vec = add([dense, user_embedding])
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

    attrs_latent = concatenate([pl_embedding,time_embedding, text_input])
    # time_dist = TimeDistributed(Dense(50))
    lstm_out = LSTM(hidden_neurons, return_sequences=True,name='lstm_layer')(attrs_latent)
    dense = Dense(place_dim + 1, name='dense')(lstm_out)
    out_vec = add([dense, user_embedding])
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

    attrs_latent = concatenate([pl_embedding,time_embedding, text_embedding])
    # time_dist = TimeDistributed(Dense(50))
    lstm_out = LSTM(hidden_neurons, return_sequences=True,name='lstm_layer0')(attrs_latent)
    # lstm_out = LSTM(hidden_neurons, return_sequences=True, name='lstm_layer1')(lstm_out)
    # lstm_out = LSTM(hidden_neurons, return_sequences=True, name='lstm_layer2')(lstm_out)
    dense = Dense(place_dim + 1, name='dense')(lstm_out)
    out_vec = add([dense,user_embedding])
    pred = Activation('softmax')(out_vec)
    model = Model([pl_input,time_input,user_input,text_input], pred)

    # model.load_weights('./model/User_RNN_Seg_Epoch_0.3_rmsprop_300.h5')
    # Optimization
    sgd = optimizers.SGD(lr=learning_rate)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy')
    model.summary()
    return model