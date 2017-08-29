from keras.layers import *
import keras.backend as K
from keras import optimizers
from keras.models import Sequential,Model

class EmbeddingMatrix(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(EmbeddingMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(EmbeddingMatrix, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, x, mask=None):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


def test():
    inputs = Input(shape=(2, 4), dtype='float32', name='text_input')
    inputs_embedded = EmbeddingMatrix(50)(inputs)

    encoded = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(inputs_embedded)
    dense = Dense(80, name='dense')(encoded)
    pred = Activation('softmax')(dense)
    model = Model(inputs,dense)

    model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=0.01, clipnorm=5))
    model.summary()

    traindata = [[[0.1,0,0,0.9] for i in range(2)] for j in range(10)]
    re = model.predict(traindata)
    print re

if __name__ == '__main__':
    test()
