from tensorflow.python.keras.layers.pooling import GlobalPooling1D

from model import BaseModel
import keras
from keras.layers import Dense, Dropout, BatchNormalization, LSTM, AveragePooling1D
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential


class CNN(BaseModel):
    def __init__(self, type, epochs, window_size, input_shape, learning_rate=1e-03):
        super().__init__(type, epochs, window_size, input_shape)
        self.name = "CNN"
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape))
        self.model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
        self.model.add(Dropout(0.2))
        # self.model.add(GlobalPooling1D())  # We don't want to extract important features
        # self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dense(4, activation="softmax"))
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.adam(learning_rate=learning_rate),
            metrics=["accuracy"]
        )
        print(self.model.summary())
