import keras
import random
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.models import load_model
from datetime import datetime

N_CLASSES = 4


class CNNModel:
    def __init__(self, type, epochs, window_size, input_shape):
        self.name = "CNN"
        self.type = type
        self.epochs = epochs
        self.window_size = window_size
        self.history = None
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=24, activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling1D(pool_size=24))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(122, activation='relu'))
        self.model.add(Dense(N_CLASSES, activation='softmax'))
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.adam(),
            metrics=['accuracy']
        )

    def fit(self, x, y, callbacks):
        self.history = self.model.fit(
            x=x,
            y=y,
            epochs=self.epochs,
            callbacks=callbacks
        )
        self.save_model()

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self):
        self.model.save('saved_models/{}/{}_{}_epochs_{}_window_{}'.format(
            self.name,
            datetime.today().strftime("%Y%m%d_%H_%M_%S"),
            self.type,
            self.epochs,
            self.window_size
        ))

    def load_model(self, name):
        self.model = load_model(name)
