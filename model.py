import keras
import random
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.models import load_model
from datetime import datetime

N_CLASSES = 4


class CNNModel:
    def __init__(self, input_shape):
        self.name = "CNN"
        self.history = None
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(122, activation='relu'))
        self.model.add(Dense(N_CLASSES))
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy']
        )

    def fit(self, generator, epochs, steps_per_epoch):
        self.history = self.model.fit(
            x=generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
        self.save_model(epochs)

    def predict(self, generator, steps):
        return self.model.predict(generator, steps=steps)

    def save_model(self, epochs):
        self.model.save('saved_models/{}/{}_epochs_{}'.format(
            self.name,
            datetime.today().strftime("%Y%m%d_%H_%M_%S"),
            epochs
        ))

    def load_model(self, name):
        self.model = load_model(name)
