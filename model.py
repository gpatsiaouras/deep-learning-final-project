from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.models import load_model

N_CLASSES = 4


class CNNModel:
    def __init__(self, input_shape):
        self.name = "CNN"
        self.history = None
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(292, activation='relu'))
        self.model.add(Dense(N_CLASSES))
        self.model.compile(optimizer='adam', loss='mse')

    def fit_generator(self, generator, epochs):
        self.history = self.model.fit_generator(generator, epochs=epochs, steps_per_epoch=32)

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def save_model(self, name):
        self.model.save('saved-models/{}/{}'.format(self.name, name))

    def load_model(self, name):
        self.model = load_model(name)
