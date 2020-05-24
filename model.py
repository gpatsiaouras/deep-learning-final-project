from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.models import load_model

N_CLASSES = 4


class CNNModel:
    def __init__(self, n_samples, n_features):
        self.name = "CNN"
        self.history = None
        self.model = Sequential()
        self.model.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_samples, n_features)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(n_samples, activation='relu'))
        self.model.add(Dense(N_CLASSES))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, x, y, epochs, verbose=2, batch_size=32):
        self.history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def save_model(self, name):
        self.model.save('saved-models/{}/{}'.format(self.name, name))

    def load_model(self, name):
        self.model = load_model(name)
