import numpy as np
import keras
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.models import load_model
from datetime import datetime

from sklearn.metrics import confusion_matrix

N_CLASSES = 4


class CNNModel:
    def __init__(self, type, epochs, window_size, input_shape):
        self.name = "CNN"
        self.type = type
        self.epochs = epochs
        self.window_size = window_size
        self.history = None
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape))
        self.model.add(MaxPooling1D(pool_size=3))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        # self.model.add(Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape))
        # self.model.add(MaxPooling1D(pool_size=3))
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(N_CLASSES, activation="softmax"))
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.adam(),
            metrics=["accuracy"]
        )
        print(self.model.summary())

    def fit(self, x, y, callbacks):
        self.history = self.model.fit(
            x=x,
            y=y,
            epochs=self.epochs,
            callbacks=callbacks
        )

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x_test, y_test):
        avg_eval = 0
        # Check if x_test, y_test are not lists (single test). If they are not, make them lists of one item
        if not isinstance(x_test, list):
            x_test = [x_test]
            y_test = [y_test]

        # Evaluate model for each test
        for i in range(len(x_test)):
            print("Test " + str(i + 1))
            evaluation = self.model.evaluate(x_test[i], y_test[i], verbose=0)
            print("Loss: {:.2f}, Accuracy: {:.2f}".format(evaluation[0], evaluation[1]))
            # Confusion Matrix
            y_pred = self.model.predict(x=x_test[i])
            print(confusion_matrix(np.argmax(y_test[i], axis=1), np.argmax(y_pred, axis=1)))
            avg_eval += evaluation[1]

        # After finished calculate avg evaluation
        avg_eval = avg_eval / len(x_test)
        print("Average accuracy of all tests: {:.2f}".format(avg_eval))

        # Save high performing models
        if avg_eval > 0.7:
            self.save_model(avg_eval)

    def save_model(self, accuracy):
        filename = "saved_models/{}/{}_{}_e{}_w{}_acc{:.2f}".format(
            self.name,
            datetime.today().strftime("%Y%m%d_%H_%M_%S"),
            self.type,
            self.epochs,
            self.window_size,
            accuracy,
        )
        self.model.save(filename)
        print("Accuracy more than 70%, saved model in {}".format(filename))

    def load_model(self, name):
        self.model = load_model(name)
