import numpy as np
from keras.models import Sequential
from keras.models import load_model
from datetime import datetime

from sklearn.metrics import confusion_matrix

N_CLASSES = 4


class BaseModel:
    def __init__(self, type, epochs, window_size, input_shape):
        self.name = "Base"
        self.type = type
        self.epochs = epochs
        self.window_size = window_size
        self.history = None
        # Model comes from child class
        self.model = Sequential()

    def fit(self, x, y, callbacks, validation_data=None):
        self.history = self.model.fit(
            x=x,
            y=y,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x_test, y_test, callbacks):
        avg_eval = 0
        # Check if x_test, y_test are not lists (single test). If they are not, make them lists of one item
        if not isinstance(x_test, list):
            x_test = [x_test]
            y_test = [y_test]

        # Evaluate model for each test
        accuracies = []
        for i in range(len(x_test)):
            print("Test " + str(i + 1))
            evaluation = self.model.evaluate(x_test[i], y_test[i], verbose=0, callbacks=callbacks)
            print("Loss: {:.2f}, Accuracy: {:.2f}".format(evaluation[0], evaluation[1]))
            accuracies.append(evaluation[1])
            # Confusion Matrixe
            y_pred = self.model.predict(x=x_test[i])
            print(confusion_matrix(np.argmax(y_test[i], axis=1), np.argmax(y_pred, axis=1)))
            avg_eval += evaluation[1]

        # After finished calculate avg evaluation
        avg_eval = avg_eval / len(x_test)
        print("Average accuracy of all tests: {:.2f}".format(avg_eval))

        # Save high performing models
        if avg_eval > 0.7:
            self.save_model(avg_eval)

        return accuracies

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
