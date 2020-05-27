import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import confusion_matrix

from model import CNNModel
from dataGenerator import data_generator


if __name__ == "__main__":
    model = CNNModel((122, 248))
    model.fit(
        generator=data_generator("Intra", "train", 122),
        epochs=5,
        steps_per_epoch=32,
    )
    predictions = model.predict(
        generator=data_generator("Intra", "test", 122),
        steps=8
    )
