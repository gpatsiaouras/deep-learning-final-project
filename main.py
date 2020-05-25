import matplotlib.pyplot as plt
from scipy import signal
from model import CNNModel
from dataGenerator import data_generator


if __name__ == "__main__":
    model = CNNModel((122, 248))
    model.fit_generator(data_generator("Intra"), epochs=5)
    plt.clf()
    plt.plot(model.history)
    plt.show()
