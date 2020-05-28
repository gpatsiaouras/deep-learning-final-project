from datetime import datetime
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np
from model import CNNModel
from dataGenerator import get_all_dataset

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
window_size = 122

parser = argparse.ArgumentParser(description='Training Configuration')
parser.add_argument('-e', '--epochs', action="store", default=20, type=int, help='Number of epochs to train')
args = parser.parse_args()


def evaluate_model(x_test, y_test):
    evaluation = model.model.evaluate(x_test, y_test)
    print("Test Loss: {}".format(evaluation[0]))
    print("Test Accuracy: {}".format(evaluation[1]))
    y_pred = model.model.predict(x=x_test)
    print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))


if __name__ == "__main__":
    # For Cross
    dataset_type = "Cross"
    x_train, y_train = get_all_dataset(dataset_type, "train", window_size)
    x_test1, y_test1 = get_all_dataset(dataset_type, "test1", window_size)
    x_test2, y_test2 = get_all_dataset(dataset_type, "test2", window_size)
    x_test3, y_test3 = get_all_dataset(dataset_type, "test3", window_size)

    # For Intra
    # dataset_type = "Intra"
    # x_train, y_train = get_all_dataset(dataset_type, "train", window_size)
    # x_test, y_test = get_all_dataset(dataset_type, "test", window_size)

    model = CNNModel(type=dataset_type, epochs=args.epochs, input_shape=(x_train.shape[1], x_train.shape[2]))
    model.fit(
        x=x_train,
        y=y_train,
        callbacks=[tensorboard_callback]
    )

    evaluate_model(x_test1, y_test1)
    evaluate_model(x_test2, y_test2)
    evaluate_model(x_test3, y_test3)

