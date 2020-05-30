from datetime import datetime
from keras.callbacks import TensorBoard
import argparse
from model import CNNModel
from dataGenerator import get_all_dataset

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

parser = argparse.ArgumentParser(description='Training Configuration')
parser.add_argument('-e', '--epochs', action="store", default=20, type=int, help='Number of epochs to train')
parser.add_argument('-w', '--window', action="store", default=122, type=int, help='Window size to split the data')
args = parser.parse_args()
window_size = args.window


def train_intra(window_size):
    dataset_type = "Intra"
    x_train, y_train = get_all_dataset(dataset_type, "train", window_size)
    x_test, y_test = get_all_dataset(dataset_type, "test", window_size)

    model = CNNModel(
        type=dataset_type,
        epochs=args.epochs,
        window_size=window_size,
        input_shape=(x_train.shape[1], x_train.shape[2]))
    model.fit(
        x=x_train,
        y=y_train,
        callbacks=[tensorboard_callback]
    )

    print("Evaluation:")
    model.evaluate(x_test, y_test)


def train_cross(window_size):
    dataset_type = "Cross"
    x_train, y_train = get_all_dataset(dataset_type, "train", window_size)
    x_test1, y_test1 = get_all_dataset(dataset_type, "test1", window_size)
    x_test2, y_test2 = get_all_dataset(dataset_type, "test2", window_size)
    x_test3, y_test3 = get_all_dataset(dataset_type, "test3", window_size)

    model = CNNModel(
        type=dataset_type,
        epochs=args.epochs,
        window_size=window_size,
        input_shape=(x_train.shape[1], x_train.shape[2]))
    model.fit(
        x=x_train,
        y=y_train,
        callbacks=[tensorboard_callback]
    )

    print("Evaluation:")
    model.evaluate([x_test1, x_test2, x_test3], [y_test1, y_test2, y_test3])


if __name__ == "__main__":
    train_cross(window_size)

    # Window size experiment
    # window_sizes = [61, 73, 122, 292, 488, 584, 4453]
    # for ws in window_sizes:
    #     print(ws)
    #     print(train_cross(ws))

