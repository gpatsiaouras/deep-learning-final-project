from datetime import datetime
from keras.callbacks import TensorBoard
import argparse
from CNN import CNN
from LSTM import LSTMModel
from tabulate import tabulate
from dataGenerator import get_all_dataset

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir="logs/fit/" + timestamp, histogram_freq=1)
tensorboard_callback_eval = TensorBoard(log_dir="logs/eval/" + timestamp, histogram_freq=1)

window_size_options = [61, 73, 122, 292, 488, 584, 4453]

parser = argparse.ArgumentParser(description="Training Configuration")
parser.add_argument("-e", "--epochs", action="store", default=20, type=int, help="Number of epochs to train")
parser.add_argument("-l", "--learning_rate", action="store", default=1e-03, type=float, help="Value of learning rate")
parser.add_argument("-w", "--window", choices=window_size_options, action="store", default=122, type=int,
                    help="Window size to split the data")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--intra", action="store_true", help="")
group.add_argument("--cross", action="store_true")
group.add_argument("--experiment_window", action="store_true")
group.add_argument("--experiment_learning_rate", action="store_true")

args = parser.parse_args()
window_size = args.window
learning_rate = args.learning_rate


def train_intra(window_size, learning_rate=1e-03):
    dataset_type = "Intra"
    x_train, y_train = get_all_dataset(dataset_type, "train", window_size)
    x_test, y_test = get_all_dataset(dataset_type, "test", window_size)

    model = CNN(
        type=dataset_type,
        epochs=args.epochs,
        window_size=window_size,
        input_shape=(x_train.shape[1], x_train.shape[2]),
        learning_rate=learning_rate
    )
    model.fit(
        x=x_train,
        y=y_train,
        callbacks=[tensorboard_callback]
    )

    print("Evaluation:")
    return model.evaluate(x_test, y_test, callbacks=[tensorboard_callback_eval])


def train_cross(window_size, learning_rate=1e-03):
    dataset_type = "Cross"
    x_train, y_train = get_all_dataset(dataset_type, "train", window_size)
    x_test1, y_test1 = get_all_dataset(dataset_type, "test1", window_size)
    x_test2, y_test2 = get_all_dataset(dataset_type, "test2", window_size)
    x_test3, y_test3 = get_all_dataset(dataset_type, "test3", window_size)

    model = CNN(
        type=dataset_type,
        epochs=args.epochs,
        window_size=window_size,
        input_shape=(x_train.shape[1], x_train.shape[2]),
        learning_rate=learning_rate
    )
    model.fit(
        x=x_train,
        y=y_train,
        callbacks=[tensorboard_callback]
    )

    print("Evaluation:")
    return model.evaluate([x_test1, x_test2, x_test3], [y_test1, y_test2, y_test3],
                          callbacks=[tensorboard_callback_eval])


def experiment_window_sizes():
    accuracies = []
    for ws in window_size_options:
        window_size_evaluation = [ws]
        window_size_evaluation.extend(train_intra(ws))
        window_size_evaluation.extend(train_cross(ws))
        accuracies.append(window_size_evaluation)

    print(tabulate(
        accuracies,
        floatfmt=(".0f", ".2f", ".2f", ".2f", ".2f"),
        headers=["Window size", "Intra", "Cross T1", "Cross T2", "Cross T3"])
    )


def experiment_learning_rate():
    learning_rates = [1e-04, 5e-04, 1e-03, 1e-02, 1e-01]
    accuracies = []
    for lr in learning_rates:
        learning_rate_accuracy = [lr]
        learning_rate_accuracy.extend(train_intra(61, lr))
        learning_rate_accuracy.extend(train_cross(61, lr))
        accuracies.append(learning_rate_accuracy)

    print(tabulate(
        accuracies,
        floatfmt=(".4f", ".2f", ".2f", ".2f", ".2f"),
        headers=["Learning Rate", "Intra", "Cross T1", "Cross T2", "Cross T3"])
    )


if __name__ == "__main__":
    if args.cross:
        train_cross(window_size, learning_rate)
    elif args.intra:
        train_intra(window_size, learning_rate)
    elif args.experiment_window:
        experiment_window_sizes()
    elif args.experiment_learning_rate:
        experiment_learning_rate()

