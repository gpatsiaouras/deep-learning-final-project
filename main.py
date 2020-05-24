import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import signal
from model import CNNModel

# Constants
REST = "rest"
MOTOR = "task_motor"
MATH = "task_story_math"
MEMORY = "task_working_memory"
SUBJECT_1 = "105923"

dataset = '_'.join([MEMORY, SUBJECT_1])
filename = '/'.join(["Intra", "train", dataset + "_1.h5"])
window_size = 50

def construct_dataset(data):
    input = []
    labels = []
    for idx in range(len(data) - window_size):
        input.append(data[idx: idx + window_size])
        # Label is the temperature of the last city
        labels.append(data[idx + window_size][-1][0])
    return np.array(input), np.array(labels)


if __name__ == "__main__":
    file = h5py.File(filename, 'r')
    matrix = file.get(dataset)[()]
    matrix = np.transpose(matrix)
    sampled_dataset = signal.decimate(matrix, 15, axis=0)
    plt.clf()
    plt.plot(matrix)
    plt.show()
    plt.clf()
    plt.plot(sampled_dataset)
    plt.show()
    file.close()
