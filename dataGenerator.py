import h5py
import glob
import numpy as np

# Constants
classes = [
    "rest",
    "task_motor",
    "task_story_math",
    "task_working_memory"
]
SUBJECT_1 = "105923"
INTEGER_DIVISION_WINDOW_SIZES = [1, 2, 4, 8, 61, 73, 122, 146, 244, 292, 488, 584, 4453, 8906, 17812]


def data_generator(folder, data_type, window_size):
    """
    Generator that prepares the data and labels and the yields a tuple of x, y based on the step of the epoch
    :param folder:
    :param data_type:
    :param window_size:
    :return: x, y
    """
    if not window_size in INTEGER_DIVISION_WINDOW_SIZES:
        raise Exception("Unsupported window size {}, only {} are supported".format(window_size, INTEGER_DIVISION_WINDOW_SIZES))
    # Scan the train directory for files
    files = glob.glob("{}/{}/*.h5".format(folder, data_type))

    # Read all the files and for each file save the data and labels (same indexing as files)
    data = []
    labels = []
    for i in range(len(files)):
        dataset_name = "_".join(files[i].split("/")[-1].split("_")[:-1])
        data.append(np.transpose(h5py.File(files[i], "r").get(dataset_name)[()]).reshape((35624 // window_size, window_size, 248)))
        labels.append(get_y(dataset_name))

    # Start the generator
    current_file_idx = 0
    while True:
        yield data[current_file_idx], labels[current_file_idx]
        current_file_idx += 1
        if current_file_idx == len(files):
            current_file_idx = 0


def get_y(dataset_name):
    y = np.zeros((292, 4))
    if dataset_name.startswith(classes[3]):
        y[:, 3] = 1
    elif dataset_name.startswith(classes[2]):
        y[:, 2] = 1
    elif dataset_name.startswith(classes[1]):
        y[:, 1] = 1
    elif dataset_name.startswith(classes[0]):
        y[:, 0] = 1

    return y
