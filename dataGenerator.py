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
file_numbers = {
    "test": [9, 10],
    "train": [1, 2, 3, 4, 5, 6, 7, 8]
}


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
        unscaled_data = np.transpose(h5py.File(files[i], "r").get(dataset_name)[()])
        dataMax = np.max(unscaled_data)
        dataMin = np.min(unscaled_data)
        scaled_data = (unscaled_data - dataMin) / (dataMax - dataMin)
        scaled_data = scaled_data.reshape((35624 // window_size, window_size, 248))
        data.append(scaled_data)
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


def get_all_dataset(folder, data_type, window_size):
    data = []
    labels = []
    class_counter = 0
    for className in classes:
        files = glob.glob("{}/{}/{}_*.h5".format(folder, data_type, className))
        for file in files:
            # Set the dataset name (to open the dataset)
            dataset_name = "_".join(file.split("/")[-1].split("_")[:-1])

            # Read the data from the file and transpose them
            unscaled_data = np.transpose(h5py.File(file, "r").get(dataset_name)[()])

            # Scale using Min Max
            dataMax = np.max(unscaled_data)
            dataMin = np.min(unscaled_data)
            scaled_data = (unscaled_data - dataMin) / (dataMax - dataMin)

            # Reshape to create windows according to window size
            scaled_data = scaled_data.reshape((35624 // window_size, window_size, 248))

            # Create zeroed array and replace the current class index with 1 to build label for that file
            # and for that number of windows
            label = np.zeros((35624 // window_size, 4))
            label[:, class_counter] = 1

            # Append data and label to the lists
            data.extend(scaled_data)
            labels.extend(label)
        class_counter += 1

    return np.asarray(data), np.asarray(labels)
