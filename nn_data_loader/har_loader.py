"""
Implements the SensorDataLoader class by inheriting the DataLoader base class.
"""
import os
from numpy import dstack
from tensorflow.keras.utils import to_categorical
from pandas import read_csv
import pandas as pd
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# import from other folders
import sys
# sys.path.append('../')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from nn_base.nn_base_data_loader import DataLoader
from nn_utils.process_argument_har import get_args
from helper_functions.read_yaml import ReadYml

# Main Application directory
main_app_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))

# read yml configuration for nn setup
nn_setup_yml = ReadYml('nn_setup.yml')
nn_setup_conf = nn_setup_yml.load_yml()

# load a single file as a numpy array
def load_file(filepath):
    datasets_dir = 'datasets'
    # curr_dir = Path().absolute()  # current dir
    filepath = os.path.join(main_app_path, datasets_dir, filepath)
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    # print(dataframe.head())
    return dataframe.values


# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load the dataset, returns train and test X and y elements
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y

# standardize data
def scale_data(trainX, testX):
	# remove overlap
	cut = int(trainX.shape[1] / 2)
	longX = trainX[:, -cut:, :]
	# flatten windows
	longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
	# flatten train and test
	flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
	flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
	# standardize
	# if standardize:
	s = StandardScaler()
	# fit on training data
	s.fit(longX)
	# apply to training and test data
	longX = s.transform(longX)
	flatTrainX = s.transform(flatTrainX)
	flatTestX = s.transform(flatTestX)
	# reshape
	flatTrainX = flatTrainX.reshape((trainX.shape))
	flatTestX = flatTestX.reshape((testX.shape))
	return flatTrainX, flatTestX


class SensorDataLoader(DataLoader):
    def __init__(self, config):
        """
        Constructor to initialize the training and testing datasets for FashionMNIST.

        :param config: the json configuration namespace.
        :return none
        :raises none
        """

        super().__init__(config)
        return

    # load the dataset, returns train and test X and y elements
    def load_dataset(self, prefix=''):
        """
        Loads the fashion_mnist image dataset, and
        Updates the respective class members.

        :param none
        :return none
        :raises none
        """
        # load all train
        self.train_data, self.train_labels = load_dataset_group('train', prefix + nn_setup_conf['data_folder'] + '/')
        # load all test
        self.test_data, self.test_labels = load_dataset_group('test', prefix + nn_setup_conf['data_folder'] + '/')

        # data standardization
        print('Standardizing the data..')
        self.train_data, self.test_data = scale_data(self.train_data, self.test_data)
        print('Data standardization completed.')

    # plot a histogram of each variable in the dataset
    def plot_variable_distributions(self, data, which_data):
        print('Plot a histogram of each sensor variable in the dataset')
    	# remove overlap
        cut = int(data.shape[1] / 2)
        longX = data[:, -cut:, :]
        # flatten windows
        longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
        print('Flattened windows of the data:', longX.shape)
        plt.figure()
        xaxis = None
        for i in range(longX.shape[1]):
            ax = plt.subplot(longX.shape[1], 1, i+1, sharex=xaxis)
            ax.set_xlim(-1, 1)
            if i == 0:
                xaxis = ax
            plt.hist(longX[:, i], bins=50, ec='black')

        # Save the plot to disk.
        if self.config.config_namespace.mode == 'save':
        	dist_file_name = "save_train_val"
        elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'true':
        	dist_file_name = "load_test"
        elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'false':
        	dist_file_name = "load_pred.png"

        sensor_data_dist_path = os.path.join(main_app_path, self.config.config_namespace.graph_dir, "{}_sensor_{}_values_distributions.png".format(dist_file_name, which_data))
        if(self.config.config_namespace.save_plots == 'true'):
            plt.savefig(sensor_data_dist_path, bbox_inches='tight')
            print('Sensor {} plot distributions saved successfully.'.format(which_data))
        else:
            plt.show()
        plt.close()

    def display_data_element(self, which_data, index):
        """
        Displays a data element from the FashionMNIST dataset (training/testing).

        :param  which_data: Specifies the dataset to be used (i.e., training or testing).
        :param index: Specifies the index of the data element within a particular dataset.
        :returns none
        :raises none
        """

        # Display a training data element.
        if which_data == "train_data":
            print('TRAIN DATA:')
            # Plot axes' values distributions
            print(self.train_data[index, 1, 1])
            self.plot_variable_distributions(self.train_data, which_data)
            # Plot time series of a train data instance
            print('Plot time series of a train data instance.')
            df_time_series = pd.DataFrame(data=self.train_data[index])
            df_time_series.plot(figsize=(20,10), linewidth=2, fontsize=20)
            plt.xlabel('Sample', fontsize=20);
            plt.ylabel('Axes', fontsize=20);
            # Save the plot to disk.
            sensor_data_ts_path = os.path.join(main_app_path, self.config.config_namespace.graph_dir, "sensor_ts_index_{}_{}.png".format(index, which_data))
            plt.savefig(sensor_data_ts_path)
            print('Time series plot for the {} element is saved successfully.'.format(index))
            plt.close()
        elif which_data == 'test_data':
            print('TEST DATA')
            print(self.test_data[index, 1, 1])
            self.plot_variable_distributions(self.test_data, which_data)
            # Plot time series of a test data instance
            print('Plot time series of a test data instance.')
            df_time_series = pd.DataFrame(data=self.test_data[index])
            df_time_series.plot(figsize=(20,10), linewidth=2, fontsize=20)
            plt.xlabel('Sample', fontsize=20);
            plt.ylabel('Axes', fontsize=20);
            # Save the plot to disk.
            sensor_data_ts_path = os.path.join(main_app_path, self.config.config_namespace.graph_dir, "sensor_ts_index_{}_{}.png".format(index, which_data))
            plt.savefig(sensor_data_ts_path)
            print('Time series plot for the {} element is saved successfully.'.format(index))
            plt.close()
        else:
            print('Error: display_data_element: "which_data" parameter is invalid!')


    def preprocess_dataset(self):
        """
        Preprocess the Sensors dataset.

        Performs data type conversion and normalization on data values of training and testing dataset, and
        Converts the categorical class labels to boolean one-hot encoded vector for training and testing datasets.

        :param none
        :returns none
        :raises none
        """

        # zero-offset class values
        self.train_labels = self.train_labels - 1
        self.test_labels = self.test_labels - 1

        # Convert the class labels from categorical to boolean one hot encoded vector.
        self.train_label_one_hot = to_categorical(self.train_labels)
        self.test_label_one_hot = to_categorical(self.test_labels)

        print("Training and testing datasets respective class labels are converted to one-hot encoded vector. \n")
        return

# if __name__ == '__main__':
#     data = SensorDataLoader(get_args())
