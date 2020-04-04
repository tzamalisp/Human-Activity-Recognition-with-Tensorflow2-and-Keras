#!/usr/bin/env python3
"""
Execution Flow for the PAT experiment.
"""
from datetime import datetime
from pprint import pprint
# Reproduce results by seed-ing the random number generator.
from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
import tensorflow as tf
tf.random.set_seed(2)

# import scripts from other folders
import os
import sys
# sys.path.append('../')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from helper_functions.read_yaml import ReadYml
from helper_functions.libraries_checker import LibrariesChecker
from nn_utils.process_configuration import ConfigurationParameters
from nn_data_loader.har_loader import SensorDataLoader
from nn_model.nn_model_har import HarModel
from nn_utils.model_utils_har import Report
from nn_utils.process_argument_har import get_args
from nn_utils.softmax_processing import SoftmaxResults
from nn_model.nn_model_predictions import Predict


# read yml configuration for nn setup
nn_setup_yml = ReadYml('nn_setup.yml')
nn_setup_conf = nn_setup_yml.load_yml()
print('YML conf:')
pprint(nn_setup_conf)

def main():
	print('Time of NN train execution: {}'.format(datetime.now()))
	print()
	try:

		# Capture the command line arguments from the interface script.
		args = get_args()

		# Parse the configuration parameters for the ConvNet Model.
		config = ConfigurationParameters(args)

	except:
		print('Missing or invalid arguments!')
		exit(0)

	# check the libraries compatibility
	if config.config_namespace.libraries_checker == True:
		check_libraries = LibrariesChecker()
		check_libraries.checker()
	else:
		print('The libraries checking functionality is OFF.')
		print()

	# Load the dataset from the library, process and print its details.
	dataset = SensorDataLoader(config)

	# Test the loaded dataset by displaying the first image in both training and testing dataset.
	dataset.display_data_element('train_data', 1)
	dataset.display_data_element('test_data', 1)

	# train and save the model / load  the model for preditions
	if config.config_namespace.mode == 'save':
		# Construct, compile, train and evaluate the ConvNet Model.
		model = HarModel(config, dataset)

		# Save the ConvNet model to the disk.
		model.save_model()

	elif config.config_namespace.mode == 'load':
		# load the saved ConvNet model from the disk
		print('Loading the model..')
		model = Predict(config, dataset)
		model.load_model()
		if config.config_namespace.evaluate_test == 'true':
			model.evaluate_model()
			report = Report(config, model)
			report.model_classification_report()
			report.plot_confusion_matrix()
	else:
		print('Please, give a valid value (save / load)')
		print('or give a valid value for test set evaluation (true / false)')

	# Model evaluation Report + Softmax Results
	if config.config_namespace.mode == 'save':
		# Generate graphs, classification report, confusion matrix.
		report = Report(config, model)
		report.plot()
		report.model_classification_report()
		report.plot_confusion_matrix()
		print()
		# Generate the softmax exports in probabilities per class
		softmax = SoftmaxResults(config=config,
								model=model,
								dataset=dataset)
		df_softmax = softmax.get_softmax_results()
		print('Softmax results Dataframe information:')
		df_softmax.info()


if __name__ == '__main__':
	main()
