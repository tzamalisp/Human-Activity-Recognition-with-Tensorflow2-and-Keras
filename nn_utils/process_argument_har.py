"""
Process the json configuration file.

Configuration file holds the parameters to intialize the CNN model.
These files are located in configaration_files folder.
"""
import argparse
import json
from pprint import pprint
# import scripts from other folders
import os
import sys
# sys.path.append('../')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from helper_functions.find_create_directory import NnConfDirectory
from helper_functions.read_yaml import ReadYml

def get_args():
	"""
	Get arguments from the command line.

	:param none
	:return none
	:raises none
	"""
	# read yml configuration for nn setup
	nn_setup_yml = ReadYml('nn_setup.yml')
	nn_setup_conf = nn_setup_yml.load_yml()

	# default NN JSONconfiguration path
	path_obj = NnConfDirectory()
	path_conf_default =  path_obj.nn_conf_directory() + nn_setup_conf['nn_conf_file']
	print('Config file path: {}'.format(path_conf_default))

	parser = argparse.ArgumentParser( description = __doc__ )

	# Configuration file path argument.
	parser.add_argument(
						'-c', '--config',
						metavar = 'C',
						help = 'The Configuration file',
						default = path_conf_default,
						required = False
					)

	# Epoch size argument (overrides "num_epochs" of the configuration file).
	parser.add_argument(
						'-e', '--epoch',
						metavar = 'E',
						type = int,
						help = 'Number of epochs for training the model (default: num_epochs from the config file)',
						default = None,
						required = False
					)

	# Mode argument (overrides "mode" of the configuration file).
	parser.add_argument(
						'-m', '--mode',
						metavar = 'M',
						choices = ['save', 'load'],
						help = 'save: train/validate and save the model, load: load the model (default: mode from the config file)',
						default = None,
						required = False
					)

	# Test set evaluation (overrides "evaluate_test" of the configuration file).
	parser.add_argument(
						'-t', '--testevaluation',
						metavar = 'T',
						choices = ['true', 'false'],
						help = 'Evaluate on the test set in load mode: true / false (default: evaluate_test from the config file)',
						default = None,
						required = False
					)

	# Convert to dictonary.
	args = vars(parser.parse_args())
	if args['testevaluation'] is not None:
		args['testevaluation'] = args['testevaluation'] == 'true'

	if args['config'] == path_conf_default:
		print('Using default configuration file.')
	else:
		print('Using configurations from file:', args['config'])

	for key, name in (('epoch', 'epoch size'), ('mode', 'mode'), ('testevaluation', 'test set evaluation')):
		if args[key] is None:
			print('Using {} from the configuration file.'.format(name))
		else:
			print('Using {} from the command line: {}'.format(name, args[key]))

	return args
