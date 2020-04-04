"""
Parse the JSON configuration file of the experiment.

Configuration file holds the parameters to intialize the ConvNet model.
These files are located in configaration_files folder.
"""
import json
import os
from bunch import Bunch
# Main Application directory
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
main_app_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))

class ConfigurationParameters:

	def __init__(self, args):
		"""
		Intialize the data members.

		:param json_file: Path to the JSON configuration file.
		:return none
		:raises none
		"""

		self.args = args

		json_file = self.args['config']

		# Parse the configurations from the config json file provided.
		with open(json_file, 'r') as config_file:
			self.config_dictionary = json.load(config_file)

		# Convert the dictionary to a namespace using bunch library.
		self.config_namespace = Bunch(self.config_dictionary)

		# Update the command line arguments in the configuration namespace.
		self.update_namespace()

		# Process the configuration parameters.
		self.process_config()

		return

	def update_namespace(self):
		"""
		Updates the value of JSON keys received from the command line to the namespace file.

		:param none
		:return none
		:raises none
		"""

		# Update epoch size.
		if 'epoch' in self.args.keys():
			self.config_namespace.num_epochs = int(self.args['epoch'])

		# Update mode (save/load).
		if 'mode' in self.args.keys():
			self.config_namespace.mode = self.args['mode']

		# Update mode (save/load).
		if 'testevaluation' in self.args.keys():
			self.config_namespace.evaluate_test = self.args['testevaluation']

		return

	def process_config(self):
		"""
		Processes the configuration parameters of the ConvNet experiment.

		:param none
		:return none
		:raises none
		"""

		# Saved-Model directory.
		self.config_namespace.saved_model_dir = os.path.join(main_app_path, "nn_experiments", self.config_namespace.exp_name, "saved_models/")

		# Graph directory.
		self.config_namespace.graph_dir = os.path.join(main_app_path, "nn_experiments", self.config_namespace.exp_name, "graphs/")

		# Image directory.
		self.config_namespace.image_dir = os.path.join(main_app_path, "nn_experiments", self.config_namespace.exp_name, "images/")

		# DataFrame directory.
		self.config_namespace.df_dir = os.path.join(main_app_path, "nn_experiments", self.config_namespace.exp_name, "dataframes/")

		# Classification Report directory.
		self.config_namespace.cr_dir = os.path.join(main_app_path, "nn_experiments", self.config_namespace.exp_name, "class_reports/")

		# Create the above directories.
		self.create_dirs([self.config_namespace.graph_dir,
						self.config_namespace.image_dir,
						self.config_namespace.saved_model_dir,
						self.config_namespace.df_dir,
						self.config_namespace.cr_dir])

		return

	def create_dirs(self, dirs):
		"""
		Creates a directory structure for Graphs and Images generated during the run of the experiment.

		:param dirs: a list of directories to create if these directories are not found
		:return exit_code: 0:success -1:failed
		:raises none
		"""

		try:
			for d in dirs:
				if not os.path.exists(d):
					os.makedirs(d)
			return 0

		except Exception as err:
			print("Creating directories error: {0}".format(err))
			exit(-1)
