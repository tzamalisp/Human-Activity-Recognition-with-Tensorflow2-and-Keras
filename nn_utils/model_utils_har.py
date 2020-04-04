"""
Utility to generate classification report, confusion matrix and
graph for loss and accuracy for the ConvNet model.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import itertools
import os
import sys
from datetime import datetime

# Main Application directory
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
main_app_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))

class Report():

	def __init__(self, config, model):
		"""
		Constructor to initialize the Report.

		:param config: the json configuration namespace.
		:param model: the ConvNet model.
		:return: none
		:raises: none
		"""

		self.config = config
		self.model = model

		return

	def plot(self):
		"""
		Plot loss and accuracy for the training and validation set of the ConvNet model.

		:param none
		:return none
		:raises none
		"""
		if self.config.config_namespace.validation_split == True:
			validation_method = "Validation: validation_plit"
		elif self.config.config_namespace.validation_split == False:
			validation_method = "Validation: Test data"

		loss_list = [s for s in self.model.history.history.keys() if 'loss' in s and 'val' not in s]
		val_loss_list = [s for s in self.model.history.history.keys() if 'loss' in s and 'val' in s]
		acc_list = [s for s in self.model.history.history.keys() if 'acc' in s and 'val' not in s]
		val_acc_list = [s for s in self.model.history.history.keys() if 'acc' in s and 'val' in s]

		if len(loss_list) == 0:
			print('Loss is missing in history')
			return

        # As loss always exists
		epochs = range(1, len(self.model.history.history[loss_list[0]]) + 1)

       	# Loss graph.
		plt.figure(1)

		for l in loss_list:
			plt.plot(epochs,
					self.model.history.history[l],
					'b',
					label = 'Training loss (' + str(str(format(self.model.history.history[l][-1],'.5f')) + ')')
					)

		for l in val_loss_list:
			plt.plot(epochs,
					self.model.history.history[l],
					'g',
					label = 'Validation loss (' + str (str(format(self.model.history.history[l][-1],'.5f')) + ')')
					)

		plt.title('Loss per Epoch - {}'.format(validation_method))
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()

		# Save the plot to disk.
		if self.config.config_namespace.mode == 'save':
			loss_file_name = "save_train_val_loss.png"
		elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'true':
			loss_file_name = "load_test_loss.png"
		elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'false':
			loss_file_name = "load_pred_loss.png"

		loss_path = os.path.join(main_app_path, self.config.config_namespace.graph_dir, loss_file_name)

		if(self.config.config_namespace.save_plots == 'true'):
			plt.savefig(loss_path, bbox_inches='tight')
		else:
			plt.show()

		plt.close()

		# Accuracy graph.
		plt.figure(2)
		for l in acc_list:
			plt.plot(epochs,
					self.model.history.history[l],
					'b',
					label = 'Training accuracy (' + str(format(self.model.history.history[l][-1],'.5f')) + ')'
					)

		for l in val_acc_list:
			plt.plot(epochs,
					self.model.history.history[l],
					'g',
					label = 'Validation accuracy (' + str(format(self.model.history.history[l][-1],'.5f')) + ')'
					)

		plt.title('Accuracy per Epoch - {}'.format(validation_method))
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend()

		# Save the plot to disk.
		if self.config.config_namespace.mode == 'save':
			acc_file_name = "save_train_val_accuracy.png"
		elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'true':
			acc_file_name = "load_test_accuracy.png"
		elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'false':
			acc_file_name = "load_pred_accuracy.png"

		acc_path = os.path.join(main_app_path, self.config.config_namespace.graph_dir, acc_file_name)
		if(self.config.config_namespace.save_plots == 'true'):
			plt.savefig(acc_path, bbox_inches='tight')
		else:
			plt.show()

		plt.close()

		return

	def model_classification_report(self):
		"""
		Generate classification report of the ConvNet model here.

		:param none
		:return none
		:raises none
		"""

		predicted_classes = np.argmax(self.model.predictions, axis = 1)
		print('Accuracy: '+ str(accuracy_score(self.model.dataset.test_labels, predicted_classes)))
		print('Classification Report')
		print('------------------------------------------------------------------')
		target_names = ['Class {}'.format(i) for i in range(self.config.config_namespace.num_classes)]
		print(
				classification_report(
					self.model.dataset.test_labels,
					predicted_classes,
					target_names = target_names
				)
			)
		# Save the Classification Report to disk.
		if self.config.config_namespace.mode == 'save':
			cr_file_name = "save_train_val_class_report.txt"
		elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'true':
			cr_file_name = "load_test_class_report.txt"
		elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'false':
			cr_file_name = "load_pred_class_report.txt"

		if self.config.config_namespace.validation_split == True:
			validation_method = "Validation method: Validation split on Train data --> Evaluate on Test"
		elif self.config.config_namespace.validation_split == False:
			validation_method = "Validation method: Validation on Test data --> Evaluate on Test"

		cr_path = os.path.join(main_app_path, self.config.config_namespace.cr_dir, cr_file_name)
		with open(cr_path, 'w+') as file:
			file.write('Classification Report:')
			file.write('\n')
			file.write('Time exported: {}'.format(datetime.now()))
			file.write('\n')
			file.write(validation_method)
			file.write('\n')
			file.write('------------------------------------------------------------------')
			file.write('\n')
			file.write(
				classification_report(
					self.model.dataset.test_labels,
					predicted_classes,
					target_names = target_names
				)
			)
			file.close()
		print('Classification Report saved successfully!')
		return

	def plot_confusion_matrix(self):
		"""
		Generate and plot the classification confusion matrix.

		:param none
		:return none
		:raises none
		"""
		if self.config.config_namespace.validation_split == True:
			validation_method = "Validation: validation_plit"
		elif self.config.config_namespace.validation_split == False:
			validation_method = "Validation: Test data"

		predicted_classes = np.argmax(self.model.predictions, axis = 1)
		target_names = ['Class {}'.format(i) for i in range(self.config.config_namespace.num_classes)]

		title = 'Confusion matrix - {}'.format(validation_method)
		cm = confusion_matrix(self.model.dataset.test_labels, predicted_classes)
		print(title)
		print('------------------------------------------------------------------')
		print(cm)

		plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation = 45)
		plt.yticks(tick_marks, target_names)

		fmt = 'd'
		thresh = cm.max() / 2

		for i, j in itertools.product(
										range(cm.shape[0]),
										range(cm.shape[1])
									): plt.text(
												j,
												i,
												format( cm[i, j], fmt ),
												horizontalalignment = 'center',
												color = 'white' if cm[i, j] > thresh else 'black'
											)

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')

		# Save the plot to disk.
		if self.config.config_namespace.mode == 'save':
			cm_file_name = "save_train_val_confusion_matrix.png"
		elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'true':
			cm_file_name = "load_test_confusion_matrix.png"
		elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'false':
			cm_file_name = "load_pred_confusion_matrix.png"

		cm_path = os.path.join(main_app_path, self.config.config_namespace.graph_dir, cm_file_name)

		if(self.config.config_namespace.save_plots == 'true'):
			plt.savefig(cm_path, bbox_inches='tight')
		else:
			plt.show()

		plt.close()

		return
