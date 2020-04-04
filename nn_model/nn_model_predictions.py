import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from bson.objectid import ObjectId
# import scripts from other folders
import os
import sys
# sys.path.append('../')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from nn_base.nn_base_model_har import BaseModel

# main path of the app
main_app_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))

class Predict:
    def __init__(self, config, dataset):
        # Configuaration parameters.
        self.config = config

        # Testing datasets or data for prediction.
        self.dataset = dataset

        # initial model
        self.cnn_model = None

        # Saved model path.
        model_prefix = self.config.config_namespace.exp_name
        self.saved_model_path = os.path.join(main_app_path, self.config.config_namespace.saved_model_dir, "{}_model.h5".format(model_prefix))

        # Evaluation scores.
        self.scores = []

        # Predicted class labels.
        self.predictions = np.array([])

    def load_model(self):
        """
        Loads the saved model from the disk.
        :param none
        :return none
        :raises NotImplementedError: Implement this method.
        """
        if os.path.exists(self.saved_model_path):
            self.cnn_model = keras.models.load_model(self.saved_model_path)
            self.cnn_model.load_weights(self.saved_model_path)
            print("ConvNet model loaded from the path: ", self.saved_model_path, "\n")

        elif self.cnn_model is None:
            raise Exception("ConvNet model not configured and trained !")

        return

    def evaluate_model(self):
        """
        Evaluate the ConvNet model on Test data.
        :param none
        :return none
        :raises none
        """

        class_names = self.config.config_namespace.class_names
        print('Classes:', class_names)
        print()
        self.predictions = self.cnn_model.predict(self.dataset.test_data)
        # print(self.predictions.round(2))

        self.scores = self.cnn_model.evaluate(x=self.dataset.test_data,
                                              y=self.dataset.test_label_one_hot,
                                              verbose=self.config.config_namespace.evaluate_verbose
                                              )

        print("Test loss: ", self.scores[0])
        print("Test accuracy: ", self.scores[1])

        return

    def evaluate_predictions():
        """
        Data load, evaluate predictions, and put on DataFrame with Softmax
        :param none
        :return none
        :raises none
        """
