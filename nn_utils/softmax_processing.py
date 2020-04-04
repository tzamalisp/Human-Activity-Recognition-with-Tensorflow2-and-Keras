"""
Utility to generate the softmax exports in instance's probabilities per class
"""
import os
import tensorflow
import pandas

# Main Application directory
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
main_app_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))


class SoftmaxResults:

    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.df = pandas.DataFrame()

        return

    def get_softmax_results(self):
        """
        The test data is passed to the model and we get the results.
        training = False is set so that we evaluate the model's predictions.
        """
        # predictions = self.model(self.dataset.test_data, training=False)
        # we get the logits and iterate over the predictions
        for i, logits in enumerate(self.model.predictions):
            # get the probabilities from the logits, from the softmax layer.
            p = logits
            # then, multiply by 100 to get the probabilities as a percentage.
            p = 100 * p
            # turn the NumPy array to a list for easier manipulation
            smaxlist = p.tolist()
            # index is the class index, so we have to get the index of the max value of softmax.
            index = smaxlist.index(max(smaxlist))
            # create a df based on the probabilities matrix and column names.
            # [p] is used because p is a numpy array.
            df = pandas.DataFrame([p], columns=self.config.config_namespace.class_names)
            # add a column to the df with the class name (classification result)
            df['CLASS'] = self.config.config_namespace.class_names[index]
            # append the row to the final DataFrame
            self.df = self.df.append(df)
        # create a new column with the max value of the DataFrame (efficient)
        self.df['ARGMAX'] = self.df.max(axis=1)
        if self.df.isnull().values.any() == False and self.df.isna().values.any() == False:

            if self.config.config_namespace.mode == 'save':
            	df_file_name = "save_train_val_softmax_exports.csv"
            elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'true':
            	df_file_name = "load_test_softmax_exports.csv"
            elif self.config.config_namespace.mode == 'load' and self.config.config_namespace.evaluate_test == 'false':
            	df_file_name = "load_pred_softmax_exports.csv"

            df_path = os.path.join(main_app_path, self.config.config_namespace.df_dir, df_file_name)
            self.df.to_csv(df_path, index=False)
            print('Softmax results Dataframe saved sucessfully to:')
            print(df_path)
            return self.df
        else:
            print('Null values found in the dataset, please check your processing scripts.')
            return None
