import time
import tensorflow as tf
from tensorflow import keras
# import scripts from other folders
import os
import sys
# sys.path.append('../')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from nn_base.nn_base_model_har import BaseModel

# Main Application directory
main_app_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))

class HarModel(BaseModel):

    def __init__(self, config, dataset):
        """
        Constructor to initialize the ConvNet for PAT wearables' sensors dataset.
        :param config: the JSON configuration namespace.
        :param dataset: Training and testing datasets.
        :return none
        :raises none
        """

        super().__init__(config, dataset)
        return

    def define_model(self):
        """
        Construct the ConvNet model.
        :param none
        :return none
        :raises none
        """

        # TODO: two ways of defining
        # 1) intialize an array with all layers(remember there is layer size parameter in JSON) and pass it to Sequential constructor.
        # 2) add the layers to the model through "add" method of Sequential class.
        model_design_name = ''
        if self.config.config_namespace.model_type == 'Sequential':
            print("The Keras ConvNet model type used for this experiment is: ", self.config.config_namespace.model_type)
            self.cnn_model = self.define_sequential_model()
            model_design_name = 'model_design_{}.png'.format(self.config.config_namespace.model_type)
        else:
            # TODO: handle functional model here.
            # self.cnn_model = Model()
            self.define_functional_model()
            model_design_name = 'model_design_{}.png'.format(self.config.config_namespace.model_type)
        # Summary of the ConvNet model.
        print('Summary of the model:')
        self.cnn_model.summary()


        # save the model design
        model_design_path = os.path.join(main_app_path, self.config.config_namespace.image_dir, model_design_name)
        keras.utils.plot_model(self.cnn_model, model_design_path, show_shapes=True)
        return

    def define_sequential_model(self):
        """
        Design a sequential ConvNet model.
        :param none
        :return cnn_model: The ConvNet sequential model.
        :raises none
        """
        n_timesteps, n_features = self.dataset.train_data.shape[1], self.dataset.train_data.shape[2],
        print('for input_shape in 1st layer --> input_shape=(n_timesteps, n_features)')
        print('n_timesteps: {}'.format(n_timesteps))
        print('n_features: {}'.format(n_features))

        # define the Neural Network Layers
        self.cnn_model = tf.keras.models.Sequential()

        # Conv1D layer - Input layer --> 01
        self.cnn_model.add(keras.layers.Conv1D(filters=self.config.config_namespace.oned_no_of_filters_l1,  # filters = 64
                                            kernel_size=self.config.config_namespace.oned_kernel_row,  # kernel_size = 3
                                            activation=self.config.config_namespace.oned_conv_activation_l1,  # activation = 'relu'
                                            input_shape=(n_timesteps, n_features),  # n_timesteps = 128, n_features = 9
                                            padding=self.config.config_namespace.oned_padding,  # oned_padding = 'valid' --> default
                                            strides=self.config.config_namespace.oned_stride_size  # strides = 1 --> default
                                            )
                                        )

        # Leaky ReLu Layer --> 01
        if self.config.config_namespace.leaky_relu == True:
            self.cnn_model.add(keras.layers.LeakyReLU(alpha=self.config.config_namespace.relu_alpha))

        # MaxPooling1D layer --> 01
        self.cnn_model.add(keras.layers.MaxPooling1D(pool_size=self.config.config_namespace.oned_pool_size_row, # pool_size = 2
                                                    padding=self.config.config_namespace.oned_padding # oned_padding = 'valid' --> default
                                                    )
                                                )

        # Dropout layer --> 01
        if self.config.config_namespace.dropout == True:
            self.cnn_model.add(keras.layers.Dropout(self.config.config_namespace.oned_dropout_probability_l1)) # probability = 0.5

        # Batch Normalization Layer - Applied just efore the activation functions
        if self.config.config_namespace.batch_normalization == True:
            self.cnn_model.add(keras.layers.BatchNormalization())

        # Conv1D layer  --> 02
        self.cnn_model.add(keras.layers.Conv1D(filters=self.config.config_namespace.oned_no_of_filters_l2,  # filters = 64
                                            kernel_size=self.config.config_namespace.oned_kernel_row,  # kernel_size = 3
                                            activation=self.config.config_namespace.oned_conv_activation_l2,  # activation = 'relu'
                                            padding=self.config.config_namespace.oned_padding,  # oned_padding = 'valid' --> default
                                            strides=self.config.config_namespace.oned_stride_size  # strides = 1 --> default
                                            )
                                        )
        # Leaky ReLu Layer --> 01
        if self.config.config_namespace.leaky_relu == True:
            self.cnn_model.add(keras.layers.LeakyReLU(alpha=self.config.config_namespace.relu_alpha))

        # MaxPooling1D layer --> 01
        self.cnn_model.add(keras.layers.MaxPooling1D(pool_size=self.config.config_namespace.oned_pool_size_row, # pool_size = 2
                                                    padding=self.config.config_namespace.oned_padding # oned_padding = 'valid' --> default
                                                    )
                                                )

        # Dropout layer --> 01
        if self.config.config_namespace.dropout == True:
            self.cnn_model.add(keras.layers.Dropout(self.config.config_namespace.oned_dropout_probability_l1)) # probability = 0.5

        # Batch Normalization Layer - Applied just efore the activation functions
        if self.config.config_namespace.batch_normalization == True:
            self.cnn_model.add(keras.layers.BatchNormalization())

        # Conv1D layer  --> 02
        self.cnn_model.add(keras.layers.Conv1D(filters=self.config.config_namespace.oned_no_of_filters_l2,  # filters = 64
                                            kernel_size=self.config.config_namespace.oned_kernel_row,  # kernel_size = 3
                                            activation=self.config.config_namespace.oned_conv_activation_l2,  # activation = 'relu'
                                            padding=self.config.config_namespace.oned_padding,  # oned_padding = 'valid' --> default
                                            strides=self.config.config_namespace.oned_stride_size  # strides = 1 --> default
                                            )
                                        )
        # Leaky ReLu Layer --> 01
        if self.config.config_namespace.leaky_relu == True:
            self.cnn_model.add(keras.layers.LeakyReLU(alpha=self.config.config_namespace.relu_alpha))

        # MaxPooling1D layer --> 01
        self.cnn_model.add(keras.layers.MaxPooling1D(pool_size=self.config.config_namespace.oned_pool_size_row, # pool_size = 2
                                                    padding=self.config.config_namespace.oned_padding # oned_padding = 'valid' --> default
                                                    )
                                                )

        # Dropout layer --> 01
        if self.config.config_namespace.dropout == True:
            self.cnn_model.add(keras.layers.Dropout(self.config.config_namespace.oned_dropout_probability_l1)) # probability = 0.5

        # Flatten layer --> 01
        self.cnn_model.add(keras.layers.Flatten())

        # Batch Normalization Layer - Applied just efore the activation functions
        if self.config.config_namespace.batch_normalization == True:
            self.cnn_model.add(keras.layers.BatchNormalization())

        # Dense layer --> 01
        self.cnn_model.add(keras.layers.Dense(units=self.config.config_namespace.oned_no_of_units_l1,  # units = 100
                                 activation=self.config.config_namespace.oned_dense_activation_l1  # activation = "relu"
                                 )
                           )

        # Leaky ReLu Layer --> 01
        if self.config.config_namespace.leaky_relu == True:
            self.cnn_model.add(keras.layers.LeakyReLU(alpha=self.config.config_namespace.relu_alpha))

        # Dropout layer --> 01
        if self.config.config_namespace.dropout == True:
            self.cnn_model.add(keras.layers.Dropout(self.config.config_namespace.oned_dropout_probability_l1)) # probability = 0.5

        # Batch Normalization Layer - Applied just efore the activation functions
        if self.config.config_namespace.batch_normalization == True:
            self.cnn_model.add(keras.layers.BatchNormalization())

        # Dense layer - Output layer  --> 02
        self.cnn_model.add(keras.layers.Dense(self.dataset.no_of_classes,
                                            activation=self.config.config_namespace.oned_dense_activation_l2)  # activation = softmax
                                            )


        return self.cnn_model

    def define_functional_model(self):
        """
        Define (construct) a functional ConvNet model.
        :param none
        :return cnn_model: The ConvNet sequential model.
        :raises none
        """

        print("yet to be implemented\n")
        return

    def compile_model(self):
        """
        Configure the ConvNet model.
        :param none
        :return none
        :raises none
        """

        self.cnn_model.compile(loss=self.config.config_namespace.compile_loss,
                               optimizer=self.config.config_namespace.compile_optimizer,
                               metrics=[self.config.config_namespace.compile_metrics1]
                               )
    # FIXME; check if dataset is need in the whole class, if not needed just pass it to required functions. (see grid search)
    def fit_model(self):
        """
        Train the ConvNet model.
        :param none
        :return none
        :raises none
        """

        start_time = time.time()

        if self.config.config_namespace.validation_split == True:
            print('Training phase uses "validation_plit" parameter with a ratio retrieved from the configuration file."')
            if (self.config.config_namespace.save_model == "true"):
                print("Training phase under progress, trained ConvNet model will be saved at path", self.saved_model_path,
                      " ...\n")
                self.history = self.cnn_model.fit(x=self.dataset.train_data,
                                                  y=self.dataset.train_label_one_hot,
                                                  batch_size=self.config.config_namespace.batch_size,
                                                  epochs=self.config.config_namespace.num_epochs,
                                                  callbacks=self.callbacks_list_sm,
                                                  verbose=self.config.config_namespace.fit_verbose,
                                                  validation_split=self.config.config_namespace.validation_split_ratio
                                                  )
            else:
                print("Training phase under progress ...\n")
                self.history = self.cnn_model.fit(x=self.dataset.train_data,
                                                  y=self.dataset.train_label_one_hot,
                                                  batch_size=self.config.config_namespace.batch_size,
                                                  epochs=self.config.config_namespace.num_epochs,
                                                  callbacks=self.callbacks_list_es,
                                                  verbose=self.config.config_namespace.fit_verbose,
                                                  validation_split=self.config.config_namespace.validation_split_ratio
                                                  )

        elif self.config.config_namespace.validation_split == False:
            print('Training phase uses "validation_data" parameter, and handles the test data."')
            if (self.config.config_namespace.save_model == "true"):
                print("Training phase under progress, trained ConvNet model will be saved at path", self.saved_model_path,
                      " ...\n")
                self.history = self.cnn_model.fit(x=self.dataset.train_data,
                                                  y=self.dataset.train_label_one_hot,
                                                  batch_size=self.config.config_namespace.batch_size,
                                                  epochs=self.config.config_namespace.num_epochs,
                                                  callbacks=self.callbacks_list,
                                                  verbose=self.config.config_namespace.fit_verbose,
                                                  validation_data=(self.dataset.test_data, self.dataset.test_label_one_hot)
                                                  )
            else:
                print("Training phase under progress ...\n")
                self.history = self.cnn_model.fit(x=self.dataset.train_data,
                                                  y=self.dataset.train_label_one_hot,
                                                  batch_size=self.config.config_namespace.batch_size,
                                                  epochs=self.config.config_namespace.num_epochs,
                                                  verbose=self.config.config_namespace.fit_verbose,
                                                  validation_data=(self.dataset.test_data, self.dataset.test_label_one_hot)
                                                  )
        else:
            print('Not valid value is declared for the validation process. Set <true> for "validation_split" or <false> for "validation_data".')

        end_time = time.time()

        self.train_time = end_time - start_time
        print("The model took %0.3f seconds to train.\n" % self.train_time)

        return

    def evaluate_model(self):
        """
        Evaluate the ConvNet model.
        :param none
        :return none
        :raises none
        """

        self.scores = self.cnn_model.evaluate(x=self.dataset.test_data,
                                              y=self.dataset.test_label_one_hot,
                                              verbose=self.config.config_namespace.evaluate_verbose
                                              )

        print("Test loss: ", self.scores[0])
        print("Test accuracy: ", self.scores[1])

        return

    def predict(self):
        """
        Predict the class labels of testing dataset.
        :param none
        :return none
        :raises none
        """

        self.predictions = self.cnn_model.predict(x=self.dataset.test_data,
                                                  verbose=self.config.config_namespace.predict_verbose
                                                  )

        return
