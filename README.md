# Human Activity Recognition using Convolutional Neural Networks with Tensorflow 2 and Keras

More information about the project setup will be available soon..

## Resources

The project uses the famous *Human Activity Recognition Dataset*. Please, read the information link provided below to have a good indication of the dataset format.

* information: [Human Activity Recognition Using Smartphones DataSet](http://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
* download: [UCI HAR Dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip)

## Setup

### Environment

Requires Python 3.10 - 3.12 and TensorFlow 2.18+ (Keras 3).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -r requirements.lock (exact tested versions)
```

Optional:
* `pip install tensorflow-metal` for GPU acceleration on Apple Silicon.
* `brew install graphviz` (or `apt install graphviz`) to export the model design diagram; without it the export is skipped.

### Run

```bash
python nn_mains/har_main.py -e 50                 # train, evaluate and save the model (.keras)
python nn_mains/har_main.py -m load -t true       # load the saved model and evaluate on the test set
```

Models are saved in the native Keras format (`nn_experiments/<exp_name>/saved_models/<exp_name>_model.keras`).
Models saved as `.h5` with older TensorFlow versions must be retrained.

### Dataset

Download the file you can find in the [Resources](#Resources) section, unzip the file and put the folder inside the datasets folder. If you want to change its default name, please update the `data_folder` value inside the `nn_setup.yml` file where the NN reads the data for processing.

## Related Work

The Object-Oriented Programming design is deployed for the implementation of this project. It is also based on the following related works, but it extends them in the Neural Network Design and the functionalities that previous authors implemented:

* https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
* https://github.com/SantoshPattar/ConvNet-OOP
* https://github.com/MrGemy95/Tensorflow-Project-Template

## Future Work

* More comments will be added to the code.
* The logs creation functionality will be added soon.

More information will be available soon..
