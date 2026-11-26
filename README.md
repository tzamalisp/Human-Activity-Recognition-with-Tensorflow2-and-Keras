# Human Activity Recognition using Convolutional Neural Networks with Tensorflow 2 and Keras

More information about the project setup will be available soon..

## Resources

The project uses the famous *Human Activity Recognition Dataset*. Please, read the information link provided below to have a good indication of the dataset format.

* information: [Human Activity Recognition Using Smartphones DataSet](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
* download: handled by `helper_functions/download_dataset.py` (see [Dataset](#dataset))

## Setup

### Environment

Requires Python 3.10 - 3.12 and TensorFlow 2.18+ (Keras 3).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -r requirements.lock (exact tested versions)
```

> **Note:** since TensorFlow 2.21, TensorBoard is no longer installed together with TensorFlow (versions 2.18 - 2.20 still installed it). This project does not need it; if you want to use it, install it separately with `pip install tensorboard`.

Optional:
* `pip install tensorflow-metal` for GPU acceleration on Apple Silicon.
* `pip install tensorboard` to visualize training logs.
* `brew install graphviz` (or `apt install graphviz`) to export the model design diagram; without it the export is skipped.

### Run

```bash
python nn_mains/har_main.py                       # train, evaluate and save the model (.keras)
python nn_mains/har_main.py -m load -t true       # load the saved model and evaluate on the test set
```

All settings come from `nn_configuration_files/har_config.json`. The command line options `-e/--epoch`, `-m/--mode` (`save`/`load`) and `-t/--testevaluation` (`true`/`false`) override `num_epochs`, `mode` and `evaluate_test` only when given; `-c/--config` selects another configuration file.

Models are saved in the native Keras format (`nn_experiments/<exp_name>/saved_models/<exp_name>_model.keras`).
Models saved as `.h5` with older TensorFlow versions must be retrained.

### Dataset

The dataset is **downloaded automatically** on the first training run (`auto_download: true` in `configs/nn_setup.yml`). To download it manually:

```bash
python helper_functions/download_dataset.py
```

The script downloads the archive from this repository's [`dataset-v1.0` release](https://github.com/tzamalisp/Human-Activity-Recognition-with-Tensorflow2-and-Keras/releases/tag/dataset-v1.0), falls back to the official UCI repository, verifies the SHA-256 checksum and extracts it to `datasets/UCI HAR Dataset/`. Options: `--force` to download again, `--zip <file>` to use a local copy, `--folder <name>` for a different folder name (it must match `data_folder` in `configs/nn_setup.yml`).

#### Dataset license and citation

The dataset is not covered by this repository's license. Its authors require that use in publications is acknowledged by citing:

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium, 24-26 April 2013.

The dataset is distributed AS-IS and commercial use is prohibited.

## Related Work

The Object-Oriented Programming design is deployed for the implementation of this project. It is also based on the following related works, but it extends them in the Neural Network Design and the functionalities that previous authors implemented:

* https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
* https://github.com/SantoshPattar/ConvNet-OOP
* https://github.com/MrGemy95/Tensorflow-Project-Template

## Future Work

* More comments will be added to the code.
* The logs creation functionality will be added soon.

More information will be available soon..
