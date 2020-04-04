import os
import yaml
import pathlib
import glob
# data analysis
import pandas as pd
import numpy as np
# data visualization
import matplotlib
import seaborn as sns
# NN
import tensorflow as tf


class LibrariesChecker:
    @staticmethod
    def checker():
        print('---- CHECK THE LIBRARIES INSTALLED ----')
        print('File handling:')
        print('Yaml version: {}'.format(yaml.__version__))
        print()
        print('Analysis and Visualization:')
        print('Pandas version: {}'.format(pd.__version__))
        print('NumPy version: {}'.format(np.__version__))
        print('MatplotLib version: {}'.format(matplotlib.__version__))
        print('Seaborn version: {}'.format(sns.__version__))
        print()
        print('NN Models:')
        print('Tensorflow version: {}'.format(tf.__version__))
        print()
