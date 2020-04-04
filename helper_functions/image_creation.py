import os
# data analysis
import pandas as pd
import numpy as np
import seaborn as sns
# data visualization
import matplotlib
import matplotlib.pyplot as plt

# import scripts from other folders
import sys
# sys.path.append('../')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from helper_functions.find_create_directory import FindDirectory, LogDirectory, ExportsDirectory


""" Class to save the figure created """

class SaveFig:
    def __init__(self, fig_id, images_directory):
        self.fig_id = fig_id
        self.images_directory = images_directory


    # internal function to plot data analysis and save the figures
    def save_fig(self, tight_layout=True, fig_extension="png", resolution=300):
        app_dir = FindDirectory(self.images_directory)
        path = app_dir.create_directory()
        print('Path directory - images: {}'.format(path))
        path = os.path.join(path, self.fig_id + "." + fig_extension)
        print("Saving figure: {}".format(self.fig_id))
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)
        plt.close()
