import yaml
import os
# import scripts from other folders
import sys
# sys.path.append('../')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from helper_functions.find_create_directory import FindDirectory, LogDirectory, ExportsDirectory


""" Read the Yml file """


class ReadYml:
    def __init__(self, yml_file_name):
        self.yml_file_name = yml_file_name

    def load_yml(self):
        app_dir = FindDirectory('configs')
        path = app_dir.create_directory()
        # load data analysis yml data
        with open(path + '/' + self.yml_file_name) as yml_file:
            yml_data = yaml.load(yml_file, Loader=yaml.FullLoader)
        return yml_data
