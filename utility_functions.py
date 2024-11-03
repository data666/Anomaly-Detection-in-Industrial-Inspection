import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import os 

import logging
logging.basicConfig(
    level=logging.INFO
)
logger = logging.getLogger(__name__)

os.chdir('..')
DATA_PATH = './data/project_data.mat'

def load_data(fpath: str = DATA_PATH):
    """
    Load dataset splits and data from a path.

    Args:
        fpath (str): File Path

    Returns:
        tuples(np.array, np.array...): Numpy array of data sets.
    """
    try:
        data = scio.loadmat(fpath)
    except FileNotFoundError:
        logger.info(f"Error: The path '{fpath}' cannot be found.")
        raise e
    except Exception as e:
        logger.info(f"An unexpected error occurred: {e}")
        raise e

    try:
        x_set = np.concatenate((data['x_train'], data['x_test']))
        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        sample_size = x_set.shape[0]
        train_size = x_train.shape[0]
        test_size = x_test.shape[0]
        
        train_pct = train_size/sample_size
        test_pct = test_size/sample_size
        logger.info(f'Sample Size: {sample_size} | Train size: {train_size} ({train_pct:.3f}) | Test size: {test_size} ({test_pct:.3f})')
        logger.info(f'Image Dimension: {x_set.shape}')
    except Exceptipon as e:
        logger.info(f'An unexpected error occured: {e}')
        raise e
    
    return data, x_train, x_test, y_train, y_test