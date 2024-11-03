import logging
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
import seaborn as sns
import joblib
from matplotlib.axes import Axes
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score, accuracy_score)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(fpath: str):
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
        y_train = np.squeeze(data['y_train'])
        y_test = np.squeeze(data['y_test'])

        sample_size = x_set.shape[0]
        train_size = x_train.shape[0]
        test_size = x_test.shape[0]

        train_pct = train_size / sample_size
        test_pct = test_size / sample_size
        logger.info(
            f'Sample Size: {sample_size} | Train size: {train_size} ({train_pct:.3f}) | Test size: {test_size} ({test_pct:.3f})')
        logger.info(f'Image Dimension: {x_set.shape}')
    except Exception as e:
        logger.info(f'An unexpected error occured: {e}')
        raise e

    return data, x_train, x_test, y_train, y_test


def reshape_images(image: np.array, new_shape: Tuple[int, int] = None):
    """
    Reshape an image (or batch of images) into a specified shape or flatten into a 1D array.

    Args:
        image (np.array): Input image or array of images.
        new_shape (tuple, optional): Desired new shape (height, width, channels).
                                      If None, will flatten the image.

    Returns:
        np.array: Reshaped image.
    """
    # Store the original shape
    original_shape = image.shape

    # Check if the input image is 2D (single image) or 4D (batch of images)
    if image.ndim == 3:  # Single image
        if new_shape is None:
            reshaped_image = image.flatten()  # Flatten the image to 1D
        else:
            reshaped_image = image.reshape(new_shape)
    elif image.ndim == 4:  # Batch of images
        if new_shape is None:
            reshaped_image = image.reshape(image.shape[0], -1)  # Flatten each image in the batch
        else:
            reshaped_image = image.reshape(image.shape[0], *new_shape)  # Reshape each image to new shape
    else:
        raise ValueError("Input image must be 2D (single image) or 4D (batch of images).")

    print(f"Original dimensions: {original_shape}")
    print(f"Reshaped dimensions: {reshaped_image.shape}")

    return reshaped_image


def plot_class_distribution(classes: np.array, figsize: Tuple[int, int] = (10, 6), title: str = 'Class Distribution'):
    """
    Plot the distribution of classes in a numpy array.

    Args:
        classes (np.array): Array containing class labels.
        title (str): Title of the plot.
    """
    unique, counts = np.unique(classes, return_counts=True)
    total_count = counts.sum()
    percentages = (counts / total_count) * 100
    plt.figure(figsize=figsize)
    bars = plt.bar(unique, counts, color='darkblue')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(unique)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{percentage:.1f}%', ha='center', va='bottom')

    plt.show()


def plot_class_distribution_grid(classes: np.array, ax: Axes, title: str = 'Class Distribution'):
    """
    Plot the distribution of classes in a numpy array on the given axis.

    Args:
        classes (np.array): Array containing class labels.
        ax (matplotlib.axes.Axes): Axis to plot on.
        title (str): Title of the plot.

    Returns:
        matplotlib bar plot for ax.
    """
    # Count occurrences of each class
    unique, counts = np.unique(classes, return_counts=True)

    # Calculate total count and percentage
    total_count = counts.sum()
    percentages = (counts / total_count) * 100

    # Create a bar plot on the provided axis
    bars = ax.bar(unique, counts, color='#36454F')

    # Add labels and title
    ax.set_xlabel('Classes')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate bars with percentage values
    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{percentage:.1f}%', ha='center', va='bottom')


def _report_metrics(y_test: np.array, y_train: np.array, preds: np.array, model_name: str, to_pd: bool = True):
    """
    Report the performance metrics for classification tasks.

    Args:
        y_test (np.array): y_test set.
        y_train (np.array): y_train set.
        preds (np.array): The predicted class.
        model_name (str): Name of the model.

    Returns:
        dict[str, Any]: Dictionary containing the result.
    """
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    precision = precision_score(y_test, preds, average='macro')
    recall = recall_score(y_test, preds, average='macro')
    metrics = {'Model': model_name, 'Train Size': len(y_train), 'Test Size': len(y_test), 'Accuracy': accuracy,
        'F1 Score': f1, 'Precision': precision, 'Recall': recall, }

    if to_pd:
        return pd.DataFrame(metrics, index=[0])
    else:
        return metrics

def _filter_to_subset(y_set: np.array, preds: np.array, target_classes: List):
    """Filter the numpy array to those under target_classes"""
    combined = np.array(list(zip(y_set, preds)))
    mask = np.isin(combined[:, 0], target_classes)
    filtered_combined = combined[mask]
    filtered_y_test = filtered_combined[:, 0]
    filtered_preds = filtered_combined[:, 1]
    return filtered_y_test, filtered_preds

def _report_metrics_subset(y_test: np.array, y_train: np.array, preds: np.array, model_name: str, target_classes: List,
                           to_pd: bool = True):
    """
    Report the performance metrics for classification tasks.

    Args:
        y_test (np.array): y_test set.
        y_train (np.array): y_train set.
        preds (np.array): The predicted class.
        model_name (str): Name of the model.
        subset_list (list): list of

    Returns:
        dict[str, Any]: Dictionary containing the result.
    """
    filtered_y_test, filtered_preds = _filter_to_subset(y_test, preds, target_classes)
    return _report_metrics(filtered_y_test, y_train, filtered_preds, model_name, to_pd)


def plot_confusion_matrix(y_test: np.array, preds: np.array, model_name: str, title: str = None):
    """
    Plot the confusion matrix with absolute values and proportions.

    Args:
        y_test (np.array): y_test.
        preds (np.array): Predicted classes.
        model_name (str): The name of the model.
        title (str): Custom string title.

    Returns:
        matplotlib.figure
    """
    cm = confusion_matrix(y_test, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=False, xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            absolute_value = cm[i, j]
            percentage_value = cm_normalized[i, j]
            plt.text(j + 0.5, i + 0.5, f'{absolute_value}\n({percentage_value:.2%})', ha='center', va='center',
                     color='black')

    if title:
        plt.title(f'{title}')
    else:
        plt.title(f'Confusion Matrix for Model: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def _report_metrics_class(y_test, y_train, preds, model_name, class_instance):
    """Report the performance metrics"""
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='micro')
    precision = precision_score(y_test, preds, average='micro')
    recall = recall_score(y_test, preds, average='micro')
    return {'Class': class_instance, 'Model': model_name, 'Train Size': len(y_train), 'Test Size': len(y_test),
        'Accuracy': accuracy, 'F1 Score': f1, 'Precision': precision, 'Recall': recall, }


def _isolate_class(class_instance, y_set, x_set):
    """Carve the original numpy to a specific digit"""
    idx = np.where(y_set == class_instance)[0]
    X_set = x_set[idx]
    y_set = y_set[idx]
    return X_set, y_set


def _report_metrics_per_class_instance(y_test: np.array, y_train: np.array, X_test: np.array, classifier: object,
                                       model_name: str, to_pd: bool = True):
    """
    Report the performance metrics for each class.

    Args:
        y_test (np.array): y_test set.
        y_train (np.array): y_train set.
        X_test (np.array): X_test set.
        model (scikit learn model): Classifier model.
        model_name (str): The name of the model.
        to_pd (bool): Whether to return as pandas DataFrame.

    Returns:
        List[dict[str, Any]]: Dictionary containing the result.
    """
    class_instances = np.unique(y_test)
    performance_list = []

    for class_instance in class_instances:
        X_test_isolated, y_test_isolated = _isolate_class(class_instance, y_test, X_test)
        preds = classifier.predict(X_test_isolated)
        performance = _report_metrics_class(y_test_isolated, y_train, preds, model_name, class_instance)
        performance_list.append(performance)

    if to_pd:
        return pd.DataFrame(performance_list)
    else:
        return performance_list


def report_all_metrics(y_test, y_train, y_pred, X_test, classifier, model_name, to_pd, to_display=True):
    overall_performance = _report_metrics(y_test=y_test, y_train=y_train, preds=y_pred, model_name=model_name)
    result_per_class = _report_metrics_per_class_instance(y_test=y_test, y_train=y_train, X_test=X_test,
                                                          classifier=classifier, model_name=model_name, to_pd=to_pd)
    if to_display:
        display(overall_performance)
        display(result_per_class)
    return overall_performance, result_per_class

def plot_confusion_matrix_simple(y_true, y_pred, class_names, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots a confusion matrix.

    Parameters:
    - y_true: True labels (ground truth)
    - y_pred: Predicted labels
    - class_names: List of class names for labeling the matrix
    - title: Title for the plot
    - cmap: Color map for the plot
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    # Normalize the confusion matrix by row (i.e., by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(18, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)

    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def load_model_and_predict(model_path, X_sample):
    """
    Loads a model from a .pkl file and predicts using the provided input samples.

    Parameters:
    - model_path: Path to the .pkl file containing the trained model.
    - X_sample: Input samples for prediction (must be in the same format as the training data).

    Returns:
    - model: The loaded model.
    - preds: Predictions for the input samples.
    """
    # Load the model
    model = joblib.load(model_path)

    # Ensure X_sample is a numpy array and has the right shape
    if isinstance(X_sample, list):
        X_sample = np.array(X_sample)
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)  # Reshape for a single sample if necessary

    # Make predictions
    preds = model.predict(X_sample)

    return model, preds
