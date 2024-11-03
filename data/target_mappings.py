from enum import Enum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import (
    SVC,
    LinearSVC
)

class TargetMappings(Enum):
    """Mappings of Numerical Target to its Description"""
    MAPPING_DICT = {
    'defective_bottle': 0,
    'defect_free_bottle': 1,
    'defective_cable': 2,
    'defect_free_cable': 3,
    'defective_capsule': 4,
    'defect_free_capsule': 5,
    'defective_carpet': 6,
    'defect_free_carpet': 7,
    'defective_grid': 8,
    'defect_free_grid': 9,
    'defective_hazelnut': 10,
    'defect_free_hazelnut': 11,
    'defective_leather': 12,
    'defect_free_leather': 13,
    'defective_metal_nut': 14,
    'defect_free_metal_nut': 15,
    'defective_pill': 16,
    'defect_free_pill': 17,
    'defective_screw': 18,
    'defect_free_screw': 19,
    'defective_tile': 20,
    'defect_free_tile': 21,
    'defective_toothbrush': 22,
    'defect_free_toothbrush': 23,
    'defective_transistor': 24,
    'defect_free_transistor': 25,
    'defective_wood': 26,
    'defect_free_wood': 27,
    'defective_zipper': 28,
    'defect_free_zipper': 29
}

class Classifier(Enum):
    """Classifiers"""
    models = {
        "Neural Network": MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='linear'),
        "Kernel SVM": SVC(kernel='rbf'),
    }