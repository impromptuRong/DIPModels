import math
import numpy as np

from ..utils_g import utils_misc
from ..utils_g import utils_keras


class ModelConfig(utils_misc.Config):
    NAME = None  # Name the configurations. 
    NUM_CLASSES = {'output': 2} # give a list of (output_name, num_labels) for multi-task classification
    
    MODEL_INPUT_SHAPE = (256, 256, 3)
    MODEL_TYPE = ('resnet', {'architecture': 'ResNet101'}) # Use ("efficientnet", "EfficientNetB5") as default in future
    USE_FPN = False
    POOLING = 'ave+max'
    TOP_DOWN_PYRAMID_SIZE = 256
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    TOP_FC_LAYERS = [{'units': 1024, 'apply_film': None}] * 2 # dense layer built on top of feature backbones
    DROPOUT_RATE = 0.0 # dropout rate in backbones, default is 0
    TOP_DENSE_DROPOUT_RATE = 0.4  # the dropout rate in final dense layer
    TOP_ACTIVATION = None

class TrainConfig(utils_keras.TrainConfig):
    ## for each loss, metric, provide a str, or tuple of (fn_name, kwargs)
    LOSSES = {'output': 'cross_entropy'} # default use cross_entropy, for unbalanced case, use focal_loss
    METRICS = {'output': 'acc'} # other options: precision, recall, etc
    CLASS_WEIGHTS = None # give a list of numpy array for unbalanced class to calculate weighted_acc, or weighted_cross_entropy
