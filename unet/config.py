import math
import numpy as np

from ..utils_g import utils_misc
from ..utils_g import utils_keras


class ModelConfig(utils_misc.Config):
    """ UNet configuration. 
        Base configuration class. For custom configurations, create a
        sub-class that inherits from this one and override properties
        that need to be changed.
    """
    # Name the configurations. 
    NAME = None  # Override in sub-classes

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + positive classes
    
    # Input image size
    MODEL_INPUT_SHAPE = (512, 512, 3)
    MODEL_OUTPUT_SHAPE = None
    
    ## different model has differen structures
    # vanilla unet: ('UNet', {'filters': (32, 64, 128, 256, 512, 1024), 'use_batch_norm': False})
    # resnet unet: ('ResUNet', {'architecture': 'resnet101'})
    # mobile unet: ('MobileUNet', {'architecture': 'mobilenet'})
    # fcdensenet: ('FCDenseNet', {'blocks': (4, 5, 7, 10, 12, 15), 'growth_rate': 16, 'first_conv_channels': 48})
    MODEL_TYPE = ('UNet', {'filters': (32, 64, 128, 256, 512, 1024), 'use_batch_norm': False})
    
    # Dropout rate
    DROPOUT_RATE = 0.2


class TrainConfig(utils_keras.TrainConfig):
    # LOSS_TYPE: crossentropy, dice_coef
    LOSS = ['dice_coef', 'crossentropy']
    # use border weights, parse parameters to border weights function: sigma=MODEL_INPUT_SIZE/16
    BORDER_WEIGHTS = None  # {sigma: MODEL_INPUT_SIZE/16}
    # use class weights
    CLASS_WEIGHTS = None
