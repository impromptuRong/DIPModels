"""
Inherit original Mask R-CNN config.py.
Add an initializer with input kwargs for convenience.
"""
from .mrcnn import config
from ..utils_g import utils_keras

import numpy as np

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

## TODO: put batch_size, gpu_count etc in ModelConfig is not good. Try to remove them
# gpu_count is used in utils.batch_slice, try to rewrite with tf.map_fn
# batch_size is used in generate anchors, try to remove it
class ModelConfig(config.Config):
    def __init__(self, **kwargs):
        """ Set values of computed attributes. """
        for k in kwargs:
            if k in dir(self):
                setattr(self, k, kwargs[k])
        super(ModelConfig, self).__init__()
    
    def __str__(self):
        res = []
        for x in dir(self):
            if not x.startswith("__") and not callable(getattr(self, x)):
                res.append("{} = {}".format(x, self._display_config(x)))
        return '\n'.join(res) + '\n'
    
    def display(self):
        """ Display Configuration values. """
        hide_attr_list = ['BATCH_SIZE', 'GPU_COUNT', 'GRADIENT_CLIP_NORM', 'IMAGES_PER_GPU', 'IMAGE_META_SIZE', 
                          'IMAGE_MAX_DIM', 'IMAGE_MIN_DIM', 'IMAGE_MIN_SCALE', 'IMAGE_RESIZE_MODE', 
                          'LEARNING_MOMENTUM', 'LEARNING_RATE', 'LOSS_WEIGHTS', 'STEPS_PER_EPOCH', 'VALIDATION_STEPS']
        for x in dir(self):
            if not x.startswith("__") and not callable(getattr(self, x)) and x not in hide_attr_list:
                print("{:30} {}".format(x, self._display_config(x)))
    
    def _display_config(self, x):
        return repr(getattr(self, x))


class TrainConfig(utils_keras.TrainConfig):
    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
