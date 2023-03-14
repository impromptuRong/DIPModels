import math
import numpy as np

class Config(object):
    """ UNet configuration. 
        Base configuration class. For custom configurations, create a
        sub-class that inherits from this one and override properties
        that need to be changed.
    """
    # Name the configurations. 
    NAME = None  # Override in sub-classes

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train on each GPU. A 12GB GPU can typically
    # handle 1*[128, 128, 128, 4]
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = None

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = None
    
    # Number of classification classes (including background)
    NUM_CLASSES = 1 + 1  # background + positive classes
    
    ## For vanilla UNet encoder
    # The filter size of eash UNet layer
    ENCODER = 'UNet3D'
    FILTER_SIZE = (32, 64, 128, 256, 512, 1024)
    ENCODER_USE_BN = False
    
    ## The upsampling layer for UNet decoder,
    # optinoal choices are: ['nearest']
    DECODER = 'UNet3D'
    UPSAMPLING_MODE = 'nearest'
    DECODER_USE_BN = False
    
    # Dropout rate
    DROPOUT_RATE = 0.2

    # Input image size
    MODEL_INPUT_SIZE = (256, 256, 256)
    CHANNEL_SIZE = 3
    
    # Loss weights for bg-fg loss and class-losses
    LOSS_WEIGHTS = {
        "unet_mask_loss": 1.,
        "unet_class_loss": 1.
    }

    # Learning rate and momentum
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    
    def __init__(self, **kwargs):
        """Set values of computed attributes."""
        for k in kwargs:
            if k in dir(self):
                setattr(self, k, kwargs[k])
        
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for x in dir(self):
            if not x.startswith("__") and not callable(getattr(self, x)):
                print("{:30} {}".format(x, getattr(self, x)))
        print("\n")
