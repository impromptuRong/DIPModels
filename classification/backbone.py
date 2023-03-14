from ..utils_g.keras_layers import *

CHANNEL_AXIS = -1 if K.image_data_format() == 'channels_last' else 1


def resnet_identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
        shape: (None, h, w, d) => (None, h, w, filters[-1])
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    ## input_tensor: (None, n, n, filters0)
    # res2x_branch2a: (None, n, n, filters1) kernel_size(1)*filters0*filter1 + filters1
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    # bn2x_branch2a: (None, n, n, filters1) filters1*4
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2a')(x)
    # activation_...: (None, n, n, filters1) 0
    x = Activation('relu')(x)

    # res2x_branch2b: (None, n, n, filters2) kernel_size(9)*filters1*filters2 + filters2
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    # bn2x_branch2b: (None, n, n, filters2) filters2*4
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2b')(x)
    # activation_...: (None, n, n, filters2) 0
    x = Activation('relu')(x)

    # res2x_branch2c: (None, n, n, filters2) kernel_size(1)*filters2*filters3 + filters3
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # bn2b_branch2c: (None, n, n, filters3) filters3*4
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2c')(x)
    
    # add_k (Add): (None, n, n, filters3) 0
    x = Add()([x, input_tensor])
    # res2x_out (Activation): (None, n, n, filters3) 0
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    
    return x


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block. 
        h1, w1 = ([h, w]−kernel_size+2*padding=0)/strides+1 = ceiling([h, w]/strides)
        shape: (None, h, w, d) => (None, h1, w1, filters[-1])

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    ## Dimension reduction: x=(None, h, w, filters0), 
    ## n = (h/w−kernel_size+2*padding=0)/strides+1 = ceiling(n/strides)
    # res2x_branch2a: (None, n, n, filters1) kernel_size(1)*filters0*filter1 + filters1
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    # bn2x_branch2a: (None, n, n, filters1) filters1*4
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2a')(x)
    # activation_...: (None, n, n, filters1) 0
    x = Activation('relu')(x) # (None, h, w, filters1)
    
    # res2x_branch2b: (None, n, n, filters2) kernel_size(9)*filters1*filters2 + filters2
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    # bn2x_branch2b: (None, n, n, filters2) filters2*4
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2b')(x)
    # activation_...: (None, n, n, filters2) 0
    x = Activation('relu')(x)
    
    # res2x_branch2c: (None, n, n, filters3) kernel_size(1)*filters2*filters3 + filters3
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # bn2x_branch2c: (None, n, n, filters3) filters3*4
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2c')(x)
    
    # res2x_branch1: (None, n, n, filters3) kernel_size(1)*filters0*filters3 + filters3
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    # bn2x_branch1: (None, n, n, filters3) filters3*4
    shortcut = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '1')(shortcut)
    
    # add_k (Add): (None, n, n, filters3) 0
    x = Add()([x, shortcut])
    # res2x_out (Activation): (None, n, n, filters3) 0
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    
    return x


def resnet_encoder(architecture, **kwargs):
    """ Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
    """
    assert architecture in ["resnet50", "resnet101"], "Resnet architecture is not supported!"
    
    def f(x, modulation=None):
        # Stage 1
        # input_image: (None, 1024, 1024, 3) 0
        # x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_image) # conv1_pad: (None, 1030, 1030, 3) 0
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(x) # conv1: (None, 512, 512, 64) 49*3*64+64
        x = BatchNormalization(axis=CHANNEL_AXIS, name='bn_conv1')(x) # bn_conv1: (None, 512, 512, 64) 64*4
        C1 = x = Activation('relu')(x) # activation_1: (None, 512, 512, 64) 0

        # Stage 2
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x) # max_pooling2d_1: (None, 256, 256, 64) 0
        x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1)) # res2a_out: (None, 256, 256, 256), s=1
        x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block='b') # res2b_out: (None, 256, 256, 256)
        C2 = x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block='c') # res2c_out: (None, 256, 256, 256)

        # Stage 3
        x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block='a') # res3a_out: (None, 128, 128, 512), s=2
        x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block='b') # res3b_out: (None, 128, 128, 512)
        x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block='c') # res3c_out: (None, 128, 128, 512)
        C3 = x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block='d') # res3d_out: (None, 128, 128, 512)

        # Stage 4
        x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block='a') # res3a_out: (None, 64, 64, 1024), s=2
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i)) # res3x_out: (None, 64, 64, 1024)
        C4 = x

        # Stage 5
        x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        C5 = x

        return [C1, C2, C3, C4, C5]
    
    return f


def fpn_feature_map(feature_layers, top_down_pyramid_size, **kwargs):
    """ build a rpn based on feature layers. """
    C1, C2, C3, C4, C5 = feature_layers
    P5 = Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c5p5')(C5)
    P4 = Add(name="fpn_p4add")([
        UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c4p4')(C4)])
    P3 = Add(name="fpn_p3add")([
        UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c3p3')(C3)])
    P2 = Add(name="fpn_p2add")([
        UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    
    return [P2, P3, P4, P5, P6]


def mix_pooling(method):
    def f(x):
        v = []
        if "max" in method:
            v.append(GlobalMaxPooling2D()(x))
        if "ave" in method:
            v.append(GlobalAveragePooling2D()(x))
        if "flatten" in method:
            v.append(Flatten()(x))
        return Concatenate()(v) if len(v) > 1 else v[0]
    
    return f


## TODO: add support to feature layers
def keras_applications_classifiers(model_type, architecture, **kwargs):
    """ Default keras applications builtin models. """
    import keras_applications
    model_builder = getattr(getattr(keras_applications, model_type), architecture)
    # weights = kwargs.get('weights', None)
    
    def f(x):
        base_model = model_builder(
            input_tensor=x, weights=None, include_top=False, 
            backend=keras.backend, layers=keras.layers, 
            models=keras.models, utils=keras.utils, **kwargs)
        
        return [base_model.output]
    return f
