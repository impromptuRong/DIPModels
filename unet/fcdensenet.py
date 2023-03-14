## Original implementation: https://github.com/SimJeg/FC-DenseNet
## pytorch version: translated from https://github.com/bfortuner/pytorch_tiramisu
from ..utils_g.keras_layers import *

def DenseBlock(n_layers, growth_rate=16, dropout_rate=0., upsample=False, name=None):
    def f(x):
        features = []
        for _ in range(n_layers):
            l = BatchNormalization()(x)
            l = Activation('relu')(l)
            l = Conv2D(growth_rate, kernel_size=3, padding='same', strides=1)(l)
            if dropout_rate > 0.:
                l = Dropout(dropout_rate)(l)
            features.append(l)
            x = Concatenate()([x, l])
        if upsample:
            return Concatenate()(features)
        else:
            return x
    return f


def TransitionUp(filters, dropout_rate=0., name=None):
    def f(x, skip):
        x = Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same')(x)
        if dropout_rate > 0.:
            x = Dropout(dropout_rate)(x)
        # skip_size, x_size = K.int_shape(skip), K.int_shape(x)
        # dx, dy = int((x_size[0] - skip_size[0])/2), int((x_size[1] - skip_size[1])/2)
        # x = Cropping2D(cropping=((0, 0), (0, 0)))(x)
        return Concatenate()([x, skip])
    return f


def TransitionDown(filters, dropout_rate=0., name=None):
    def f(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
        if dropout_rate > 0.:
            x = Dropout(dropout_rate)(x)
        x = MaxPooling2D(pool_size=2)(x)
        return x
    return f


def FCDenseNet(blocks=(4, 5, 7, 10, 12, 15), growth_rate=16, first_conv_channels=48, dropout_rate=0.):
    ## add bottleneck to up_block and get last up_blocks to simplify for-loop
    out_chans_first_conv = first_conv_channels  ## an alias name in original codes
    down_blocks, up_blocks, last_blocks = blocks[:-1], blocks[:0:-1], blocks[0]
    
    def f(x):
        x = Conv2D(out_chans_first_conv, kernel_size=3, strides=1, padding='same', name='firstconv')(x)
        n_filters = out_chans_first_conv
        
        skip_connections = []
        ## Down sampling
        for n_layers in down_blocks:
            x = DenseBlock(n_layers, growth_rate, dropout_rate=dropout_rate, upsample=False)(x)
            skip_connections.append(x)
            n_filters += (growth_rate * n_layers)
            x = TransitionDown(n_filters)(x)
            
        ## Bottle neck + Up-sampling
        for n_layers in up_blocks:
            x = DenseBlock(n_layers, growth_rate, dropout_rate=dropout_rate, upsample=True)(x)
            skip = skip_connections.pop()
            x = TransitionUp(growth_rate * n_layers)(x, skip)
        x = DenseBlock(last_blocks, growth_rate, dropout_rate=dropout_rate, upsample=False)(x)
        return x

    return f
