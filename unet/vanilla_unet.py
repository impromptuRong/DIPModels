from ..utils_g.keras_layers import *

CHANNEL_AXIS = -1 if K.image_data_format() == 'channels_last' else 1

## Function not called, for display purpose only. 
def vanilla_unet_2d(x, filter_size, use_batch_norm=False, dropout_rate=0.0):
    conv_1 = unet_conv_block(x, 1*filter_size, use_batch_norm, dropout_rate, name='conv_1')
    pool_1 = MaxPooling2D(pool_size=(2, 2), name='conv_1_pool')(conv_1)

    conv_2 = unet_conv_block(pool_1, 2*filter_size, use_batch_norm, dropout_rate, name='conv_2')
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='conv_2_pool')(conv_2)

    conv_3 = unet_conv_block(pool_2, 4*filter_size, use_batch_norm, dropout_rate, name='conv_3')
    pool_3 = MaxPooling2D(pool_size=(2, 2), name='conv_3_pool')(conv_3)

    conv_4 = unet_conv_block(pool_3, 8*filter_size, use_batch_norm, dropout_rate, name='conv_4')
    pool_4 = MaxPooling2D(pool_size=(2, 2), name='conv_4_pool')(conv_4)

    conv_5 = unet_conv_block(pool_4, 16*filter_size, use_batch_norm, dropout_rate, name='conv_5')
    pool_5 = MaxPooling2D(pool_size=(2, 2), name='conv_5_pool')(conv_5)

    conv_6 = unet_conv_block(pool_5, 32*filter_size, use_batch_norm, dropout_rate, name='conv_6')

    up_5 = Conv2D(16*filter_size, 2, activation='relu', padding='same', name='up_5_upconv')(UpSampling2D(size=(2, 2))(conv_6))
    up_5 = Concatenate(axis=CHANNEL_AXIS, name='up_5_concat')([up_5, conv_5])
    up_conv_5 = unet_conv_block(up_5, 16*filter_size, use_batch_norm, dropout_rate, name='up_5')

    up_4 = Conv2D(8*filter_size, 2, activation='relu', padding='same', name='up_4_upconv')(UpSampling2D(size=(2, 2))(up_conv_5))
    up_4 = Concatenate(axis=CHANNEL_AXIS, name='up_4_concat')([up_4, conv_4])
    up_conv_4 = unet_conv_block(up_4, 8*filter_size, use_batch_norm, dropout_rate, name='up_4')

    up_3 = Conv2D(4*filter_size, 2, activation='relu', padding='same', name='up_3_upconv')(UpSampling2D(size=(2, 2))(up_conv_4))
    up_3 = Concatenate(axis=CHANNEL_AXIS, name='up_3_concat')([up_3, conv_3])
    up_conv_3 = unet_conv_block(up_3, 4*filter_size, use_batch_norm, dropout_rate, name='up_3')

    up_2 = Conv2D(2*filter_size, 2, activation='relu', padding='same', name='up_2_upconv')(UpSampling2D(size=(2, 2))(up_conv_3))
    up_2 = Concatenate(axis=CHANNEL_AXIS, name='up_2_concat')([up_2, conv_2])
    up_conv_2 = unet_conv_block(up_2, 2*filter_size, use_batch_norm, dropout_rate, name='up_2')

    up_1 = Conv2D(1*filter_size, 2, activation='relu', padding='same', name='up_1_upconv')(UpSampling2D(size=(2, 2))(up_conv_2))
    up_1 = Concatenate(axis=CHANNEL_AXIS, name='up_1_concat')([up_1, conv_1])
    up_conv_1 = unet_conv_block(up_1, 1*filter_size, use_batch_norm, dropout_rate, name='up_1')
    
    ## return image map without logits and scores
    return up_conv_1


# def unet_upsampling(filters, use_batch_norm=True, dropout_rate=0., name=None):
#     def f(x, skip):
#         x = Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same', name=name + '_upconv')(x)
        
#         if use_batch_norm:
#             x = BatchNormalization(axis=CHANNEL_AXIS, name=name + '_bn')(x)
#         x = Activation('relu', name=name + '_activation')(x)
#         if dropout_rate > 0.:
#             x = Dropout(dropout_rate)(x)
        
#         return Concatenate(axis=CHANNEL_AXIS, name=name + '_concat')([x, skip])
#     return f


def VanillaUNet(filters=(32, 64, 128, 256, 512, 1024), dropout_rate=0., **kwargs):
    def f(x):
        skip_connections = []
        
        ## Encoder
        for i, units in enumerate(filters, 1):
            name = 'encoder_%s' % i
            x = Connect('Conv2D', units, kernel_size=3, padding='same', dropout_rate=dropout_rate, 
                        activation='relu', name='%s_1' % name, **kwargs)(x)
            x = Connect('Conv2D', units, kernel_size=3, padding='same', dropout_rate=dropout_rate, 
                        activation='relu', name='%s_2' % name, **kwargs)(x)
            skip_connections.append(x)
            x = MaxPooling2D(pool_size=2, name='%s_pool' % name)(x)
        
        ## Decoder
        x = skip_connections.pop()
        for i in range(len(skip_connections), 0, -1):
            name = 'decoder_%s' % i
            skip = skip_connections[i-1]
            units = K.int_shape(skip)[CHANNEL_AXIS]
            # x = unet_upsampling(units, use_batch_norm=use_batch_norm, 
            #                     dropout_rate=dropout_rate, name=name)(x, skip)
            x = Connect('Conv2DTranspose', units, kernel_size=2, strides=2, padding='same', 
                        activation='relu', dropout_rate=dropout_rate, merge='concat', 
                        name='%s_1' % name, **kwargs)(x, skip)
            x = Connect('Conv2D', units, kernel_size=3, padding='same', dropout_rate=dropout_rate, 
                        activation='relu', name='%s_2' % name, **kwargs)(x)
            x = Connect('Conv2D', units, kernel_size=3, padding='same', dropout_rate=dropout_rate, 
                        activation='relu', name='%s_3' % name, **kwargs)(x)
        return x
    return f


def ResUNet(architecture='resnet50', dropout_rate=0., **kwargs):
    resnet_encoder = resnet(architecture, dropout_rate=dropout_rate, 
                            return_indices='all', name='encoder')
    
    def f(x):
        skip_connections = resnet_encoder(x)[1:]
        
        ## Decoder
        x = skip_connections.pop()
        for i in range(len(skip_connections), 0, -1):
            name = 'decoder_%s' % i
            skip = skip_connections[i-1]
            units = K.int_shape(skip)[CHANNEL_AXIS]
            x = Connect('Conv2DTranspose', units, kernel_size=2, strides=2, padding='same', 
                        activation='relu', dropout_rate=dropout_rate, merge='concat', 
                        name='%s_1' % name, **kwargs)(x, skip)
            x = Connect('Conv2D', units, kernel_size=3, padding='same', dropout_rate=dropout_rate, 
                        activation='relu', name='%s_2' % name, **kwargs)(x)
            x = Connect('Conv2D', units, kernel_size=3, padding='same', dropout_rate=dropout_rate, 
                        activation='relu', name='%s_3' % name, **kwargs)(x)
        x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='decoder_upsampling')(x)
        
        return x
    return f
