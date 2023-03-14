from ..utils_g.keras_layers import *

CHANNEL_AXIS = -1 if K.image_data_format() == 'channels_last' else 1

def MobileUNet_fixed(architecture='mobilenet', alpha_up=1.0, **kwargs):
    if isinstance(architecture, str):
        assert architecture in ['mobilenet', 'mobilenetv2']
        filters = {'mobilenet': [32, 64, 128, 256, 512, 1024]}[architecture]
        alpha = {'mobilenet': 1.0}[architecture]
        depth_multiplier = {'mobilenet': 1}[architecture]
    else:
        filters, alpha, depth_multiplier = architecture
    mobilenet_encoder = mobilenet(architecture, return_indices="all")
    
    def f(x):
        skip_connections = mobilenet_encoder(x)
        # b00, b01, b03, b05, b11, b13 = skip_connections
        
        x = skip_connections.pop()
        for i in range(len(skip_connections), 1, -1):
            skip = skip_connections[i-1]
            f = int(filters[i] * alpha)
            up = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(x)
            up = Concatenate(axis=CHANNEL_AXIS)([up, skip])
            x = mobilenet_depthwise_conv_block(up, f, alpha_up, depth_multiplier, block_id='u%s' % i)
        
        x = Concatenate(axis=CHANNEL_AXIS)([x, skip_connections[0]])
        # b18 = _depthwise_conv_block(up5, int(filters[0] * alpha), alpha_up, depth_multiplier, block_id=18)
        x = mobilenet_conv_block(x, int(filters[0] * alpha), alpha_up, block_id='u1')
        
        res = UpSampling2D(size=(2, 2), interpolation='bilinear', name='decoder_upsampling')(x)
        
        return res

    return f

def MobileUNet(architecture='mobilenet', alpha_up=1.0, **kwargs):
    if isinstance(architecture, str):
        assert architecture in ['mobilenet', 'mobilenetv2']
        filters = {'mobilenet': [32, 64, 128, 256, 512, 1024]}[architecture]
        alpha = {'mobilenet': 1.0}[architecture]
        depth_multiplier = {'mobilenet': 1}[architecture]
    else:
        filters, alpha, depth_multiplier = architecture
    mobilenet_encoder = mobilenet(architecture, return_indices=None)
    ## return_indices=None returns [b01, b03, b05, b11, b13], the codes will use b01 twice. 
    ## This bug make pneumoconiosis model perform a little bit better...
    ## return_indices='all' returns [b00, b01, b03, b05, b11, b13], 
    
    def f(x):
        skip_connections = mobilenet_encoder(x)
        # (b00), b01, b03, b05, b11, b13 = skip_connections
        
        x = skip_connections.pop()
        for i in range(len(skip_connections), 0, -1):
            skip = skip_connections[i-1]
            f = int(filters[i] * alpha)
            up = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(x)
            up = Concatenate(axis=CHANNEL_AXIS)([up, skip])
            x = mobilenet_depthwise_conv_block(up, f, alpha_up, depth_multiplier, block_id='u%s' % i)
        
        x = Concatenate(axis=CHANNEL_AXIS)([x, skip_connections[0]])
        # b18 = _depthwise_conv_block(up5, int(filters[0] * alpha), alpha_up, depth_multiplier, block_id=18)
        x = mobilenet_conv_block(x, int(filters[0] * alpha), alpha_up, block_id='u1')
        
        res = UpSampling2D(size=(2, 2), interpolation='bilinear', name='decoder_upsampling')(x)
        
        return res

    return f

#         filters = int(filters[4] * alpha)
#         up1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b13)
#         up1 = Concatenate(axis=CHANNEL_AXIS)([up1, b11])
#         b14 = mobilenet_depthwise_conv_block(up1, filters, alpha_up, depth_multiplier, block_id=14)

#         filters = int(filters[3] * alpha)
#         up2 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b14)
#         up2 = Concatenate(axis=CHANNEL_AXIS)([up2, b05])
#         b15 = mobilenet_depthwise_conv_block(up2, filters, alpha_up, depth_multiplier, block_id=15)

#         filters = int(filters[2] * alpha)
#         up3 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b15)
#         up3 = Concatenate(axis=CHANNEL_AXIS)([up3, b03])
#         b16 = mobilenet_depthwise_conv_block(up3, filters, alpha_up, depth_multiplier, block_id=16)

#         filters = int(filters[1] * alpha)
#         up4 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b16)
#         up4 = Concatenate(axis=CHANNEL_AXIS)([up4, b01])
#         b17 = mobilenet_depthwise_conv_block(up4, filters, alpha_up, depth_multiplier, block_id=17)

