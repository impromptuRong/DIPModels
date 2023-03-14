import re
import sys
import tensorflow as tf
from functools import partial

try:
    import keras
    import keras.backend as K
    import keras.layers as KL
    from keras.layers import *
except ImportError:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as KL
    from tensorflow.keras.layers import *

# KERAS_VERSION = keras.__version__

current_module = sys.modules[__name__]
# CHANNEL_AXIS = -1 if K.image_data_format() == 'channels_last' else 1

######################################################
#################  Wrapper Funciton  #################
######################################################

## TODO: When a list of list is given, generated nested outputs (like for resnet blocks)
## TODO: currently kwargs is not used, try add global parameter control for **kwargs
def Unamed(configs, name=None, **kwargs):
    """ Build network based on abstract inputs. 
        configs: list of (layer_name, input_node_ids, function, parameters)
            layer_name (str): default is None
            input_node_ids: one of (int, list, tuple, dict)
                int: a single tensor
                list: for functions take a list of tensors as single input
                tuple: for functions take lots of inputs, each is a single tensor
                dict: specify name=val
            function (str): the function name in keras_layers of keras.layers
            parameters (dict): the named parameters for function
    """
    def _call(tensor_list, l_name=None, indices=-1, layer=None, args={}):
        if layer is not None:
            if isinstance(layer, str):
                layer = getattr(current_module, layer)
            elif isinstance(layer, tuple):
                layer = getattr(layer[0], layer[1])
            # args.update(kwargs)
            # args = dict((k, args[k]) for inspect.getargspec(layer)[0])
            layer = layer(name=x_name(name, l_name), **args)
            # print("Cannot create layer: %s(name=%s, %s)" % 
            #       (layer, c_name, ", ".join("=".join(_) for _ in args.items())))
        if isinstance(indices, list):
            x = [None if _ is None else tensor_list[_] for _ in indices]
            return layer(x) if layer is not None else x
        elif isinstance(indices, tuple):
            x = [None if _ is None else tensor_list[_] for _ in indices]
            return layer(*x) if layer is not None else x
        elif isinstance(indices, dict):
            x = dict([(k, None if _ is None else tensor_list[_]) for k, _ in indices.items()])
            return layer(**x) if layer is not None else x
        elif isinstance(indices, int):
            x = tensor_list[indices]
            return layer(x) if layer is not None else x
        else:
            raise "Unsupported config (indices) type!"
    
    def f(x):
        if not configs:
            return x
        nodes_memo = [_ for _ in x] if isinstance(x, list) else [x]
        for config in configs:
            if config:
                r = _call(nodes_memo, *config)
                nodes_memo.append(r)
        return r
    return f


def x_name(name=None, suffix=None):
    if name is None or suffix is None:
        return None
    else:
        return ('%s_%s' % (name, suffix)).strip('_')


######################################################
#################  Customize Layers  #################
######################################################
class BatchNormalization(KL.BatchNormalization):
    """ Hard code trainable status of the Keras BatchNormalization layers.
        Make BN layers functions as a linear normalization layer.
        
        Two cases:
        1. Batch normalization has a negative effect on training if batches 
        are small (like in Mask RCNN). So this layer is sometimes frozen.
        2. In GAN, when freeze a model  with (model.trainable = False). BN 
        layers are still updating self.moving_mean and self.moving_variance. 
        We don't want this when freeze generator and train discriminator.
        
        Official discussion of BN problem:
        https://github.com/keras-team/keras/issues/4762#issuecomment-299606870
    
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def Add(name=None):
    def f(inputs):
        inputs = [_ for _ in inputs if _ is not None]
        if len(inputs) == 0:
            return None
        if len(inputs) == 1:
            return inputs[0]
        return KL.Add(name=name)(inputs)
    return f


def Multiply(name=None):
    def f(inputs):
        inputs = [_ for _ in inputs if _ is not None]
        if len(inputs) == 0:
            return None
        if len(inputs) == 1:
            return inputs[0]
        return KL.Multiply(name=name)(inputs)
    return f


def Concat(axis=-1, name=None):
    def f(inputs):
        inputs = [_ for _ in inputs if _ is not None]
        if len(inputs) == 0:
            return None
        if len(inputs) == 1:
            return inputs[0]
        return KL.Concatenate(axis, name=name)(inputs)
    return f


def FiLM(gamma_u=None, gamma_i=None, beta_u=None, beta_i=None, name=None):
    """ Return a generalized function to merge two feature_map.
        inputs: [v_user, v_item, b_user (None), b_item (None)]
        if film_funcs is None: 
            outputs = v_user * v_item + b_user + b_item
        if film_funcs is not None:
            outputs = gamma_u(v_user) * gamma_i(v_item) + beta_u(b_user) + beta_i(b_item)
        Functions will ignore any inputs that is None and try to generate vectors.
        Functions will override inputs if there are conflicts.
        For instance: both beta_i and b_item are not None, beta_i(b_item) will replace b_item
        
        Some example:
        1. Use drug to modulate cellline with FiLM:
            # cell * gamma(drug) + beta(drug)
            x = FiLM(gamma_i=f1, beta_i=f2)([v_cell, v_drug, None, v_drug])
        2. basic CF + baseline
            # v_user * v_item + b_user + b_item
            x = FiLM()([v_cell, v_drug, b_user, b_item])
    """
    def f(v_user, v_item=None, b_user=None, b_item=None):
        if gamma_u is not None:
            v_user = gamma_u(v_user)
        if gamma_i is not None and v_item is not None:
            v_item = gamma_i(v_item)
        if beta_u is not None and b_user is not None:
            b_user = beta_u(b_user)
        if beta_i is not None and b_item is not None:
            b_item = beta_i(b_item)
        return Add()([Multiply()([v_user, v_item]), b_user, b_item])
    return f


class GlobalDensePooling1D(KL.Dropout, KL.Dense):
    """ Copy from Dense and Dropout layer in keras.layers/tf.leras.layers.
        theoretically data_format should not be used here. It's used to 
        check Pooling2D. But simply put here don't influence the result.
        K.normalize_data_format(data_format) is only used in keras, so use 
        the function K.image_data_format() works for both tf.keras and keras.
    """
    def __init__(self, data_format='channels_last', dropout_rate=0.0, 
                 noise_shape=None, seed=None, **kwargs):
        KL.Dense.__init__(self, units=1, **kwargs)
        # KL.Dropout.__init__(rate=dropout_rate, noise_shape=noise_shape, seed=seed)
        self.rate = min(1., max(0., dropout_rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.input_spec = KL.InputSpec(ndim=3)
        if data_format is None:
            data_format = K.image_data_format()
        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('data_format must be in '
                             '{"channels_last", "channels_first"}')
        self.data_format = data_format
        # self.data_format = K.normalize_data_format(data_format) # this function only work in keras
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        if self.data_format == 'channels_last':
            input_shape = (input_shape[0], input_shape[2], input_shape[1])
        KL.Dense.build(self, input_shape)
        steps_axis = 1 if self.data_format == 'channels_last' else -1
        self.input_spec = KL.InputSpec(ndim=3, axes={steps_axis: input_shape[-1]})
        self.built = True
    
    def call(self, inputs, training=None):
        if self.data_format == 'channels_last':
            x = K.permute_dimensions(inputs, (0, 2, 1))
        x = KL.Dense.call(self, x)
        x = KL.Dropout.call(self, x, training)
        return K.squeeze(x, axis=-1)
    
    def get_config(self):
        config = KL.Dense.get_config(self)
        config.update(KL.Dropout.get_config(self))
        return config
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1])
        else:
            return (input_shape[0], input_shape[2])


def GlobalPooling1D(method, dropout_rate=0., name=None):
    return {'ave': KL.GlobalAveragePooling1D(name=name),
            'max': KL.GlobalMaxPooling1D(name=name),
            'dense': GlobalDensePooling1D(name=name, dropout_rate=dropout_rate)}[method]


def Conv1DTranspose(filters, kernel_size, strides=1, padding='valid', 
                    output_padding=None, data_format=None, dilation_rate=1,
                    activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                    activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
    kernel_size = (1, kernel_size)
    strides = (1, strides)
    # output_padding = (0, output_padding) if output_padding is not None else None
    dilation_rate = (1, dilation_rate)
    layer = KL.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding,
                               # output_padding=output_padding, 
                               data_format=data_format, 
                               dilation_rate=dilation_rate, activation=activation, 
                               use_bias=use_bias, kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, 
                               bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, 
                               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
    
    def fn(x):
        x = ExpandDim(axis=1)(x)
        x = layer(x)
        return Squeeze(axis=1)(x)
    return fn


def Embed(vocab_size, embedding_dim, dropout_rate=0., merge=None, activation=None, name=None, **kwargs):
    def f(x):
        x = KL.Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
                         name=x_name(name, 'embedding'), **kwargs)(x)
        if merge is not None:
            x = GlobalPooling1D(method=merge, name=x_name(name, merge + '_pooling'))(x)
        if activation is not None:
            x = KL.Activation(activation)(x)
        if dropout_rate > 0. and K.ndim(x) == 3:
            x = KL.SpatialDropout1D(rate=dropout_rate, name=x_name(name, 'pooling'))(x)
        return x
    return f


def ExpandDim(axis=-1, name=None):
    return KL.Lambda(lambda x: K.expand_dims(x, axis=axis), name=name)


def Squeeze(axis=-1, name=None):
    return KL.Lambda(lambda x: K.squeeze(x, axis=axis), name=name)


def Connect(method, units=None, filters=None, activation=None, dropout_rate=0.0, 
            use_batch_norm=True, use_bn=None, bn_momentum=0.99, bn_epsilon=0.001,
            merge_method='film', merge=None, film_funcs=None, name=None, **kwargs):
    """ Build a conv/dense -> BN -> FiLM -> activation -> dropout block. 
        method: the name of linear connection keras.layers: 'Conv2D', 'Conv1D', 'Dense', etc.
                or a function, keras.layers.Conv2D, etc.
        units/filters: the number of output channels (units in Dense, filters in Conv)
        use_batch_norm/use_bn: whether to add keras.layers.BatchNormalization
        activation: the activation function parsed to keras.layers.Activation
        dropout_rate: add keras.layers.Dropout(dropout_rate) if dropout_rate > 0.
        merge_method/merge: method to merge modulate vectors, add, multiply, concat, film
        film_funcs (optional): {'gamma': film_gamma_func, 'beta': film_beta_func}. 
                    each function takes (units, name) as inputs, and use modulate to adjust x.
                    Default modulate=None, means no film is applied if only provide x.
                    (See FiLM for details)
        name: the name of this block used to identify layers
        **kwargs: parameters for linear. 
            common: use_bias = kwargs.setdefault('use_bias', True)
                    kernel_initializer = kwargs.setdefault('kernel_initializer', 'he_normal')
                    kernel_regularizer = kwargs.setdefault('kernel_regularizer', None)
            conv:   kernel_size = kwargs.setdefault('kernel_size', (3, 3))
                    strides = kwargs.setdefault('strides', (1, 1))
                    padding = kwargs.setdefault('padding', 'same')
                    dilation_rate = kwargs.setdefault('dilation_rate', (1, 1))
    """
    ## treat alias names for units/filters, use_batch_norm/use_bn, merge_method/merge
    if units is None:
        units = filters
    if use_bn is not None:
        use_batch_norm = use_bn
    if merge is not None:
        merge_method = merge
    
    # Get connection layer functions
    if isinstance(method, str):
        method = {'Dense': KL.Dense, 'Conv1D': KL.Conv1D, 'SeparableConv1D': KL.SeparableConv1D, 
                  'Conv1DTranspose': Conv1DTranspose, 'Conv2D': KL.Conv2D, 'SeparableConv2D': KL.SeparableConv2D, 
                  'DepthwiseConv2D': KL.DepthwiseConv2D, 'Conv2DTranspose': KL.Conv2DTranspose, 'Conv3D': KL.Conv3D, 
                  'Conv3DTranspose': KL.Conv3DTranspose, 
                  # 'UpSampling1D': KL.UpSampling1D, 'UpSampling2D': KL.UpSampling2D, 'UpSampling3D': KL.UpSampling3D, 
                 }[method]
    
    def f(x, modulate=None):
        c_name = x_name(name, re.split('\d', method.__name__.lower())[0])
        x = method(units, name=c_name, **kwargs)(x)
        if use_batch_norm:
            x = KL.BatchNormalization(axis=-1, momentum=bn_momentum, epsilon=bn_epsilon, name=x_name(name, 'bn'))(x)
        if modulate is not None:
            if merge_method == 'film':
                x = FiLM(film_funcs['gamma'], film_funcs['beta'], name=x_name(name, 'film'))(x, modulate)
            elif merge_method == 'concat':
                x = Concat(name=x_name(name, 'concat'))([x, modulate])
            elif merge_method == 'add':
                x = Add(name=x_name(name, 'add'))([x, modulate])
            elif merge_method == 'multiply':
                x = Multiply(name=x_name(name, 'multiply'))([x, modulate])
            else:
                raise ValueError("%s is not a supported merge_method" % merge_method)
        if activation is not None:
            c_name = x_name(name, (activation if isinstance(activation, str) else 'activation'))
            x = KL.Activation(activation, name=c_name)(x)
        if dropout_rate and dropout_rate > 0.:
            x = KL.Dropout(dropout_rate, name=x_name(name, 'dropout'))(x)        
        return x

    return f


def VAE_Sampler(connect, name=None):
    def _sampler(args):
        z_mean, z_log_sigma = args
        shape = K.shape(z_mean)
        epsilon = K.random_normal(shape=shape, mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon
    
    def fn(x):
        z_mean = Unamed(connect, name=x_name(name, 'mean'))(x)
        z_log_sigma = Unamed(connect, name=x_name(name, 'log_sigma'))(x)
        # batch_size = K.shape(z_mean)[0]
        # latent_dim = K.int_shape(z_mean)[1]
        z = KL.Lambda(_sampler, name=x_name(name, 'sampler'))([z_mean, z_log_sigma])
        return z, z_mean, z_log_sigma
    return fn


class InstanceNormalization(KL.Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = KL.InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def regularizer(inputs):
    if inputs is not None:
        l1 = inputs.setdefault('l1', None)
        l2 = inputs.setdefault('l2', None)
        if l1 is None and l2 is None:
            return None
        elif l1 is None:
            return keras.regularizers.l2(l2)
        elif l2 is None:
            return keras.regularizers.l1(l1)
        else:
            return keras.regularizers.l1_l2(l1=l1, l2=l2)

######################################################
#################  ResNet Architect  #################
######################################################
def resnet_block1(units, method='Conv2D', kernel=3, strides=1, conv_shortcut=True, dropout_rate=0., name=None):
    if conv_shortcut:
        shortcut = ('0', 0, 'Connect', {'method': method, 'units': units * 4, 'kernel_size': 1, 
                                        'activation': None, 'strides': strides, 'padding': 'same', 
                                        'dropout_rate': dropout_rate})
    else:
        shortcut = ('0', 0)
    
    config = [
        ('1', 0, 'Connect', {'method': method, 'units': units, 'kernel_size': 1, 'activation': 'relu', 
                             'strides': strides, 'padding': 'same', 'dropout_rate': dropout_rate}),
        ('2', 1, 'Connect', {'method': method, 'units': units, 'kernel_size': kernel, 'activation': 'relu', 
                             'strides': 1, 'padding': 'same', 'dropout_rate': dropout_rate}),
        ('3', 2, 'Connect', {'method': method, 'units': units * 4, 'kernel_size': 1, 'activation': None, 
                             'strides': 1, 'padding': 'same', 'dropout_rate': dropout_rate}),
        shortcut,
        ('add', [-2, -1], 'Add', {}),
        ('out', -1, 'Activation', {'activation': 'relu'}),
    ]
    return Unamed(config, name=name)


def resnet_stack1(method, units, kernel, blocks, stride1=2, dropout_rate=0., name=None):
    assert blocks > 0
    pars_1 = {'method': method, 'units': units, 'kernel': kernel, 
              'strides': stride1, 'dropout_rate': dropout_rate}
    pars_2 = {'method': method, 'units': units, 'kernel': kernel, 
              'conv_shortcut': False, 'dropout_rate': dropout_rate}
    config = ([('block1', 0, 'resnet_block1', pars_1)] + 
              [('block%d' % (i+1), i, 'resnet_block1', pars_2) for i in range(1, blocks)])
    return Unamed(config, name=name)


# ## TODO: add resnext
#     def stack_fn_resnext101(x):
#         x = stack3(x, 128, 3, stride1=1, name='conv2')
#         x = stack3(x, 256, 4, name='conv3')
#         x = stack3(x, 512, 23, name='conv4')
#         x = stack3(x, 1024, 3, name='conv5')
def resnet(architecture='resnet50', method='Conv2D', kernel=3, dropout_rate=0., 
           return_indices=-1, preact_layers='default', name=None):
    if isinstance(architecture, str):
        assert architecture in ['resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101']
        stack, N, _ = re.split(r'(\d+)', architecture)
        stack_f = {'resnet': 'resnet_stack1', 'resnext': 'resnet_stack2'}[stack]
        blocks = {'50': [3, 4, 6, 3], '101': [3, 4, 23, 3], '152': [3, 8, 36, 3]}[N]
        filters = {'resnet': [64, 128, 256, 512], 'resnext': [128, 256, 512, 1024]}[stack]
        strides = [1, 2, 2, 2]
    else:
        stack_f, blocks, filters, strides = architecture
    
    ## Default first block
    if preact_layers == 'default':
        pooling = {"Conv2D": 'MaxPooling2D', "Conv1D": 'MaxPooling1D'}[method]
        f1, s1_kernel, s1_strides, s1_pool_size, s1_padding = [filters[0], 7, 1, 2, 'same']
        preact_layers = [
            ('conv1', 0, 'Connect', {'method': method, 'units': f1, 'kernel_size': s1_kernel, 'activation': 'relu', 
                                     'strides': s1_strides, 'padding': s1_padding, 'dropout_rate': dropout_rate}),
            ('pool1', 1, pooling, {'pool_size': s1_pool_size, 'padding': 'same'}),
        ]
    
    def f(inputs):
        res = [Unamed(preact_layers, name=name)(inputs)]
        for i, (f, c, s) in enumerate(zip(filters, blocks, strides), 2):
            pars = {'method': method, 'units': f, 'kernel': kernel, 'blocks': c, 
                    'stride1': s, 'dropout_rate': dropout_rate}
            config = [('conv%s' % i, 0, stack_f, pars)]
            res.append(Unamed(config, name=name)(res[-1]))
        if return_indices == 'all' or return_indices is None:
            return res
        elif isinstance(return_indices, list):
            return [res[_] for _ in return_indices]
        else:
            return res[return_indices]
    return f

######################################################
################  MobileNet Architect  ###############
######################################################
## keras.activations.relu doesn't have threshold args in 2.1.6
relu6 = lambda x: keras.activations.relu(x, alpha=0.0, max_value=6) #, threshold=0.0)

## A copy from https://github.com/keras-team/keras-applications/blob/dc1416f329cb7fd3639b8fbc3fec01a0b74cded3/keras_applications/mobilenet.py
## Personally I don't like the Pad_Layer + padding="valid" style. So use padding='same'

def mobilenet_conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv%s' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv%s_bn' % block_id)(x)
    x = Activation(relu6, name='conv%s_relu' % block_id)(x)
    # x = ReLU(6., name='conv%s_relu' % block_id)(x)
    
    return x

def mobilenet_depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                                   depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%s' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%s_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%s_relu' % block_id)(x)
    # x = ReLU(6., name='conv_dw_%s_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%s_bn' % block_id)(x)
    x = Activation(relu6, name='conv_pw_%s_relu' % block_id)(x)
    # x = ReLU(6., name='conv_pw_%s_relu' % block_id)(x)
    
    return x

## No top layers and no dropout
def mobilenet(architecture='mobilenet', return_indices=-1, name=None):
    if isinstance(architecture, str):
        assert architecture in ['mobilenet', 'mobilenetv2']
        filters = {'mobilenet': [32, 64, 128, 256, 512, 1024]}[architecture]
        alpha = {'mobilenet': 1.0}[architecture]
        depth_multiplier = {'mobilenet': 1}[architecture]
    else:
        filters, alpha, depth_multiplier = architecture
    
    def f(x):
        img_input = x
        
        ## make the layer name consistent with official model. (Easy loading pretrained model.
        b00 = mobilenet_conv_block(img_input, filters[0], alpha, strides=(2, 2), block_id=1)
        b01 = mobilenet_depthwise_conv_block(b00, filters[1], alpha, depth_multiplier, block_id=1)
        
        b02 = mobilenet_depthwise_conv_block(b01, filters[2], alpha, depth_multiplier, block_id=2, strides=(2, 2))
        b03 = mobilenet_depthwise_conv_block(b02, filters[2], alpha, depth_multiplier, block_id=3)

        b04 = mobilenet_depthwise_conv_block(b03, filters[3], alpha, depth_multiplier, block_id=4, strides=(2, 2))
        b05 = mobilenet_depthwise_conv_block(b04, filters[3], alpha, depth_multiplier, block_id=5)

        b06 = mobilenet_depthwise_conv_block(b05, filters[4], alpha, depth_multiplier, block_id=6, strides=(2, 2))
        b07 = mobilenet_depthwise_conv_block(b06, filters[4], alpha, depth_multiplier, block_id=7)
        b08 = mobilenet_depthwise_conv_block(b07, filters[4], alpha, depth_multiplier, block_id=8)
        b09 = mobilenet_depthwise_conv_block(b08, filters[4], alpha, depth_multiplier, block_id=9)
        b10 = mobilenet_depthwise_conv_block(b09, filters[4], alpha, depth_multiplier, block_id=10)
        b11 = mobilenet_depthwise_conv_block(b10, filters[4], alpha, depth_multiplier, block_id=11)

        b12 = mobilenet_depthwise_conv_block(b11, filters[5], alpha, depth_multiplier, block_id=12, strides=(2, 2))
        b13 = mobilenet_depthwise_conv_block(b12, filters[5], alpha, depth_multiplier, block_id=13)
        
        res = [b00, b01, b03, b05, b11, b13]
        if return_indices == 'all': 
            return res
        elif return_indices is None:
            return res[1:]
        elif isinstance(return_indices, list):
            return [res[_] for _ in return_indices]
        else:
            return res[return_indices]
    return f

## Some backbones in keras_applications
## TODO: add support to feature layers
def keras_applications_backbone(model_type, architecture, weights=None, **kwargs):
    """ Get latest keras_applications builtin models. 
        Keras applications updates sota backbones much faster than keras and tf.keras.
        We only need to update/replace this folder: 
        /ml_env/lib/python3.x/site-packages/keras_applications/
        and then use the following to get the sota models.
        
        model_type: model type (in __init__.py)
        architecture: subtype.
            keras_applications.model_type.architecture
        weights: 'imagenet', None
    """
    import keras_applications
    ## keras is already specified as keras or tensorflow.keras
    # keras_applications._KERAS_BACKEND = keras.backend
    # keras_applications._KERAS_LAYERS = keras.layers
    # keras_applications._KERAS_MODELS = keras.models
    # keras_applications._KERAS_UTILS = keras.utils
    
    backbone = getattr(getattr(keras_applications, model_type), architecture)
    
    def f(x):
        base_model = backbone(
            input_tensor=x, weights=None, include_top=False, 
            backend=keras.backend, layers=keras.layers, 
            models=keras.models, utils=keras.utils, **kwargs)
        weights_path = weights
        if weights_path == 'imagenet':
            weights_path = os.path.join('pretrained_model', WEIGHTS[model_type][architecture])
        if weights_path is not None:
            base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        
        return [base_model.output]
    return f

######################################################
#################  Loss and Metrics  #################
######################################################

def pearson_correlation_coefficient(y_true, y_pred):
    y_true = K.cast(y_true, K.dtype(y_pred))
    x = y_true - K.mean(y_true)
    y = y_pred - K.mean(y_pred)
    return K.sum(x * y) / (K.sqrt(K.sum(K.square(x)) * K.sum(K.square(y))) + K.epsilon())


def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_pred - y_true)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))


def precision(y_true, y_pred, from_softmax=False):
    if from_softmax:
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
    else:
        y_true = K.round(K.clip(y_true, 0.0, 1.0))
        y_pred = K.round(K.clip(y_pred, 0.0, 1.0))
    
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_f, 0, 1)))

    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred, from_softmax=False):
    if from_softmax:
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
    else:
        y_true = K.round(K.clip(y_true, 0.0, 1.0))
        y_pred = K.round(K.clip(y_pred, 0.0, 1.0))
    
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))

    return true_positives / (possible_positives + K.epsilon())


def f1_score(y_true, y_pred):
    return 2. / (1. / recall(y_true, y_pred) + 1. / precision(y_true, y_pred))


def _transform_target(target, output, axis=-1):
    """ Transform (sparse) target to match output. """
    N_classes = output.shape[axis]
    dtype = output.dtype.base_dtype
    
    is_sparse = K.ndim(target) != K.ndim(output)
    if is_sparse:
        ## squeeze target if last dimension is 1
        if len(target.shape) == len(output.shape) and target.shape[axis] == 1:
            target = K.squeeze(target, axis=axis)
        ## transfer to one hot tensor (ignore negative values)
        target = tf.one_hot(indices=K.cast(target, dtype=tf.int32), 
                            depth=N_classes, dtype=dtype, axis=axis)
    else:
        target = tf.cast(target, dtype)
    
    return target


def _apply_class_weights(x, class_weights=None, axis=-1):
    """ Apply class_weights, ignore nan. """
    # N_classes = K.int_shape(x)[axis]
    dtype = x.dtype.base_dtype
    
    if class_weights is None:
        class_weights = tf.ones(tf.shape(x)[axis], dtype=dtype)
    else:
        # assert len(class_weights) == N_classes, "class_weights do not match output"
        # class_weights = K.constant(config.CLASS_WEIGHTS, dtype=K.floatx())
        class_weights = tf.convert_to_tensor(class_weights, dtype=dtype)
    
    ## remove nan
    class_weights = tf.cast(tf.logical_not(tf.is_nan(x)), dtype) * class_weights
    
    return x * class_weights / tf.reduce_sum(class_weights, axis=axis, keepdims=True)
    # return x * (class_weights / K.sum(class_weights))


# def _apply_class_weights(x, class_weights=None, axis=-1):
#     """ Apply class_weights, ignore nan. """
#     # N_classes = K.int_shape(x)[axis]
#     dtype = x.dtype.base_dtype
    
#     if class_weights is None:
#         class_weights = tf.ones(tf.shape(x)[axis], dtype=dtype)
#     else:
#         # assert len(class_weights) == N_classes, "class_weights do not match output"
#         # class_weights = K.constant(config.CLASS_WEIGHTS, dtype=K.floatx())
#         class_weights = tf.convert_to_tensor(class_weights, dtype=dtype)
    
#     ## remove nan
#     valid_entry = tf.cast(tf.logical_not(tf.is_nan(x)), dtype)
#     class_score = valid_entry * class_weights
#     ## w[i] = w[i] * N/sum(w)
#     weights_map = class_score * tf.reduce_sum(valid_entry, axis=axis, keepdims=True)/tf.reduce_sum(class_score, axis=axis, keepdims=True)
    
#     return x * weights_map
#     # return x * (class_weights / K.sum(class_weights))


def categorical_crossentropy(target, output, from_logits=False, class_weights=None, axis=-1):
    """ (weighted) categorical crossentropy
    Args:
        target: can be either a dense one-hot tensor or a sparse label tensor.
        output: logits or probabilities after softmax.
        from_logits: whether output is logits or probabilities
    Return: a tensor with same shape as output without last dimension
        -0: target don't have valid label.
    """
    target = _transform_target(target, output, axis=axis)
    target = _apply_class_weights(target, class_weights, axis=axis)
    
    return K.categorical_crossentropy(target, output, from_logits=from_logits, axis=axis)


def categorical_focal_loss(target, output, gamma=2., class_weights=None, from_logits=False, axis=-1):
    dtype = output.dtype.base_dtype
    output /= K.sum(output, axis=-1, keepdims=True)
    epsilon = K.epsilon()
    output = K.clip(output, epsilon, 1. - epsilon)
    
    if class_weights is None:
        class_weights = tf.ones(tf.shape(target)[axis], dtype=dtype)
    else:
        class_weights = tf.convert_to_tensor(class_weights, dtype=dtype)
    
    target = target * class_weights
    return -K.sum(target * K.pow(1 - output, gamma) * K.log(output), axis=-1)

#     class_weights = alpha * K.pow(1 - prob, gamma)
#     target = _transform_target(target, output, axis=axis) * class_weights
#     # target = _apply_class_weights(target, class_weights, axis=axis)
    
#     return K.categorical_crossentropy(target, output, from_logits=False, axis=axis)


def categorical_score_map(target, output, class_weights=None, axis=-1):
    """ (weighted) categorical accuracy. 
    Args:
        target: can be either a dense one-hot tensor or a sparse label tensor.
        output: logits or probabilities after softmax.
        
    Return: a tensor with same shape as output without last dimension.
        positive: label is correct, [1, 0, 0, 0], [0.5, 0, 0, 0,] -> +0.5
        negative: label is incorrect, [1, 0, 0, 0], [0, 0.5, 0, 0,] -> -0.5
        0: label is ignored. [1, 0, 0, 0], [0, 0, 0, 0] -> +/-0
    
    Example:
        res = categorical_crossentropy(target, output, class_weights=class_weights)
        acc = K.sum(K.clip(res, min_value=0., max_value=None))/K.sum(K.abs(res))
    """
    target = _transform_target(target, output, axis=axis)
    target = _apply_class_weights(target, class_weights, axis=axis)
    
    index = tf.equal(tf.argmax(target, axis=axis), tf.argmax(output, axis=axis))
    score_map = tf.reduce_sum(target, axis=axis)
    
    return tf.where(index, score_map, -score_map)


def categorical_accuracy(target, output, class_weights=None, axis=-1):
    score_map = categorical_score_map(target, output, class_weights=class_weights, axis=-1)
    return K.sum(K.clip(score_map, min_value=0., max_value=None))/K.sum(K.abs(score_map))


def iou_coef(target, output, mode='iou', from_logits=False, 
             class_weights=None, binary=False, axis=-1):
    """ Calculate (soft) iou/dice coefficient for y_true ad y_pred
        target: [batch_size, h, w, N_classes]
        output: [batch_size, h, w, N_classes]
        
        Return: iou/dice coefficient for each classes. [batch_size, N_classes]
                Apply weight to each classes, use 0 to ignor background
                weights = tf.convert_to_tensor([0, ...], dice_coef.dtype.base_dtype)
                dice_coef *= weights / tf.reduce_sum(weights)
    """
    y_true = _transform_target(target, output, axis=axis)
    y_pred = K.softmax(output, axis=axis) if from_logits else output
    
    if binary:
        N_classes = y_pred.shape[axis]
        y_true = tf.one_hot(K.argmax(y_true, axis=axis), depth=N_classes, axis=axis)
        y_pred = tf.one_hot(K.argmax(y_pred, axis=axis), depth=N_classes, axis=axis)
    
    sum_axis = list(range(1, K.ndim(y_pred)))
    del sum_axis[axis]
    
    intersect = K.sum(y_true * y_pred, axis=sum_axis)
    union = K.sum(y_true + y_pred, axis=sum_axis) - intersect
    
    ## We allow nan in res to let apply_class_weights indentify 0/0 case
    if mode == 'dice':
        res = 2.0 * intersect/(union + intersect) # + K.epsilon())
    elif mode == 'iou':
        res = 1.0 * intersect/(union) # + K.epsilon())
    else:
        raise ValueError("mode=%s is not supported!" % mode)
    
    ## Apply weights and give 0 weights to nan
    res = _apply_class_weights(res, class_weights, axis=-1)
    
    ## Remove nan from res after apply weights
    res = tf.where(tf.is_nan(res), tf.zeros_like(res), res)
    
    return res

######################################################
## define some alias for loss and metrics, compliment default keras alias
######################################################

LOSS_METRIC_ALIAS = {
    'cross_entropy': (K.categorical_crossentropy, {'from_logits': False, 'axis': -1}),
    'w_cent': (categorical_crossentropy, {'from_logits': False, 'class_weights': None, 'axis': -1}),
    'acc': (keras.metrics.categorical_accuracy, {}),
    'w_acc': (categorical_accuracy, {'class_weights': None, 'axis': -1}),
    'coef': (pearson_correlation_coefficient, {}),
    'soft_iou': (iou_coef, {'mode': 'iou', 'from_logits': False, 'class_weights': None, 'binary': False, 'axis': -1}),
    'soft_dice': (iou_coef, {'mode': 'dice', 'from_logits': False, 'class_weights': None, 'binary': False, 'axis': -1}),
    'iou': (iou_coef, {'mode': 'iou', 'from_logits': False, 'class_weights': None, 'binary': True, 'axis': -1}),
    'dice': (iou_coef, {'mode': 'dice', 'from_logits': False, 'class_weights': None, 'binary': True, 'axis': -1}),
    'focal': (categorical_focal_loss, {'gamma': 2., 'class_weights': None, 'from_logits': False, 'axis': -1}),
}


def get_loss_fn(cfg):
    if callable(cfg):
        return cfg
    
    if isinstance(cfg, str):  # cfg is a str with no extra parameters:
        fn_name, fn_pars = cfg, {}
    else:  # cfg is a tuple with (fn_name, fn_kwargs)
        fn_name, fn_pars = cfg
        
    if fn_name in LOSS_METRIC_ALIAS:
        fn, default_pars = LOSS_METRIC_ALIAS[fn_name]
        fn_pars = {**default_pars, **fn_pars}
    else:
        try:  # search current_module for matched functions
            fn = getattr(current_module, fn_name)
        except:  # search keras.losses for matched functions
            fn = getattr(keras.losses, fn_name)
    
    res = partial(fn, **fn_pars)
    res.__name__ = fn_name  # Keras requires function names
    
    return res


def get_metric_fn(cfg):
    if callable(cfg):
        return cfg
    
    if isinstance(cfg, str):  # cfg is a str with no extra parameters:
        fn_name, fn_pars = cfg, {}
    else:  # cfg is a tuple with (fn_name, fn_kwargs)
        fn_name, fn_pars = cfg
        
    if fn_name in LOSS_METRIC_ALIAS:
        fn, default_pars = LOSS_METRIC_ALIAS[fn_name]
        fn_pars = {**default_pars, **fn_pars}
    else:
        try:  # search current_module for matched functions
            fn = getattr(current_module, fn_name)
        except:  # search keras.metrics for matched functions
            fn = getattr(keras.losses, fn_name)
    
    res = partial(fn, **fn_pars)
    res.__name__ = fn_name  # Keras requires function names
    
    return res

######################################################
######################################################


# def binary_focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
#     """
#     Binary form of focal loss.
#       FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
#       where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
#     References:
#         https://arxiv.org/pdf/1708.02002.pdf
#     :param y_true: A tensor of the same shape as `y_pred`
#     :param y_pred: A tensor resulting from a sigmoid
#     :return: Output tensor.
#     """
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

#     epsilon = K.epsilon()
#     # clip to prevent NaN's and Inf's
#     pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
#     pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

#     return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
#            -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


# def categorical_focal_loss(target, output, gamma=2., alpha=.25, from_logits=False, axis=-1):
#     """ Softmax version of focal loss.
#            m
#       FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
#           c=1
#       where m = number of classes, c = class and o = observation
#     Parameters:
#       alpha -- the same as weighing factor in balanced cross entropy
#       gamma -- focusing parameter for modulating factor (1-p)
#     Default value:
#       gamma -- 2.0 as mentioned in the paper
#       alpha -- 0.25 as mentioned in the paper
#     References:
#         Official paper: https://arxiv.org/pdf/1708.02002.pdf
#         https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    
#         :param y_true: A tensor of the same shape as `y_pred`
#         :param y_pred: A tensor resulting from a softmax
#         :return: Output tensor.
#     """
#     # Scale predictions so that the class probas of each sample sum to 1
#     output /= K.sum(output, axis=-1, keepdims=True)

#     # Clip the prediction value to prevent NaN's and Inf's
#     epsilon = K.epsilon()
#     output = K.clip(output, epsilon, 1. - epsilon)

#     # Calculate Cross Entropy
#     cross_entropy = -target * K.log(output)

#     # Calculate Focal Loss
#     loss = alpha * K.pow(1 - output, gamma) * cross_entropy

#     return K.sum(loss, axis=-1)


# def dice_coef_single(y_true, y_pred, smooth=1):
#     intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#     union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
#     return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


# def soft_dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
    
#     prod = y_true_f * y_pred_f
#     intersection = K.sum(prod)
    
#     numer = 2. * intersection
#     denom = K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()
    
#     return numer / denom


# def iou_coef(y_true, y_pred, exclude_bg=True):
#     """ Calculate IoU coefficient for y_true ad y_pred
#         y_true: [batch_size, h, w, nb_classes]
#         y_pred: [batch_size, h, w, nb_classes]
        
#         Return: iou_coef for each classes. [batch_size, nb_classes (-1)]
#                 Apply weight to each classes, use 0 to ignor background
#                 weights = tf.convert_to_tensor([0, ...], dice_coef.dtype.base_dtype)
#                 dice_coef *= weights / tf.reduce_sum(weights)
#     """
#     nb_classes = y_pred.get_shape()[-1]
#     y_true = K.one_hot(K.argmax(y_true, axis=-1), num_classes=nb_classes)
#     y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=nb_classes)
#     if exclude_bg:
#         y_true = y_true[..., 1:]
#         y_pred = y_pred[..., 1:]

#     axis = np.arange(1, K.ndim(y_pred)-1)
#     intersect = K.sum(y_true * y_pred, axis=axis)
#     union = K.sum(y_true + y_pred, axis=axis) - intersect
#     iou_coef = 1.0 * intersect/(union + K.epsilon())
    
#     return iou_coef

