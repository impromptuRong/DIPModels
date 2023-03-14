from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv3D
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import MaxPooling3D
from keras.layers import UpSampling3D
from keras.models import Model
from keras.utils import Sequence
from keras.utils import to_categorical
import keras.optimizers
import keras.callbacks

import numpy as np
import tensorflow as tf
import skimage
import time
import re
import multiprocessing

from DIPModels.utils_g import utils_image
from DIPModels.utils_g import utils_keras
from DIPModels.utils_g import utils_misc
from . import utils as unet_3d_utils

CHANNEL_AXIS = -1 if K.image_data_format() == 'channels_last' else 1

def conv_bn_relu_block(**kwargs):
    """ Build a conv -> BN (maybe) -> relu block. """
    filters = kwargs['filters']
    kernel_size = kwargs['kernel_size']
    strides = kwargs.setdefault('strides', (1, 1, 1))
    dilation_rate = kwargs.setdefault('dilation_rate', (1, 1, 1))
    kernel_initializer = kwargs.setdefault('kernel_initializer', 'he_normal')
    padding = kwargs.setdefault('padding', 'same')
    kernel_regularizer = kwargs.setdefault('kernel_regularizer', None)
    use_batch_norm = kwargs.setdefault('use_batch_norm', True)
    conv_name = kwargs.setdefault('conv_name', None)
    bn_name = kwargs.setdefault('bn_name', None)
    relu_name = kwargs.setdefault('relu_name', None)

    def f(x):
        x = Conv3D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        if use_batch_norm:
            x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x

    return f


def unet_3d_conv_block(x, filters, use_batch_norm, dropout_rate, name):
    """ Basic UNet convolution block: 2 conv->bn->relu + dropout(maybe). """
    x = conv_bn_relu_block(filters=filters, kernel_size=3, 
                           use_batch_norm=use_batch_norm, 
                           conv_name=name + '_1_conv', 
                           bn_name=name + '_1_bn',
                           relu_name=name + '_1_activation')(x)
    x = conv_bn_relu_block(filters=filters, kernel_size=3, 
                           use_batch_norm=use_batch_norm, 
                           conv_name=name + '_2_conv', 
                           bn_name=name + '_2_bn',
                           relu_name=name + '_2_activation')(x)
    
    ## Add a dropout layer
    if dropout_rate:
        x = Dropout(dropout_rate, name=name + '_1_dp')(x)
    return x

def vanilla_unet_3d_encoder(x, config):
    """ Basic UNet encoder. """
    filter_size = config.FILTER_SIZE
    use_batch_norm = config.ENCODER_USE_BN
    dropout_rate = config.DROPOUT_RATE
    
    # Model Input Size must be dividable by 2**NUM_LAYERS
    h, w, d = K.int_shape(x)[1:-1]
    if h % 2**len(filter_size) or w % 2**len(filter_size) or d % 2**len(filter_size):
        raise Exception("Image size must be dividable by " + 
                        str(2**len(filter_size)) +
                        "to avoid fractions when downscaling and upscaling.")
    
    encoder_layers = []
    for i, filters in enumerate(filter_size[:-1]):
        name = 'encoder_' + str(i + 1)
        x = unet_3d_conv_block(x, filters, use_batch_norm, dropout_rate, name)
        encoder_layers.append(x)
        x = MaxPooling3D(pool_size=2, name=name + '_pool')(x)
    
    x = unet_3d_conv_block(x, filter_size[-1], use_batch_norm, dropout_rate, 
                           name='encoder_' + str(len(filter_size)))
    
    return encoder_layers + [x]


def vanilla_unet_3d_decoder(encoder_layers, config):
    """ Basic UNet decoder. """
    upsampling_mode = config.UPSAMPLING_MODE
    use_batch_norm = config.DECODER_USE_BN
    dropout_rate = config.DROPOUT_RATE
    # assert upsampling_mode in ['nearest', 'Conv2DTranspose'], "Unsupported upsampling mode!"
    
    x = encoder_layers[-1]
    for i in range(len(encoder_layers)-2, -1, -1):
        name = 'decoder_' + str(i+1)
        filter_size = K.int_shape(encoder_layers[i])[-1]
        x = UpSampling3D(size=2, name=name + '_upsampling')(x)
        x = Conv3D(filter_size, kernel_size=2, padding='same', name=name + '_upconv')(x)
        if use_batch_norm:
            x = BatchNormalization(axis=CHANNEL_AXIS, name=name + '_bn')(x)
        x = Activation('relu', name=name + '_activation')(x)
        
        x = Concatenate(axis=CHANNEL_AXIS, name=name + '_concat')([x, encoder_layers[i]])
        x = unet_3d_conv_block(x, filter_size, use_batch_norm, dropout_rate, name=name)
        
    return x



def vanilla_unet_3d(x, filter_size, use_batch_norm=False, dropout_rate=0.0):
    conv_1 = unet_3d_conv_block(x, 1*filter_size, use_batch_norm, dropout_rate, name='conv_1')
    pool_1 = MaxPooling3D(pool_size=2, name='conv_1_pool')(conv_1)

    conv_2 = unet_3d_conv_block(pool_1, 2*filter_size, use_batch_norm, dropout_rate, name='conv_2')
    pool_2 = MaxPooling3D(pool_size=2, name='conv_2_pool')(conv_2)

    conv_3 = unet_3d_conv_block(pool_2, 4*filter_size, use_batch_norm, dropout_rate, name='conv_3')
    pool_3 = MaxPooling3D(pool_size=2, name='conv_3_pool')(conv_3)

    conv_4 = unet_3d_conv_block(pool_3, 8*filter_size, use_batch_norm, dropout_rate, name='conv_4')
    pool_4 = MaxPooling3D(pool_size=2, name='conv_4_pool')(conv_4)
    
    conv_5 = unet_3d_conv_block(pool_4, 16*filter_size, use_batch_norm, dropout_rate, name='conv_5')
    pool_5 = MaxPooling3D(pool_size=2, name='conv_5_pool')(conv_5)

    conv_6 = unet_3d_conv_block(pool_5, 32*filter_size, use_batch_norm, dropout_rate, name='conv_6')

    up_5 = Conv3D(16*filter_size, 2, activation='relu', padding='same', name='up_5_upconv')(UpSampling3D(size=2)(conv_6))
    up_5 = Concatenate(axis=CHANNEL_AXIS, name='up_5_concat')([up_5, conv_5])
    up_conv_5 = unet_3d_conv_block(up_5, 16*filter_size, use_batch_norm, dropout_rate, name='up_5')
    
    up_4 = Conv3D(8*filter_size, 2, activation='relu', padding='same', name='up_4_upconv')(UpSampling2D(size=2)(up_conv_5))
    up_4 = Concatenate(axis=CHANNEL_AXIS, name='up_4_concat')([up_4, conv_4])
    up_conv_4 = unet_3d_conv_block(up_4, 8*filter_size, use_batch_norm, dropout_rate, name='up_4')

    up_3 = Conv3D(4*filter_size, 2, activation='relu', padding='same', name='up_3_upconv')(UpSampling2D(size=2)(up_conv_4))
    up_3 = Concatenate(axis=CHANNEL_AXIS, name='up_3_concat')([up_3, conv_3])
    up_conv_3 = unet_3d_conv_block(up_3, 4*filter_size, use_batch_norm, dropout_rate, name='up_3')

    up_2 = Conv3D(2*filter_size, 2, activation='relu', padding='same', name='up_2_upconv')(UpSampling2D(size=2)(up_conv_3))
    up_2 = Concatenate(axis=CHANNEL_AXIS, name='up_2_concat')([up_2, conv_2])
    up_conv_2 = unet_3d_conv_block(up_2, 2*filter_size, use_batch_norm, dropout_rate, name='up_2')
    
    up_1 = Conv3D(1*filter_size, 2, activation='relu', padding='same', name='up_1_upconv')(UpSampling2D(size=2)(up_conv_2))
    up_1 = Concatenate(axis=CHANNEL_AXIS, name='up_1_concat')([up_1, conv_1])
    up_conv_1 = unet_3d_conv_block(up_1, 1*filter_size, use_batch_norm, dropout_rate, name='up_1')

    return up_conv_1


class UNet3D(object):
    """ Basic 3D UNet model.
        The actual Keras model is in the keras_model property.
    """
    def __init__(self, config, model_dir, weights_path=None, **kwargs):
        self.config = config
        self.weights_path = None
        self.model_dir = model_dir
        
        if config.ENCODER == 'UNet3D':
            self.encoder = vanilla_unet_3d_encoder
        else:
            raise "Unsupport encoder type!"
        
        if config.DECODER == 'UNet3D':
            self.decoder = vanilla_unet_3d_decoder
        else:
            raise "Unsupport decoder type!"
        
        self.model_name = config.NAME + '_' + config.ENCODER + '_' + config.DECODER
        self.build_model(**kwargs)
    
    def build_model(self, **kwargs):
        """ Build UNet architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        img_input = Input(shape=self.config.MODEL_INPUT_SIZE + (self.config.CHANNEL_SIZE,), name='input')
        conv_layers = self.encoder(img_input, self.config, **kwargs)
        conv_final = self.decoder(conv_layers, self.config, **kwargs)
        
        ## TODO: maybe use config.TOPLAYER == 'xxx' to specify this
        ## Add top layers for classification
        if self.config.ENCODER == 'UNet3D':
            bg_logits = Conv3D(2, kernel_size=1, name='unet_mask')(conv_final)
            cl_logits = Conv3D(self.config.NUM_CLASSES-1, kernel_size=1, name='unet_class')(conv_final)
        
        model = Model(inputs=[img_input], outputs=[bg_logits, cl_logits], name=self.model_name)
        
        # TODO: Add multi-GPU support.
        #if config.GPU_COUNT > 1:
        #    from parallel_model import ParallelModel
        #    model = ParallelModel(model, config.GPU_COUNT)
        
        self.keras_model = model
    
    def load_weights(self, layers=r".*", weights_path=None, by_name=True,
                     skip_mismatch=False, reshape=False):
        """ Load weights for part/whole network. 
            layers: either a regular expression like: r".*"
                    or a list or integers like: range(10)
            weight_path: the hdf5 file contains the weights
            by_name, skip_mismatch, reshape: pass to keras.load_weights.
        """
        self.weights_path = None
        model_layers = (self.keras_model.inner_model.layers 
                        if hasattr(self.keras_model, "inner_model") 
                        else self.keras_model.layers)
        ## If layers is a regular expression string
        if isinstance (layers, str):
            layers = [x for x in model_layers if re.fullmatch(layers, x.name)]
        else:
            layers = [model_layers[i] for i in layers]
        
        utils_keras.load_weights(weights_path, layers, by_name=by_name,
             skip_mismatch=skip_mismatch, reshape=reshape)
        print("Loading weights from: " + weights_path)
    
    def get_optimizer(self, learning_rate, decay=1e-6, momentum=0.9, clipnorm=5.0):
        """ Get optimizer. """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, 
                                         clipnorm=clipnorm, nesterov=True)
        optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                                          epsilon=None, decay=decay, amsgrad=False)
        return optimizer
        
    def get_loss(self):
        """ Get loss function. """
        def loss_function(target, output):
            return K.mean(K.categorical_crossentropy(target, K.softmax(output)))
        return {'unet_mask': loss_function, 'unet_class': loss_function}
    
    def get_metrics(self):
        def bg_iou_metrics(target, output):
            y_true, y_pred = target, K.softmax(output)
            return K.mean(utils_keras.iou_coef(y_true, y_pred))
        
        def cl_iou_metrics(target, output):
            y_true, y_pred = target, K.softmax(output)
            # Set bg region to 0 for output
            y_pred = tf.where(tf.not_equal(y_true, tf.constant(0., dtype=tf.float32)), 
                              y_pred, y_true)
            return K.mean(utils_keras.iou_coef(y_true, y_pred))
        
        return {'unet_mask': bg_iou_metrics, 'unet_class': cl_iou_metrics}
    
    def get_callbacks(self, log_dir, checkpoint_dir):
        callbacks_list = [keras.callbacks.ModelCheckpoint(checkpoint_dir, 
                                                          verbose=0, save_weights_only=True),
                          keras.callbacks.TensorBoard(log_dir, histogram_freq=0, 
                                                      write_graph=False, write_images=False)]
        return callbacks_list

    
    def train(self, train_dataset, val_dataset, layer_regex, epochs, 
              use_border_weights=False, use_class_weights=False, border_weights_sigma=None):
        """ Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        layer: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        """
        ## Set up trainable layers:
        # For multi-gpu models
        layers = (self.keras_model.inner_model.layers 
                  if hasattr(self.keras_model, "inner_model") 
                  else self.keras_model.layers)
        
        for layer in layers:
            layer.trainable = bool(re.fullmatch(layer_regex, layer.name))
            
        for l in layers:
            print('%s: trainable=%s' % (l.name, l.trainable))
        
        # Data generators
        train_generator = DataSequence(train_dataset, config=self.config, 
                                       batch_size=self.config.BATCH_SIZE,
                                       use_border_weights=use_border_weights,
                                       use_class_weights=use_class_weights,
                                       border_weights_sigma=border_weights_sigma,
                                       shuffle=True)
        
        val_generator = DataSequence(val_dataset, config=self.config, 
                                     batch_size=self.config.BATCH_SIZE,
                                     use_border_weights=use_border_weights,
                                     use_class_weights=use_class_weights,
                                     border_weights_sigma=border_weights_sigma,
                                     shuffle=True)
        
        self.optimizer = self.get_optimizer(self.config.LEARNING_RATE, 
                                            decay=self.config.WEIGHT_DECAY,
                                            momentum=self.config.LEARNING_MOMENTUM,
                                            clipnorm=self.config.GRADIENT_CLIP_NORM)
        
        self.loss = self.get_loss()
        self.loss_weights = {'unet_mask': self.config.LOSS_WEIGHTS['unet_mask_loss'],
                             'unet_class': self.config.LOSS_WEIGHTS['unet_class_loss']}
        self.metrics = self.get_metrics()
        self.keras_model.compile(optimizer=self.optimizer, 
                                 loss=self.loss, 
                                 loss_weights=self.loss_weights,
                                 metrics=self.metrics)
        
        log_dir, checkpoint_dir, initial_epoch = \
            utils_keras.get_log_dir(self.model_name, self.model_dir, self.weights_path)
        callbacks = self.get_callbacks(log_dir, checkpoint_dir)
        
        lr = self.config.LEARNING_RATE
        print("\nStarting at epoch {}. lr={}\n".format(initial_epoch, lr))
        print("Checkpoint Path: {}".format(checkpoint_dir))
        
        self.keras_model.fit_generator(generator=train_generator,
                                       epochs=epochs,
                                       steps_per_epoch=self.config.STEPS_PER_EPOCH,
                                       validation_data=val_generator,
                                       validation_steps=self.config.VALIDATION_STEPS,
                                       initial_epoch=initial_epoch,
                                       callbacks=callbacks,
                                       max_queue_size=100,
                                       workers=multiprocessing.cpu_count(),
                                       use_multiprocessing=True
                                      )
    
    def detect(self, images, preprocessor=None, verbose=0, **kwargs):
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"
        if verbose:
            print("Processing {} images".format(len(images)))
            for image in images:
                print("image: " + image)
        # Mold inputs to format expected by the neural network
        if preprocessor:
            images = [preprocessor(x, kwargs) for x in images]
        batch_images = np.stack(images, axis=0)
        bg_logits, cl_logits = self.keras_model.predict([batch_images], verbose=0)
        
        bg_scores = utils_image.softmax(bg_logits)
        cl_scores = utils_image.softmax(cl_logits)

        labels = np.argmax(bg_scores, axis=-1) * (np.argmax(cl_probs, axis=-1) + 1)
        return [_ for _ in labels], [_ for _ in bg_scores], [_ for _ in cl_scores]


class DataSequence(Sequence):
    def __init__(self, dataset, config, batch_size, repeat=1,
                 shuffle=True, **kwargs):
        self.dataset = dataset
        self.image_ids = np.copy(dataset.image_ids)
        self.config = config
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle
        
        self.use_border_weights = kwargs.setdefault("use_border_weights", True)
        self.use_class_weights = kwargs.setdefault("use_class_weights", True)
        self.border_weights_sigma = kwargs.setdefault(
            "border_weights_sigma", max(config.MODEL_INPUT_SIZE)/16)

    def __len__(self):
        ## use np.floor will cause self.on_epoch_end not being called at the end of
        ## each epoch
        return int(np.ceil(1.0 * len(self.image_ids) * self.repeat / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data. """
        # Generate indexes of the batch
        s = index * self.batch_size % len(self.image_ids)
        e = s + self.batch_size
        image_ids = np.copy(self.image_ids[s:e])
        ## Re-shuffle the data and fill 0 entries
        if e > len(self.image_ids):
            if self.shuffle:
                np.random.shuffle(self.image_ids)
            image_ids = np.append(image_ids, self.image_ids[:e-len(self.image_ids)])
    
        return self.__data_generator(image_ids)

    def on_epoch_end(self):
        """ Updates indexes after each epoch. """
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __data_generator(self, image_ids):
        # Init the matrix
        # [None, h, w, channel]
        batch_images = np.zeros((self.batch_size,) + self.config.MODEL_INPUT_SIZE + (self.config.CHANNEL_SIZE, ),
                                dtype=np.float32)
        # [None, h, w, 2]
        batch_bg_masks = np.zeros((self.batch_size,) + self.config.MODEL_INPUT_SIZE + (2,), 
                                  dtype=np.float32)
        # [None, h, w, nb_classes-1], remove background 0
        batch_cl_masks = np.zeros((self.batch_size,) + self.config.MODEL_INPUT_SIZE + 
                                  (self.config.NUM_CLASSES-1,), dtype=np.float32)
        
        for i in range(len(image_ids)):
            image, (masks, class_ids) = self.dataset.load_data(image_ids[i])

            bg_target = to_categorical(np.any(masks, axis=-1), num_classes=2)
            if self.use_border_weights:
                bg_scores = unet_3d_utils.unet_border_weights(masks, sigma=self.border_weights_sigma)
                bg_target = bg_target * np.expand_dims(bg_scores, axis=-1)

            cl_target = to_categorical(np.dot(masks, class_ids), 
                                       num_classes=self.config.NUM_CLASSES)[...,1:]
            if self.use_class_weights:
                cl_scores = unet_3d_utils.unet_class_weights(masks, class_ids)
                cl_target = cl_target * cl_scores

            batch_images[i] = image
            batch_bg_masks[i] = bg_target
            batch_cl_masks[i] = cl_target
        
        return ([batch_images], [batch_bg_masks, batch_cl_masks])


