import os
import re
# import h5py
import datetime
import numpy as np
from sklearn.metrics import roc_auc_score
import multiprocessing
# from .utils_data import *

## Inspect will probably be added in future
import inspect
try:
    getargspec = inspect.getfullargspec
except AttributeError:  # getargspec() is deprecated since Python 3.0
    getargspec = inspect.getargspec

try:
    import tensorflow as tf
    import keras
    import keras.backend as K
    from keras.models import Sequential
    from keras.models import Model
    from keras.models import model_from_json
    from keras.models import model_from_yaml
except ImportError:
    import tensorflow as tf
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import model_from_json
    from tensorflow.keras.models import model_from_yaml
    

KERAS_VERSION = keras.__version__


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def to_categorical(y, num_classes=None):
    """ keras.utils.to_categorical will squeeze dim with size=1. """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def softmax(logits):
    probs = np.exp(logits - np.amax(logits, axis=-1, keepdims=True))
    return probs/np.sum(probs, axis=-1, keepdims=True)


def np_dice_coef(y_true, y_pred):
    tr = y_true.flatten()
    pr = y_pred.flatten()

    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)


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


class DataLoader(keras.utils.Sequence):
    def __init__(self, dataset, batch_size, config=None, 
                 shuffle=True, transform=None, sample_weights=None, **kwargs):
        """ Basic DataLoader. """
        self.dataset = dataset
        self._len = len(self.dataset)
        self.indices = range(self._len)
        self.batch_size = batch_size
        self.config = config
        self.shuffle = shuffle
        self.transform = transform
        self.kwargs = kwargs
        self.sample_weights = sample_weights
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(1.0 * self._len / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data. """
        s = index * self.batch_size % self._len
        e = s + self.batch_size
        indices = self.indices[s:e]
        return self.data_generator(indices)

    def on_epoch_end(self):
        """ Updates indices after each epoch. """
        if self.shuffle:
            self.indices = np.random.permutation(self._len)

    def pad_batch(self, batch_data):
        if len(batch_data) < self.batch_size:
            pad_data = [np.zeros_like(batch_data[0])
                        for _ in range(self.batch_size-len(batch_data))]
            batch_data.extend(pad_data)
        return batch_data
            
    def data_generator(self, indices):
        """ Generate a batch of inputs. """
        # return np.stack(x) for x in zip(*[self.dataset[i] for i in indices])
        raise NotImplementedError("Subclass must implement this method!")


## Track current learning rate, optimizer Nadam is not supported
# https://github.com/keras-team/keras/issues/2823
class LearningRateTracker(keras.callbacks.Callback):
    def __init__(self, verbose=0):
        super(LearningRateTracker, self).__init__()
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose:
            print('learning_rate = {:.6f}'.format(self._get_current_lr()))
    
    def on_epoch_end(self, epoch, logs={}):
        logs['learning_rate'] = self._get_current_lr()
    
    def _get_current_lr(self):
        optimizer = self.model.optimizer
        lr = optimizer.lr
        if optimizer.initial_decay > 0:
            lr = lr * (1. / (1. + optimizer.decay * 
                             K.cast(optimizer.iterations, K.dtype(optimizer.decay))))
        return K.get_value(lr)


# class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
#     def set_model(self, model):
#         self.model = model


class MultiGpuModel(Model):
    def __init__(self, model, gpus=None, cpu_merge=True, cpu_relocation=False):
        """ Build a multi_gpu_model that compatible with keras.callbacks
            (ModelCheckpoint, ReduceLROnPlateau, etc. specifically).
            Note: 
        Params:
            model: The Keras model to parallelize
            gpus, cpu_merge, cpu_relocation: parameters for keras.utils.multi_gpu_model
        
        Note: multi_gpu_model is not compatible with tensorflow 1.14. 
              Downgraded it to 1.13.1 in order to use this funciton.
              See https://github.com/keras-team/keras/issues/13057 for details.
              multi_gpu_model will be deprected in tensorflow 2.1. 
              So don't use tf.keras.utils as backend in tensorflow 2.
        """
        super(MultiGpuModel, self).__init__()
        self.inner_model = model
        self.gpus = gpus
        self.cpu_merge = cpu_merge
        self.cpu_relocation = cpu_relocation
        p_model = keras.utils.multi_gpu_model(model, gpus, cpu_merge, cpu_relocation)
        super(MultiGpuModel, self).__init__(inputs=p_model.inputs, outputs=p_model.outputs)
    
    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(MultiGpuModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(MultiGpuModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)


## https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras
class AdditionalEvalDatasets(keras.callbacks.Callback):
    def __init__(self, datasets, run_frequency=1, batch_size=None, steps=None, verbose=0):
        """
        :param datasets:
        A dictionary of the following:
            1) a list of 3-tuples (validation_data, validation_targets)
            2) or 4-tuples (validation_data, validation_targets, sample_weights)
            3) keras.utils.Sequence
            4) a generator that has __next__ and len method
        :param verbose: 0 or 1 or 2. 0: silent, 1: print progress bar, 2: print one line result only
        :param batch_size: batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalEvalDatasets, self).__init__()
        self.epoch = []
        self.history = {}
        self.batch_size = batch_size
        self.steps = steps
        self.verbose = verbose
        self.run_frequency = run_frequency
        
        ## Organize datasets:
        self.datasets = []
        for name, dataset in datasets.items():
            is_generator = (isinstance(dataset, keras.utils.Sequence) or
                            ((hasattr(dataset, 'next') or hasattr(dataset, '__next__')) and 
                             hasattr(dataset, 'len')))
            if not is_generator:
                # Prepare data for validation
                if len(dataset) == 2:
                    val_x, val_y = dataset
                    val_sample_weight = None
                elif len(dataset) == 3:
                    val_x, val_y, val_sample_weight = dataset
                else:
                    raise ValueError('`eval_dataset` should be a tuple '
                                     '`(val_x, val_y, val_sample_weight)` '
                                     'or `(val_x, val_y)`. Found: ' +
                                     str(dataset))
                val_x, val_y, val_sample_weights = self.model._standardize_user_data(
                    val_x, val_y, val_sample_weight)
                dataset = val_x + val_y + val_sample_weights
            self.datasets.append([name, is_generator, dataset])

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        ## Run evaluation only every self.run_frequency epochs
        if (epoch + 1) % self.run_frequency:
            return
        
        # logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for name, is_generator, dataset in self.datasets:
            if is_generator:
                results = self.model.evaluate_generator(dataset, steps=self.steps, # workers=0,
                                                        verbose=self.verbose % 2)
            else:
                results = self.model.evaluate(dataset[0], dataset[1], batch_size=self.batch_size,
                                              sample_weight=dataset[2], verbose=self.verbose % 2)
            display_str = ''
            for i, result in enumerate(results):
                if hasattr(self.model, 'metrics_names'):
                    valuename = name + '_' + self.model.metrics_names[i]
                else:
                    if i == 0:
                        valuename = name + '_loss'
                    else:
                        valuename = name + '_' + self.model.metrics[i-1].__name__
                    
                display_str += ' - {:}: {:.4f}'.format(valuename, result)
                self.history.setdefault(valuename, []).append(result)
                logs[valuename] = result
            if self.verbose:
                print(display_str)


class roc_callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
    
    def on_train_begin(self, logs={}):
        return
    
    def on_train_end(self, logs={}):
        return
    
    def on_epoch_begin(self, epoch, logs={}):
        return
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' %
              (str(round(roc, 4)), str(round(roc_val, 4))))
        return

    def on_batch_begin(self, batch, logs={}):
        return
    
    def on_batch_end(self, batch, logs={}):
        return


def get_optimizer(optimizer, lr, decay=0.0, clipnorm=0.0, clipvalue=0.0, **kwargs):
    """ Get optimizer. """
    support_optimizers = {'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam'}
    assert optimizer in support_optimizers
    fn = getattr(keras.optimizers, optimizer)
    return fn(lr, decay=decay, clipnorm=clipnorm, clipvalue=clipvalue, **kwargs)


def get_log_dir(model_name=None, model_dir=None, weights_path=None, digits=4, **kwargs):
    """ Sets the model log directory and epoch counter.
    
        model_dir: specify where to write the model, log_dir=model_dir/model_name_yyyymmddThhmm
        weights_path: path/model_name_0020.h5. 
            if model_dir is None: then log_dir=path
            if model_dir is not None: log_dir=model_dir/yyyymmddThhmm
        
        return log_dir, checkpoint_path=log_dir/model_name_04d.h5, epoch
    """
    log_dir, checkpoint_path, epoch, timestamp = None, None, 0, datetime.datetime.now()
    # Set date and epoch counter as if starting a new model
    if model_dir:
        log_dir = os.path.join(model_dir, "{}_{:%Y%m%dT%H%M%S}".format(model_name, timestamp))
        while os.path.exists(log_dir):
            timestamp = datetime.datetime.now()
            log_dir = os.path.join(model_dir, "{}_{:%Y%m%dT%H%M%S}".format(model_name, timestamp))
        print("Create logdir: {}".format(log_dir))
        os.makedirs(log_dir)
    else:
        if weights_path:
            # weights_path = '/path/to/logs/yyyymmddThhmmss/model_name_0020.h5
            # regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/(\w+)\_(\d{4})\.h5"
            regex = r"(.*)/(\w+)\_(\d{4})\.h5"
            m = re.match(regex, weights_path)
            if not m:
                raise ValueError("weights_path need to be like model_name_0020.h5")
            log_dir, epoch = m.group(1), int(m.group(3))
            model_name = m.group(2) if model_name is None else model_name
    
    if log_dir is not None:
        checkpoint_path = os.path.join(log_dir, "{}_*epoch*.h5".format(model_name))
        checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:0%sd}" % digits)
    else:
        raise ValueError("log_dir could not be None!")

    return log_dir, checkpoint_path, epoch, timestamp

## It's hard to maintain different version of keras (<2.2.0) and tf.keras
## And We probably don't need this any more, 
## simpy set default skip_mismatch=True and set up train_layers should be good to go.
def load_weights(filepath, layers, by_name=False,
                 skip_mismatch=False, reshape=False):
    """ Loads all layer weights from a HDF5 save file. """
    try:
        import h5py
        HDF5_OBJECT_HEADER_LIMIT = 64512
    except ImportError:
        h5py = None
    if h5py is None:
        raise ImportError('`load_weights` requires h5py.')
    
    with h5py.File(filepath, mode='r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        
        if KERAS_VERSION.endswith('tf'):
            if by_name:
                from keras.engine.saving import load_weights_from_hdf5_group_by_name as fn
            else:
                from keras.engine.saving import load_weights_from_hdf5_group as fn
        elif KERAS_VERSION >= '2.2.0':
            if by_name:
                from keras.engine.saving import load_weights_from_hdf5_group_by_name as lf
                fn = lambda f, layers: lf(f, layers, skip_mismatch=skip_mismatch, reshape=reshape)
            else:
                from keras.engine.saving import load_weights_from_hdf5_group as lf
                fn = lambda f, layers: lf(f, layers, reshape=reshape)
        else:
            if by_name: 
                from keras.engine.topology import load_weights_from_hdf5_group_by_name as lf
                fn = lambda f, layers: lf(f, layers, skip_mismatch=skip_mismatch, reshape=reshape)
            else:
                from keras.engine.topology import load_weights_from_hdf5_group as lf
                fn = lambda f, layers: lf(f, layers, reshape=reshape)

        ## Call load function
        fn(f, layers)


def compare_layers_weights(layers1, layers2, verbose=False):
    ## Compare layer weights of two models
    assert len(layers1) == len(layers2), "number of layers are inconsistent"
    N = len(layers1)
    res = [False for i in range(N)]
    for i in range(N):
        l1, l2 = layers1[i], layers2[i]
        res[i] = np.all([np.all(x == y) for x, y in zip(l1.get_weights(), l2.get_weights())])
        if verbose and not res[i]:
            print(l1.name, l2.name, res[i])
    return res

    
def step_decay_func(decay=0.5, decay_epochs=10.0):
    def f(epoch, lr):
        return lr * decay if epoch > 0 and epoch % decay_epochs == 0 else lr
    return f


def exp_decay_func(decay=0.5, steps_per_epoch=1):
    def f(epoch, lr):
        return lr * exp(-decay*(epoch*steps_per_epoch))
    return f


class Config(object):
    """ A Config Interface. """
    def __init__(self, **kwargs):
        self.assign(**kwargs)
    
    def __str__(self):
        res = []
        for x in dir(self):
            if not x.startswith("__") and not callable(getattr(self, x)):
                res.append("{} = {}".format(x, self._display_config(x)))
        return '\n'.join(res) + '\n'
    
    def assign(self, **kwargs):
        """ Assign values to attributes."""
        for k, v in kwargs.items():
            if k in dir(self):
                setattr(self, k, v)
    
    def display(self):
        """Display Configuration values."""
        for x in dir(self):
            if not x.startswith("__") and not callable(getattr(self, x)):
                print("{:30} {}".format(x, self._display_config(x)))
    
    def _display_config(self, x):
        return repr(getattr(self, x))


class TrainConfig(Config):
    ## Parameters for DataGenerator and fit_generator
    GPU_COUNT = 1  # NUMBER OF GPUs to use. For CPU training, use 1
    BATCH_SIZE = 32  # total batch_size, multi-gpu model will split this number
    TRAINABLE_LAYERS = ".*"  # set up trainable layers, default train all layers
    SAVE_WEIGHTS_ONLY = True  # set to false if save full graph each epoch
    SAVE_FREQUENCY = 1 # period to save the weights/model
    STEPS_PER_EPOCH = None  # Number of training steps per epoch
    VALIDATION_STEPS = None  # Number of steps run on validation dataset after every training epoch
    EVAL_STEPS = None  # Number of steps run on evaluation datasets after every training epoch
    EVAL_FREQUENCY = None  # Run evaluation for every xxx steps
    LOSS_WEIGHTS = None  # {output_name: loss_weight}, weighting the loss function for multi-task
    
    ##  Optimizer, learning rate, gradient
    OPTIMIZER = ('SGD', dict(momentum=0.0, nesterov=False))
    LEARNING_RATE = 0.01
    LEARNING_RATE_DECAY = 1e-6
    LEARNING_RATE_SCHEDULER = None  # can use ('step_decay', dict(decay=0.5, decay_epochs=10.0)
    GRADIENT_CLIPNORM = 0.0  # Gradient clipnorm
    GRADIENT_CLIPVALUE = 0.0  # Gradient clipvalue
    
    ## Callbacks
    EARLY_STOPPING = {'monitor': 'val_loss', 'patience': 50, 'mode': 'min'}
    REDUCE_LR_ON_PLATEAU = {'monitor': 'val_loss', 'patience': 10, 'mode': 'min', 'factor': 0.1, 'epsilon': 1e-4}
    
    ## Default parameters for optimizers:
    # lr: {'SGD': 0.01, 'RMSprop': 0.001, 'Adagrad': 0.01, 'Adadelta': 1.0, 'Adam': 0.001}
    # ('SGD', dict(momentum=0.0, nesterov=False))
    # ('RMSprop', dict(rho=0.9, epsilon=None))
    # ('Adagrad', dict(epsilon=None))
    # ('Adadelta', dict(rho=0.95, epsilon=None))
    # ('Adam', dict(beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False))
    
    def __init__(self, **kwargs):
        super(TrainConfig, self).__init__(**kwargs)
        if self.LEARNING_RATE_SCHEDULER is None:
            self.LR_SCHEDULER = lambda epoch, lr: lr
        elif self.LEARNING_RATE_SCHEDULER[0] == 'step_decay':
            self.LR_SCHEDULER = step_decay_func(**self.LEARNING_RATE_SCHEDULER[1])
        elif self.LEARNING_RATE_SCHEDULER[0] == 'exponential_decay':
            self.LR_SCHEDULER = exp_decay_func(**self.LEARNING_RATE_SCHEDULER[1])
        else:
            self.LR_SCHEDULER = lambda epoch, lr: lr


def load_configs(filename):
    """ Load model config, train config and checkpoint path. 
        Function is incompleted, haven't load train_config yet.
        This function is used to load configs and then rebuild model based on configs.
        To inference a simple model, may consider use load_model_from_file instead.
    """
    with open(filename) as f:
        ## headers
        _ = f.readline()
        date_time = f.readline().rstrip('\n').split(': ')[1]
        weights_path = f.readline().rstrip('\n').split(': ')[1]
        weights_path = None if weights_path == 'None' else weights_path
        checkpoint_path = f.readline().rstrip('\n').split(': ')[1]
        layer_regex = f.readline().split(': ')[1].strip('\'')
        # print([date_time, weights_path, checkpoint_path, layer_regex])
        _ = f.readline()
        _ = f.readline()

        ## model configs
        model_config = {}
        _ = f.readline()
        for x in f:
            x = x.rstrip('\n')
            if not x:
                break
            name, par = x.split(' = ')
            model_config[name] = eval(par)
    return model_config, {}, checkpoint_path


def load_model_from_file(filename):
    """ load a keras_model directly from a file. 
        The function can easily restore a model. 
        But if train and inference has different structure (MaskRCNN),
        use load_configs and rebuilt model through configs instead.
    """
    if not os.path.isfile(filename):
        raise ValueError("%s is not a valid file" % filename)
    if filename.endswith('.json'):
        with open(filename, 'r') as f:
            # json_dict = json.loads(f.read()) # for a numpy dictionary
            return model_from_json(f.read())
    elif filename.endswith('.yaml'):
        with open(filename, 'r') as f:
            return model_from_yaml(f.read())
    else:
        raise ValueError("%s is not supported by keras, provide .json or .yaml file" % filename)


def save_structure_to_file(keras_model, filename, filetype=None):
    if filetype is None:
        filetype = os.path.splitext(filename)[1]
    assert filetype in ['.json', '.yaml', '.png'], "filetype %s is not supported" % filetype
    if filetype == '.json':
        with open(filename, 'w') as f:
            f.write(keras_model.to_json())
    elif filetype == '.yaml':
        with open(filename, 'w') as f:
            f.write(keras_model.to_yaml())
    else:
        keras.utils.plot_model(keras_model, to_file=filename, show_shapes=True, 
                               show_layer_names=True, rankdir='TB')


class KerasModel(object):
    """ Basic KerasModel Wrapper.
        Integrated common used functons. model is in the keras_model property.
        Must implement: build_model, get_loss, get_loss_weights, get_metrics.
    """
    def __init__(self, config=None, model_dir=None, model_file=None, model_name=None, **kwargs):
        self.config = config
        self.model_dir = model_dir
        self.weights_path = None
        self.model_name = config.NAME if config is not None else model_name
        self.build_model(model_file, **kwargs)
    
    def build_model(self, model_file=None, **kwargs):
        if model_file is not None:
            keras_model = load_model_from_file(model_file)
            layer_regex = {}
        else:
            res = self._build_model(**kwargs)
            inputs, outputs = res[0], res[1]
            keras_model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
            layer_regex = res[2] if len(res) > 2 else {}
        self.keras_model = keras_model
        self.layer_regex = kwargs.setdefault('layer_regex', layer_regex)
        
        ## Add global weight decay:
        if self.config is not None and hasattr(self.config, 'REGULARIZER') and self.config.REGULARIZER is not None:
            for layer in self.keras_model.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = regularizer(self.config.REGULARIZER)
    
    def _build_model(self, **kwargs):
        """ Build a keras model. 
            function must return two lists of tensors: inputs, outputs (layer_regex)
        """
        raise NotImplementedError("Subclass must implement this method!")
    
    def save_to(self, filename, filetype=None):
        save_structure_to_file(self.keras_model, filename, filetype)
    
    def load_weights(self, weights_path=None, by_name=True, **kwargs):
        """ exclude_layers, skip_mismatch=False, reshape=False is not used anymore. """
        if KERAS_VERSION.endswith('tf'):
            self.keras_model.load_weights(weights_path, by_name=by_name)
        else:
            self.keras_model.load_weights(weights_path, by_name=by_name, skip_mismatch=True)
    
    def load_weights_regex(self, weights_path=None, by_name=True, exclude_layers=None,
                           skip_mismatch=False, reshape=False):
        """ Load weights for part/whole network. 
            exclude_layers: a key in self.layer_regex, regular expression (r".*") or a list (range(10))
            weight_path: the hdf5 file contains the weights
            by_name, skip_mismatch, reshape: pass to keras.load_weights.
        """
        layers = self.keras_model.layers
        if exclude_layers is not None:
            if isinstance(exclude_layers, str):
                # get regular expression for a pre-defined key in self.layer_regex
                if exclude_layers in self.layer_regex:
                    exclude_layers = self.layer_regex[exclude_layers]
                # match layer names with regrular expression
                exclude_layers = [layer.name for layer in layers 
                                  if re.fullmatch(exclude_layers, layer.name)]
            layers = filter(lambda l: l.name not in exclude_layers, layers)
        
        load_weights(weights_path, layers, by_name=by_name, 
                     skip_mismatch=skip_mismatch, reshape=reshape)
        
        self.weights_path = weights_path
        print("Loading weights from: %s" % self.weights_path)
        print("Exclude layers: %s" % exclude_layers)
    
    def get_loss(self, config):
        """ Get loss function. """
        raise NotImplementedError("Subclass must implement this method!")
    
    def get_loss_weights(self, config):
        """ Get loss weights. """
        return getattr(config, "LOSS_WEIGHTS", None)
    
    def get_metrics(self, config):
        """ Get evaluation metrics. """
        return None
    
    def get_callbacks(self, config, eval_datasets=None, verbose=1):
        """ Get callbacks. 
            eval_datasets should be a list like the following:
            {'validation_dataset': (validation_X, validation_y, sample_weights(maybe))
             'validation_generator': keras.utils.Sequence(), ...}
            eval_datasets is only used to track performace, not used for model selection.
        """
        ## Base callbacks
        callbacks_list = [
            keras.callbacks.TensorBoard(self.log_dir, histogram_freq=0, 
                                        write_graph=False, write_images=False), 
            LearningRateTracker(verbose=1),
            keras.callbacks.CSVLogger(os.path.join(self.log_dir, 'train_summary.csv'), append=True)
        ]
        ## Optional callbacks
        if config.SAVE_FREQUENCY is not None and config.SAVE_FREQUENCY > 0:
            callbacks_list.append(
                keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, 
                                                save_weights_only=config.SAVE_WEIGHTS_ONLY, 
                                                period=config.SAVE_FREQUENCY)
            )
        if config.EVAL_FREQUENCY is not None and eval_datasets is not None:
            callbacks_list.append(
                AdditionalEvalDatasets(eval_datasets, run_frequency=config.EVAL_FREQUENCY, 
                                       batch_size=config.BATCH_SIZE, steps=config.EVAL_STEPS, 
                                       verbose=verbose)
            )
        if config.LR_SCHEDULER:
            callbacks_list.append(keras.callbacks.LearningRateScheduler(config.LR_SCHEDULER, verbose=0))
        if config.EARLY_STOPPING is not None:
            callbacks_list.append(keras.callbacks.EarlyStopping(verbose=0, **config.EARLY_STOPPING))
        if config.REDUCE_LR_ON_PLATEAU is not None:
            callbacks_list.append(keras.callbacks.ReduceLROnPlateau(verbose=verbose, **config.REDUCE_LR_ON_PLATEAU))
        
        return callbacks_list
    
    def set_trainable(self, trainable_layers, keras_model):
        if trainable_layers in self.layer_regex:
            trainable_layers = self.layer_regex[trainable_layers]
        
        for layer in keras_model.layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                self.set_trainable(trainable_layers, keras_model=layer)
                continue
            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(trainable_layers, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
    
    
    def add_metric(self, value, aggregation=None, name=None):
        """ add_metrics function for the current model. 
            keras and old version of tf.keras do not contain this function.
        """
        if KERAS_VERSION.endswith('tf'):
            self.keras_model.add_metric(value, aggregation=aggregation, name=name)
        else:
            if name in self.keras_model.metrics_names:
                raise ValueError("%s is already in existing metrics list." % name)
            else:
                if aggregation == 'mean':
                    value = K.mean(value)
                elif aggregation == 'sum':
                    value = K.sum(value)
                elif hasattr(aggregation, '__call__') or callable(aggregation):
                    value = aggregation(value)
                self.keras_model.metrics_names.append(name)
                self.keras_model.metrics_tensors.append(value)
    
    
    def compile_model(self, config):
        ## Set up trainable layers:
        self.set_trainable(config.TRAINABLE_LAYERS, self.keras_model)     
        
        ## Get optimizer, loss, loss_weights, metrics
        self.optimizer = get_optimizer(config.OPTIMIZER[0], config.LEARNING_RATE, 
                                       decay=config.LEARNING_RATE_DECAY, 
                                       clipnorm=config.GRADIENT_CLIPNORM, 
                                       clipvalue=config.GRADIENT_CLIPVALUE, 
                                       **config.OPTIMIZER[1])
        
        self.loss = self.get_loss(config)
        self.loss_weights = self.get_loss_weights(config)
        self.metrics = self.get_metrics(config)
        
        ## Set gpus
        if config.GPU_COUNT is not None and config.GPU_COUNT > 1:
            run_model = MultiGpuModel(self.keras_model, config.GPU_COUNT, cpu_relocation=True) # cpu_merge=False for NVLink
        else:
            run_model = self.keras_model
   
        
        run_model.compile(optimizer=self.optimizer, 
                          loss=self.loss, 
                          loss_weights=self.loss_weights,
                          metrics=self.metrics)
        
        return run_model
    
    
    def train(self, train_loader, val_loader, epochs, config, 
              trainable_layers=".*", eval_datasets=None, save_graph_to=['yaml', 'png'], 
              use_multiprocessing=True, max_queue_size=10, verbose=1, print_summary=True, 
              plot_graph=False, workers=multiprocessing.cpu_count(), **kwargs):
        ## Compile trainable layers, loss, loss_weights, metrics, optimizers
        if trainable_layers is not None:
            config.assign(TRAINABLE_LAYERS=trainable_layers)
        run_model = self.compile_model(config=config)
        
        ## Prepare/Start training
        digits = kwargs.setdefault('digits', 4)
        self.log_dir, self.checkpoint_path, self.epoch, st_start =\
            get_log_dir(self.model_name, self.model_dir, self.weights_path, digits=digits)
        callbacks = self.get_callbacks(config, eval_datasets, verbose=verbose)
        
        ## Save model structure to json, yaml and png
        if save_graph_to is not None:
            save_graph_to = set(save_graph_to)
            if plot_graph: # plot_graph will be deprected in future
                save_graph_to.add('png')

            if 'json' in save_graph_to:
                json_file = os.path.join(self.log_dir, "{}_{:%Y%m%dT%H%M%S}.json".format(self.model_name, st_start))
                self.save_to(json_file)
            if 'yaml' in save_graph_to:
                yaml_file = os.path.join(self.log_dir, "{}_{:%Y%m%dT%H%M%S}.yaml".format(self.model_name, st_start))
                self.save_to(yaml_file)
            if 'png' in save_graph_to:
                graph_file = os.path.join(self.log_dir, "{}_{:%Y%m%dT%H%M%S}.png".format(self.model_name, st_start))
                self.save_to(graph_file)
        
        print("#####################################################")
        print("Training start at: %s" % st_start.strftime('%Y-%m-%d %H:%M:%S'))
        if self.config is not None:
            print("\nModel Configurations: ")
            self.config.display()
        print("\nTraining Configurations: ")
        config.display()
        if print_summary:
            print("\nModel summary: ")
            self.keras_model.summary()
        print("\nLoad weights from: %s" % self.weights_path)
        if print_summary:
            print("Trainable layers: ")
            for layer in self.keras_model.layers:
                if layer.trainable and layer.weights:
                    print('%-35s trainable=%-10s (%s)' % 
                          (layer.name, layer.trainable, layer.__class__.__name__))
        print("\nFrom epoch = %s to epoch = %s base_lr = %s" % 
              (self.epoch, epochs, config.LEARNING_RATE))
        print("Checkpoint Path: %s\n" % self.checkpoint_path)
        
        ## write self.config, config start/end epoch into config file.
        with open(os.path.join(self.log_dir, 'train_config.log'), 'a+') as f:
            f.write("#####################################################\n")
            f.write("Training start at: %s\n" % st_start.strftime('%Y-%m-%d %H:%M:%S'))
            f.write("Load weights from: %s\n" % self.weights_path)
            f.write("Checkpoint Path: %s\n" % self.checkpoint_path)
            f.write("Trainable layers regex: %s\n" % repr(trainable_layers))
            f.write("From epoch = %s to epoch = %s base_lr = %s\n" % 
                    (self.epoch, epochs, config.LEARNING_RATE))
            f.write("\nModel Configurations: \n")
            f.write(str(self.config))
            f.write("\nTraining Configurations: \n")
            f.write(str(config))
        
        run_model.fit_generator(
            generator=train_loader,
            epochs=epochs,
            steps_per_epoch=config.STEPS_PER_EPOCH,
            validation_data=val_loader,
            validation_steps=config.VALIDATION_STEPS,
            initial_epoch=self.epoch,
            callbacks=callbacks,
            # class_weight=config.CLASS_WEIGHTS, directly apply class_weights on loss function
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            verbose=verbose
        )
        self.epoch = max(self.epoch, epochs)
        
        ## End of training
        st_end = datetime.datetime.now()
        print("\nTraining ends at: %s" % st_end.strftime('%Y-%m-%d %H:%M:%S'))
        print("Total Run Time: %s" % (st_end - st_start))
        print("#####################################################")
        with open(os.path.join(self.log_dir, 'train_config.log'), 'a') as f:
            f.write("Training ends at: %s\n" % st_end.strftime('%Y-%m-%d %H:%M:%S'))
            f.write("Total Run Time: %s\n" % (st_end - st_start))
            f.write("#####################################################\n")
    
    
    def inference(self, inference_loader, use_multiprocessing=True, 
                  max_queue_size=10, verbose=1, workers=multiprocessing.cpu_count()):
        """ inference on a given inference loader.
            Note: tensorflow 1.12 will force to call model.compile for inference.
                  So please update to tensorflow 1.13.1
                  See https://github.com/tensorflow/tensorflow/issues/24429 for details.
        """
        ## Prepare/Start inference
        st_start = datetime.datetime.now()
        print("#####################################################")
        print("Inference start at: %s" % st_start.strftime('%Y-%m-%d %H:%M:%S'))
        if self.config is not None:
            print("\nModel Configurations: ")
            self.config.display()
        print("\nLoad weights from: %s" % self.weights_path)
        results = self.keras_model.predict_generator(
            inference_loader, 
            steps=None, 
            max_queue_size=max_queue_size, 
            workers=workers, 
            use_multiprocessing=use_multiprocessing, 
            verbose=verbose
        )
        return results
