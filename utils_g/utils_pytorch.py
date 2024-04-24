import os
import re
import sys
import time
import yaml
import json
import errno
import torch
import logging
import datetime
import warnings
import importlib
import torch.distributed as dist

from collections import deque, defaultdict, OrderedDict
from tqdm import tqdm, trange
# from tqdm import tqdm_notebook


try:
    from torch.utils.tensorboard import SummaryWriter
    # spam_spec = importlib.util.find_spec("spam")
    # found = spam_spec is not None
    LOG_TENSORBOARD = True
except Exception as e:
    print(e)
    LOG_TENSORBOARD = False


PYTHON_VERSION = '{}.{}.{}'.format(*sys.version_info[:3])
TORCH_VERSION = torch.__version__


class Config(object):
    """ A Config Interface. """
    def __init__(self, **kwargs):
        self.keys = []
        self.assign(**kwargs)
    
    def assign(self, **kwargs):
        """ Assign values to attributes. """
        for k, v in kwargs.items():
            setattr(self, k, v)
            if not k.startswith("__") and k not in self.keys:
                self.keys.append(k)
    
    def to_dict(self):
        res = {}
        for k in self.keys:
            res[k] = getattr(self, k)
        return res
    
    def save_config(self, filepath):
        with open(filepath, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
    
    def __str__(self):
        return str(self.to_dict())
    
    def __repr__(self):
        """ Display Configuration values. """
        res = []
        for k in self.keys:
            print("{:30} {}".format(k, repr(getattr(self, k))))
        return '\n' + '\n'.join(res) + '\n'
        

class TrainConfig(Config):
    ## Parameters for DataGenerator and fit_generator
    GPU_IDS = [0]  # GPU IDS for training
    BATCH_SIZE = 32  # total batch_size
    TRAINABLE_LAYERS = ".*"  # default: all layers
    SAVE_STATE = True  # True: model.state_dict(), False: whole model, 
    SAVE_FREQUENCY = 1 # steps to save the weights/model
    EVAL_FREQUENCY = 1  # steps to run evaluation
    LOSS_WEIGHTS = None  # {output_name: loss_weight}, weighting multiple loss functions
    CLASS_WEIGHTS = None  # weighting multiple classes
    
    OPTIMIZER = ('SGD', {'lr': 0.001, 'momentum': 0.9})
    GRADIENT_CLIPNORM = 0.0  # Gradient clipnorm
    GRADIENT_CLIPVALUE = 0.0  # Gradient clipvalue
    
    ## Callbacks
    LR_SCHEDULER = ('MultiStepLR', {'milestones': [300, 800], 'gamma': 0.5})
    WARMUP = None  # {'multiplier': 1.0, 'total_epoch': 3}
    EARLY_STOPPING = ('loss/val', {'patience': 50, 'mode': 'min'})
    REDUCE_LR_ON_PLATEAU = ('loss/val', {'mode': 'min', 'factor': 0.1, 
                            'patience': 50, 'verbose': False, 'threshold': 1e-4})

## make an ABC for ordered dict
# https://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict
class Meter(object):
    """ Define a logger abstract class. """
    def __init__(self, fn):
        self.data = fn()
    
    def __getitem__(self, k):
        raise NotImplementedError("Subclass must implement this method!")
    
    def __contains__(self, k):
        return k in self.data
    
    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class MovingAverage(Meter):
    """ Computes and stores the average and current value. """
    def __init__(self):
        super(MovingAverage, self).__init__(OrderedDict)
    
    def update(self, vals):
        for k, (x, n) in vals.items():
            if k not in self.data:
                self.data[k] = (0., 0.)
            mean, size = self.data[k]
            size += n
            if size > 0:
                mean = mean * (1 - n/size) + x * n/size
            self.data[k] = (mean, size)
    
    def __getitem__(self, k):
        return self.data[k][0]


class WindowMovingAverage(object):
    def __init__(self, size=float('Inf')):
        self.data = deque()
        self.size = size
    
    def update(self, val):
        self.data.append(val)
        if len(self.data) > self.size:
            self.data.popleft()
            
    def __call__(self, w=0):
        if w <= 0 or w >= len(self.data):
            return sum(self.data)/len(self.data)
        else:
            array = self.data[-w:0]
        return sum(array)/w

    def item(self, w=0):
        return self.__call__()

    
class ExponentialMovingAverage(object):
    def __init__(self, mu=0.9):
        self.mu = mu
        self.shadow = {}
    
    def update(self, name, val):
        self.shadow[name] = val.clone()
    
    def __call__(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


class StatsSavers(Meter):
    """ Computes and stores the give static cross the whole dataset. 
        This composited meter contains two components:
        self.data (OrderedDict): contains all display/output stats.
            For moving average (not use memo): {k: (mva, denominator)}
            For overall average (use memo): {k: (fn, [k1, k2,... in self.memo])}
        self.memo: store accumulated raw values (y, y_pred, etc)
            For instance: {'y_pred': [], 'y: []}
        When calling update: 
    """
    def __init__(self):
        self.data = OrderedDict()
        self.memo = defaultdict(list)
    
    def update(self, vals):
        """ vals will be a dictionary.
            if k.startswith('_'): the given value will be stored into memo
            else: expect a pair of (mean, denominator)
                new_denominator = old_denominator + denominator
                new_mva = old_mva * (1-old_denominator)/new_denominator + mean * denominator/new_denominator
        """
        for k, val in vals.items():
            ## hidden states, concat into memo along first dimension
            if k.startswith('_'):
                self.memo[k].append(val)
            else:
                x, n = val
                if callable(x):
                    self.data[k] = (x, n)
                else:
                    if k not in self.data:
                        self.data[k] = (0., 0.)
                    mean, size = self.data[k]
                    size += n
                    if size > 0:
                        mean = mean * (1 - n/size) + x * n/size
                    self.data[k] = (mean, size)
    
    def __getitem__(self, k):
        s1, s2 = self.data[k]
        if callable(s1):
            return s1(*[torch.cat(self.memo[_]) for _ in s2])
        else:
            return s1


def x_name(n1, n2, c='_'):
    if n1 is None:
        return n2
    if n2 is None:
        return n1
    n1, n2 = n1.strip(c), n2.strip(c)
    return ('{}{}{}'.format(n1, c, n2)).strip(c)


def get_gpu_device(gpu_ids):
    ## Set up gpus and default gpus
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_ids[0])
        # device = torch.device("cuda", index=torch.cuda.current_device())
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
    else:
        device = torch.device('cpu')
        # if device.type == 'cpu':
        logging.warning("Using CPU to train the model!")
    
    return device


def to_device(x, device):
    """ Move objects to device.
        1). if x.to(device) is valid, directly call it.
        2). if x is a function, ignore it
        3). if x is a dict, list, tuple of objects.
            recursively send elements to device.
        Function makes a copy of all non-gpu objects of x. 
        It will skip objects already stored on gpu. 
    """
    try:
        return x.to(device)
    except:
        if not callable(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    x[k] = to_device(v, device)
            else:
                x = type(x)([to_device(v, device) for v in x])
    return x


def get_log_dir(model_name=None, model_dir=None, weights_path=None, **kwargs):
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
        ## skip distributed run folder creation error
        mkdir(log_dir)
        # os.makedirs(log_dir)
    else:
        if weights_path:
            # weights_path = '/path/to/logs/yyyymmddThhmmss/model_name_0020.h5
            # regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/(\w+)\_(\d{4})\.h5"
            regex = r"(.*)/(\w+)\_(\d{4})\.pt"
            m = re.match(regex, weights_path)
            if not m:
                raise ValueError("weights_path need to be like model_name_0020.pt")
            log_dir, epoch = m.group(1), int(m.group(3))
            model_name = m.group(2) if model_name is None else model_name
    
    if log_dir is not None:
        checkpoint_path = os.path.join(log_dir, "{}_*epoch*.pt".format(model_name))
        checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")
    else:
        raise ValueError("log_dir could not be None!")

    return log_dir, checkpoint_path, epoch, timestamp


class TorchModel(object):
    def __init__(self, config=None, model_name=None, model_dir=None, model_file=None, **kwargs):
        self.config = config
        self.model_name = model_name
        self.model_dir = model_dir
        self.epoch = 0
        self.weights_path = None
        self.device = torch.device('cpu')
        self._build_model(model_file, **kwargs)
    
    def _build_model(self, model_file=None, **kwargs):
        if model_file is not None:
            model = load_model_from_file(model_file)
            layer_regex = {}
        else:
            res = self.build_model(**kwargs)
            try:
                model, layer_regex = res
            except:
                model, layer_regex = res, {}
        self.model = model
        self.layer_regex = kwargs.setdefault('layer_regex', layer_regex)
    
    def build_model(self, **kwargs):
        """ Build a pytorch model. 
            function should return: torch.nn.Module, (layer_regex)
        """
        raise NotImplementedError("Subclass must implement this method!")
    
    def load_weights(self, weights_path, map_location='cpu', strict=True, epoch=0):
        self.model.load_state_dict(torch.load(weights_path, map_location=map_location), strict=strict)
        self.weights_path = weights_path
        self.epoch = epoch
        print("Load weights from: %s" % self.weights_path)
    
    def get_criterion(self, config):
        """ Set up criterions. """
        raise NotImplementedError("Subclass must implement this method!")
    
    def set_trainable(self, trainable_layers):
        pass
    
    def to_script(self, inputs=None, mode="eval"):
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        
        if inputs is not None:
            inputs = inputs.to(self.device)
            try:
                script = torch.jit.trace(self.model, inputs)
            except:
                script = torch.jit.script(self.model)
        else:
            script = torch.jit.script(self.model)
        
        return script
    
    def compile_model(self, config):
        """
            1. set up gpus
            2. send loss, metrics, model to gpu and device.
            3. organize optimizers, lr_schedulers, weight/log savers
        """
        ## Set up gpus and default gpus
        gpu_ids = config.GPU_IDS
        if len(gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(gpu_ids[0])
            # self.device = torch.device("cuda", index=torch.cuda.current_device())
            self.device = torch.device('cuda:{}'.format(gpu_ids[0]))
        else:
            self.device = torch.device('cpu')
            # if device.type == 'cpu':
            logging.warning("Using CPU to train the model!")
        
        ## froze non-trainable layers
        self.set_trainable(config.TRAINABLE_LAYERS)
        
        ## Register models on gpu
        self.model.to(self.device)
        if len(gpu_ids) > 1:
            self.net = torch.nn.DataParallel(self.model, gpu_ids)
        else:
            self.net = self.model
        
        # Register loss, loss_weights and metrics
        self.criterion = self.get_criterion(config)
        if not isinstance(self.criterion, dict):
            self.criterion = {'loss': self.criterion}
        self.criterion = to_device(self.criterion, self.device)
        self.loss_weights = config.LOSS_WEIGHTS or {'loss': 1.0}
        
        ## Register optimizers, lr scheduler, gradient clipping
        params = [_ for _ in self.net.parameters() if _.requires_grad]
        self.optimizer = getattr(torch.optim, config.OPTIMIZER[0])(params, **config.OPTIMIZER[1])
        if config.LR_SCHEDULER:
            if not callable(config.LR_SCHEDULER[0]):
                fn = getattr(torch.optim.lr_scheduler, config.LR_SCHEDULER[0])
            else:
                fn = config.LR_SCHEDULER[0]
            lr_scheduler = fn(self.optimizer, **config.LR_SCHEDULER[1])
            if config.WARMUP is not None:
                lr_scheduler = GradualWarmupScheduler(
                    self.optimizer, after_scheduler=lr_scheduler, **config.WARMUP)
            self.lr_scheduler = lr_scheduler
        else:
            self.lr_scheduler = None
        
        if config.REDUCE_LR_ON_PLATEAU:
            self.reduce_lr_on_plateau = (config.REDUCE_LR_ON_PLATEAU[0], 
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, **config.REDUCE_LR_ON_PLATEAU[1]))
        else:
            self.reduce_lr_on_plateau = None
        
        self.clipnorm = config.GRADIENT_CLIPNORM or 0.
        self.clipvalue = config.GRADIENT_CLIPVALUE or 0.
        print(f"optimizer={self.optimizer}: clipnorm={self.clipnorm}, clipvalue={self.clipvalue}")
    
    def train(self, train_loader, val_loader, epochs, config, 
              trainable_layers=".*", verbose=1, **kwargs):
        """ train a model.
            verbose: 0: silent, 1: progress bar, 2: one line per epoch.
            provide eval_loaders for extra dataset
        """
        ## Compile trainable layers, loss, loss_weights, metrics, optimizers
        if isinstance(config, dict):
            config = Config(**config)
        if trainable_layers is not None:
            config.assign(TRAINABLE_LAYERS=trainable_layers)
        self.compile_model(config=config)
        
        ## Prepare/Start training
        self.log_dir, self.checkpoint_path, self.epoch, st_start =\
            get_log_dir(self.model_name, self.model_dir, self.weights_path)
        self.epoch = kwargs.setdefault('initial_epoch', self.epoch)
        eval_loaders = kwargs.setdefault('eval_loaders', {})
        
        ## Train and eval, register tensorboard, export train_config and model_config
        print("Starting at epoch {}. lr={}".format(self.epoch, config.OPTIMIZER[1]['lr']))
        if not os.path.exists(self.log_dir):
            print("Create logdir: {}".format(self.log_dir))
            mkdir(self.log_dir)
            # os.makedirs(self.log_dir)
        print("Checkpoint Path: {}".format(self.checkpoint_path))
        
        model_config_file = os.path.join(self.log_dir, 'model_config.yaml')
        with open(model_config_file, 'w', encoding='utf8') as f:
            if self.config is not None:
                if isinstance(self.config, dict):
                    yaml.safe_dump(self.config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                else:
                    yaml.safe_dump(self.config.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        train_config_file = os.path.join(self.log_dir, 'train_config_{:04d}_{:04d}.yaml'.format(self.epoch+1, epochs))
        with open(train_config_file, 'w', encoding='utf8') as f:
            yaml.safe_dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        ## start training
        if LOG_TENSORBOARD:
            tf_writer = SummaryWriter(log_dir=self.log_dir)
        else:
            tf_writer = None
        
        indicator, ckpt_epoch = float('inf'), self.epoch
        while self.epoch < epochs:
            self.epoch += 1
            logger = OrderedDict({'epoch': self.epoch})
            
            self.on_epoch_begin(config, logger, **kwargs)
            self.train_on_loader(train_loader, logger, verbose=verbose)
            if val_loader is not None:
                self.evaluate_on_loader(val_loader, logger, verbose=verbose, name='val')
            for k, eval_loader in eval_loaders.items():
                self.evaluate_on_loader(eval_loader, logger, verbose=verbose, name=k)
            self.on_epoch_end(config, logger=logger, **kwargs)
            
            ## save logs and models
            if config.SAVE_FREQUENCY and self.epoch % config.SAVE_FREQUENCY == 0:
                f = self.checkpoint_path.format(epoch=self.epoch)
                if config.SAVE_STATE is True:
                    obj = self.model.state_dict()
                elif config.SAVE_STATE is False:
                    obj = self.model
                else:
                    obj = {
                        'epoch': self.epoch,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'logger': logger,
                        'scheduler': None, 
                        'reduce_lr_on_plateau': None
                    }
                    if self.lr_scheduler is not None:
                        obj['scheduler'] = self.lr_scheduler.state_dict()
                    if self.reduce_lr_on_plateau is not None:
                        obj['reduce_lr_on_plateau'] = self.reduce_lr_on_plateau.state_dict()
                    obj = obj[config.SAVE_STATE]
                torch.save(obj, f)
            
            if config.EVAL_FREQUENCY and self.epoch % config.EVAL_FREQUENCY == 0:
                ## output epoch information into log file
                self.write_log(logger, tf_writer)
            
            new_indicator = logger[config.EARLY_STOPPING[0]]
            if new_indicator < indicator - config.EARLY_STOPPING[1]['threshold']:
                indicator, ckpt_epoch = new_indicator, self.epoch
            else:
                if self.epoch - ckpt_epoch > config.EARLY_STOPPING[1]['patience']:
                    break
    
    def write_log(self, logger, writer, global_step=None):
        """ Save stats, images, etc for tensorboard.
            See https://pytorch.org/docs/stable/tensorboard.html 
            for all supported functions.
        """
        global_step = global_step or logger['epoch']
        if LOG_TENSORBOARD:
            for k, v in logger.items():
                if k != 'epoch':
                    writer.add_scalar(k, v, global_step)
        
        with open(os.path.join(self.log_dir, "train_log.txt"), "a") as f:
            f.write(json.dumps({k: (v.item() if isinstance(v, torch.Tensor) else v)
                                for k, v in logger.items()}) + "\n")

#             if k.startswith('train_'):
#                 writer.add_scalar(k.split('train_')[1] + '/train', v, global_step)
#             elif k.startswith('val_'):
#                 writer.add_scalar(k.split('val_')[1] + '/val', v, global_step)
#             elif k.startswith('test_'):
#                 writer.add_scalar(k.split('test_')[1] + '/test', v, global_step)
#             else:
#                 if k != 'epoch':
#                     writer.add_scalar(k, v, global_step)
        
    
    def train_on_loader(self, dataloader, logger=None, verbose=1, name=None):
        """ Train the model with dataloader on device
        Args:
            dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
            logger: an dictionary that memorize all information cross training.
            verbose: 1
        """
        # set model to training mode
        self.net.train()
        
        # epoch states saver, change to Exponential as gradient keep changing
        # stats_saver = MovingAverage()
        stats_saver = StatsSavers()
        
        # Train the model
        if hasattr(dataloader, '__len__'):
            # with tqdm(total=len(dataloader)) as t:
            with trange(len(dataloader), disable=verbose != 1) as t:
                for i, batch_data in enumerate(dataloader):
                    # start = time.time()
                    batch_data = to_device(batch_data, self.device)
                    batch_stats = self.train_on_batch(batch_data)
                    stats_saver.update(batch_stats)

                    postfix = [(x_name(name, k), '{:05.4f}'.format(stats_saver[k]))
                               for k in stats_saver]
                    t.set_postfix(OrderedDict(postfix))
                    t.update()
                    #tqdm.write("batch_time: %s" % (time.time()-start))
            if verbose == 2:
                log_info = ['{}={:05.4f}'.format(x_name(name, k), stats_saver[k]) 
                            for k in stats_saver]
                print(', '.join(log_info))
        else:
            for i, batch_data in enumerate(dataloader):
                # start = time.time()
                batch_data = to_device(batch_data, self.device)
                batch_stats = self.train_on_batch(batch_data)
                stats_saver.update(batch_stats)
                #print("batch_time: %s" % (time.time()-start))
            if verbose > 0:
                log_info = ['{}={:05.4f}'.format(x_name(name, k), stats_saver[k]) 
                            for k in stats_saver]
                print(', '.join(log_info))
        
        # update logger, add "train" prefix if not provided (for tensorboard)
        if logger is not None:
            for k in stats_saver:
                logger[x_name(k, 'train', '/')] = stats_saver[k]
    
    def evaluate_on_loader(self, dataloader, logger=None, verbose=1, name=None):
        """ Evaluate the model with dataloader on device
        Args:
            dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data.
            logger: an dictionary that memorize all information cross training.
            name: prefix for registered tensors.
        """
        # set model to eval mode
        self.net.eval()
        
        # Define state saver
        # stats_saver = MovingAverage()
        stats_saver = StatsSavers()
        
        # Evaluate the model
        if hasattr(dataloader, '__len__'):
            # with tqdm(total=len(dataloader)) as t:
            with trange(len(dataloader), disable=verbose != 1) as t:
                for i, batch_data in enumerate(dataloader):
                    batch_data = to_device(batch_data, self.device)
                    batch_stats = self.evaluate_on_batch(batch_data)
                    stats_saver.update(batch_stats)

                    postfix = [(x_name(name, k), '{:05.4f}'.format(stats_saver[k]))
                               for k in stats_saver]
                    t.set_postfix(OrderedDict(postfix))
                    t.update()
            if verbose == 2:
                log_info = ['{}={:05.4f}'.format(x_name(name, k), stats_saver[k]) 
                            for k in stats_saver]
                print(', '.join(log_info))
        else:
            for i, batch_data in enumerate(dataloader):
                batch_data = to_device(batch_data, self.device)
                batch_stats = self.evaluate_on_batch(batch_data)
                stats_saver.update(batch_stats)
            if verbose > 0:
                log_info = ['{}={:05.4f}'.format(x_name(name, k), stats_saver[k]) 
                            for k in stats_saver]
                print(', '.join(log_info))
        
        # update memo
        if logger is not None:
            for k in stats_saver:
                logger[x_name(k, name, '/')] = stats_saver[k]
    
    def train_on_batch(self, batch_data):
        loss, batch_stats = self._forward(batch_data)
        self._backward(loss)
        
        return batch_stats
    
    def evaluate_on_batch(self, batch_data):
        with torch.no_grad():
            _, batch_stats = self._forward(batch_data)
        
        return batch_stats
    
    def _forward(self, batch_data):
        batch_X, batch_y = batch_data
        # batch_size = batch_y.size(0)
        batch_outputs = self.net(batch_X)
        
        batch_stats = OrderedDict([(_, fn(batch_outputs, batch_y))
                                   for _, fn in self.criterion.items()])
        
        loss = sum(batch_stats[_] * self.loss_weights[_] for _ in self.loss_weights)
        if 'loss' not in batch_stats:
            batch_stats['loss'] = loss
        
        return loss, batch_stats

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.clipnorm:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipnorm)
        if self.clipvalue:
            torch.nn.utils.clip_grad_value_(self.net.parameters(), self.clipvalue)
            
        # if check_nan:
        for name, x in self.net.named_parameters():
            self._diagnose_nan(x.grad, name + '.grad', inputs=loss)
        
        self.optimizer.step()
    
    def on_epoch_begin(self, config, logger=None, **kwargs):
        logger['epoch'] = self.epoch
        logger['lr'] = [g['lr'] for g in self.optimizer.param_groups][0]
        tqdm.write("Epoch {}, lr={:.6f}".format(logger['epoch'], logger['lr']))
        
    def on_epoch_end(self, config, logger=None, **kwargs):
        # Update learning rates
        if self.lr_scheduler:
            self.lr_scheduler.step()
            # print("Set learning rate to {:.6f}".format(*self.lr_scheduler.get_lr()))
        if self.reduce_lr_on_plateau:
            self.reduce_lr_on_plateau[1].step(logger[self.reduce_lr_on_plateau[0]])
            # print("Set learning rate to {:.4f}".format(*self.reduce_lr_on_plateau[1].get_lr()))
    
    def predict_on_loader(self, dataloader):
        # set model to eval mode
        self.model.eval()
        
        res = []
        for batch_data in dataloader:
            batch_data = to_device(batch_data, self.device)
            with torch.no_grad():
                outputs = self.model(batch_data)
            res.append(outputs)
        ## res is a list of list of tensors, nrow = len(dataloader), ncol = no. of outputs
        return [torch.cat(list(output), dim=0) for output in zip(*res)]


#     def predict_on_batch(self, batch_data, device=None):
#         # set model to eval mode
#         self.model.eval()
        
#         if device:
#             self.model.to(device)
#             batch_data = to_device(batch_data, device)
#         with torch.no_grad():
#             outputs = self.model(batch_data)
#         return outputs

    def _diagnose_nan(self, x, name, inputs=None):
        if isinstance(x, torch.Tensor):
            if torch.isnan(x).any():
                raise ValueError("Nan occurs in tensor %s (inputs=%s)!" % (name, inputs))

################################################################
class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier=1.0, total_epoch=3, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


#######################################################
## vision.references.detections.utils
#######################################################
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
#             if (print_freq > 0 and i % print_freq == 0) or i == len(iterable):
#                 eta_seconds = iter_time.global_avg * (len(iterable) - i + 1)
            if (print_freq > 0 and i % print_freq == 0) or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)