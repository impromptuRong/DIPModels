import re
from ..utils_g import utils_keras
from . import utils as cl_utils
from . backbone import *
# from ..utils_g.keras_layers import *


class Classification(utils_keras.KerasModel):
    def _build_model(self, **kwargs):
        """ Build Classification model. """
        model_type, model_type_config = self.config.MODEL_TYPE
        # backbone = keras_applications_backbone(model_type, **model_type_config)
        try:
            backbone = keras_applications_classifiers(model_type, **model_type_config)
        except:
            backbone = {'resnet': resnet_encoder}[model_type](**model_type_config)
        
        img_input = Input(shape=self.config.MODEL_INPUT_SHAPE, name='input')
        model_inputs = [img_input]
        feature_layers = backbone(img_input)
        
        if self.config.USE_FPN:
            fpn_feature_maps = fpn_feature_map(feature_layers, config.TOP_DOWN_PYRAMID_SIZE, **kwargs)
            x = Add(name='final_features')([mix_pooling(self.config.POOLING)(x) for x in fpn_feature_maps])
        else:
            x = mix_pooling(self.config.POOLING)(feature_layers[-1])
            # x = GlobalAveragePooling2D()(feature_layers[-1])
        
        # Add a dense header after pooling
        if self.config.TOP_FC_LAYERS:
            for idx, lr_config in enumerate(self.config.TOP_FC_LAYERS):
                x = Dense(lr_config['units'], activation='relu', name='final_dense_%s' % (idx+1))(x)
                lr_dropout_rate = lr_config.get('dropout', self.config.TOP_DENSE_DROPOUT_RATE)
                if lr_dropout_rate:
                    x = Dropout(lr_dropout_rate)(x)
        
        model_outputs = [Dense(num_classes, activation=self.config.TOP_ACTIVATION, name=name)(x)
                         for name, num_classes in self.config.NUM_CLASSES.items()]
        
        logits_layer_regex = ''.join(['|('+key+')' for key, _ in self.config.NUM_CLASSES.items()])
        layer_regex = {
            "headers": r"(rpn\_.*)|(fpn\_.*)|(final\_.*)" + logits_layer_regex,
            "headers+conv1": r"(rpn\_.*)|(fpn\_.*)|(final\_.*)|(conv1\_.*)" + logits_layer_regex,
            "all": ".*",
        }
        
        return model_inputs, model_outputs, layer_regex


    def get_loss_old(self, config):
        """ Get loss function. """
        def ent_loss(target, output):
            return keras.backend.categorical_crossentropy(target, output, from_logits=False)
            # return K.categorical_crossentropy(target, output, from_logits=True)
            # return categorical_crossentropy(target, output, from_logits=True, axis=-1)
        def focal_loss(target, output):
            return categorical_focal_loss_with_weights(
                target, output, gamma=2., class_weights=config.CLASS_WEIGHTS['pattern'],
                from_logits=False, axis=-1)
        # return ct_loss
        c_fns = {
            'cross_entropy': ent_loss, 
            'focal': focal_loss,
        }
        
        return dict([(k, c_fns[l] if l in c_fns else l)
                     for k, l in config.LOSSES.items()])

    
    def get_loss(self, config):
        """ Get loss functions. """
        losses = {}
        for k, cfg in config.LOSSES.items():
            losses[k] = get_metric_fn(cfg)
        return losses

    
    def get_metrics(self, config):
        """ Get metrics functions. """
        metrics = {}
        for k, cfgs in config.METRICS.items():
            metrics[k] = [get_metric_fn(cfg) for cfg in cfgs]
        return metrics
                
    
    def get_metrics_old(self, config):
        def weighted_acc_fn(class_weights):
            def weighted_acc(target, output):
                score_map = categorical_accuracy(target, output, class_weights=class_weights, axis=-1)
                return K.sum(K.clip(score_map, min_value=0., max_value=None))/K.sum(K.abs(score_map))
                # return keras.metrics.categorical_accuracy(target, output)
            return weighted_acc
        
        def precision_fn(target, output):
            return precision(tf.argmax(target, axis=-1), tf.argmax(output, axis=-1))
        
        def recall_fn(target, output):
            return recall(tf.argmax(target, axis=-1), tf.argmax(output, axis=-1))
        
        return {'density': ['acc', weighted_acc_fn(config.CLASS_WEIGHTS['density'])],
                'pattern': ['acc', weighted_acc_fn(config.CLASS_WEIGHTS['pattern']), precision_fn, recall_fn]}
#         c_fns = {
#             'weighted_acc': metric_fn,
#             'precision': precision,
#             'recall': recall,
#         }
        
#         {'density': ['acc', ('weighted_acc', {'class_weights': [0.5, 0.8, 1., 1.]})]}
#         metrics = {}
#         for k, v in config.METRICS.items():
#             m = []
#             for _ in v:
#                 if isinstance(_, str):
#                     if _ in c_fns:
#                         m.append(c_fns[_])
#                     else:
#                         m.append(_)
#                 else:
#                     m.append(c_fns[_[0]](**_[1]))
#             metrics[k].append(m)
            
#         return metrics
#         # return ct_loss


#         # return ['categorical_accuracy']
#         return dict([(k, c_fns[l] if l in c_fns else l)
#                      for k, l in config.METRICS.items()])


    def inference(self, dataset, **kwargs):
        """ Inference on a dataset.

            dataset: utils_data.ImageDataset. Or any object that support indexing.
                     dataset[idx] should return a pair of (image, masks=None), 
            batch_size: inference batch_size
            kwargs: other parameters parsed to keras.predict_generator

            Returns a list of dicts, one dict per image. The dict contains:
        """
#         if not isinstance(dataset, Sequence):
#             dataset = InferenceSequence(dataset, batch_size=batch_size)
        res = super(self.__class__, self).inference(dataset, **kwargs)
        scores = [cl_utils.softmax(x) for x in res]
        
        return [[_ for _ in x] for x in scores]
    
    
    def predict(self, images, batch_size=None, processor=None, verbose=0, **kwargs):
        """ Predict on a list of images. """
        if verbose:
            print("Processing {} images".format(len(images)))
            for image in images:
                print("image: " + image)
        # Mold inputs to format expected by the neural network
        if processor is not None:
            images = [processor(x, **kwargs) for x in images]
        batch_images = np.stack(images, axis=0)
        res = self.keras_model.predict([batch_images], batch_size=batch_size, verbose=0)
        scores = [cl_utils.softmax(x) for x in res]
        
        return [[_ for _ in x] for x in scores]


class DataLoader(utils_keras.DataLoader):
    def __init__(self, dataset, batch_size, config=None, 
                 shuffle=True, transform=None, sample_weights=None, class_weights=None, **kwargs):
        super(self.__class__, self).__init__(
            dataset, batch_size=batch_size, config=config, 
            shuffle=shuffle, transform=transform, 
            sample_weights=sample_weights, **kwargs)
        self.class_weights = class_weights

    def data_generator(self, indices):
        """ Generate one batch of data. 
            keras fit_generator cannot handle sample_weights and 
            class_weights simultaneously. So here we apply class_weights
            directly to y. See the link for detail.
            https://stackoverflow.com/questions/48315094/using-sample-weight-in-keras-for-sequence-labelling
            https://stackoverflow.com/questions/48173168/use-both-sample-weight-and-class-weight-simultaneously/48174220#48174220
        """
        batch = list(zip(*[self.dataset[idx] for idx in indices]))
        if len(batch) == 3:
            X, y, sample_weights = batch
        elif len(batch) == 2:
            X, y = batch
            sample_weights = None
        else:
            raise ValueError("Dataset returns invalid results")
        X = [np.stack(_, axis=0) for _ in zip(*X)]
        y = [np.stack(_, axis=0) for _ in zip(*y)]
        ## apply class_weights to y: 
        ## make sure class_weights follow the same sequence as y
        if self.class_weights is not None:
            assert len(self.class_weights) == len(y), "class_weights does not match y. "
            y = [one_hot * weights for one_hot, weights in zip(y, self.class_weights)]
        if sample_weights:
            return X, y, sample_weights
        else:
            return X, y


#     def get_metrics(self, config):
#         def weighted_categorical_accuracy(target, output):
#             return keras.metrics.categorical_accuracy(target, output) * K.sum(target, axis=-1)
#         def categorical_accuracy(target, output):
#             return keras.metrics.categorical_accuracy(target, output)
#         return dict((k, [weighted_categorical_accuracy, categorical_accuracy]) for k, _ in self.config.NUM_CLASSES)


# class InferenceSequence(keras.utils.Sequence):
#     def __init__(self, dataset, batch_size, **kwargs):
#         self.dataset = dataset
#         self._len = len(self.dataset)
#         self.batch_size = batch_size
    
#     def __len__(self):
#         return int(np.ceil(1.0 * self._len / self.batch_size))
    
#     def __getitem__(self, index):
#         """ Generate one batch of data. """
#         s = index * self.batch_size
#         e = min(s + self.batch_size, self._len)
        
#         X = [self.dataset[idx][0] for idx in range(s, e)]
#         X = [np.stack(_, axis=0) for _ in zip(*X)]
#         return X
