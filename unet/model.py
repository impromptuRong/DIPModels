import numpy as np
import datetime
# from keras.activations import softmax

from .vanilla_unet import VanillaUNet, ResUNet
from .mobile_unet import MobileUNet
from .fcdensenet import FCDenseNet

from ..utils_g import utils_keras
from ..utils_g.keras_layers import *
from . import utils as seg_utils


class UNet(utils_keras.KerasModel):
    def _build_model(self, **kwargs):
        """ Build UNet model. """
        name, structure = self.config.MODEL_TYPE
        
        module = {'UNet': VanillaUNet, 'ResUNet': ResUNet, 
                  'FCDenseNet': FCDenseNet , 'MobileUNet': MobileUNet}[name]
        module = module(dropout_rate=self.config.DROPOUT_RATE, **structure)
        
        ## build the UNet
        img_input = Input(shape=self.config.MODEL_INPUT_SHAPE, name='input')
        img_output = module(img_input)
        logits = Conv2D(self.config.NUM_CLASSES, kernel_size=1, name='final_logits')(img_output)
        
        if self.config.MODEL_OUTPUT_SHAPE is not None:
            resize_layer = Lambda(lambda x: tf.image.resize_images(
                out, size=self.config.MODEL_OUTPUT_SHAPE, 
                method=tf.image.ResizeMethod.BILINEAR), name='final_resize')
            logits = resize_layer(logits)
            # UpSampling2D(self.config.MODEL_OUTPUT_SHAPE, interpolation='bilinear', name='final_resize')(logits)
        
        outputs = Activation('softmax')(logits)
        layer_regex = {"decoder": r"(decoder\_.*)|(final\_.*)", "all": ".*"}
        
        return [img_input], [outputs], layer_regex
    
    def get_loss(self, config):
        """ Get loss function. """
        class_weights = config.CLASS_WEIGHTS
        def crossentropy_loss(target, output):
            # return categorical_crossentropy(target, output, from_logits=False, class_weights=class_weights, axis=-1)
            return K.categorical_crossentropy(target, output, from_logits=False)
        def dice_coef_loss(target, output):
            dice_coef = iou_coef(target, output, mode='dice', 
                                 from_logits=True, class_weights=class_weights, 
                                 binary=False, axis=-1)
            return -K.sum(dice_coef, axis=-1)
            # return -dice_coef

        return {'crossentropy': crossentropy_loss, 'dice_coef': dice_coef_loss}[config.LOSS]
    
    def get_metrics(self, config):
        class_weights = config.CLASS_WEIGHTS
        
        def weighted_accuracy(target, output):
            score_map = categorical_accuracy(target, output, class_weights=class_weights, axis=-1)
            return K.sum(K.clip(score_map, min_value=0., max_value=None))/K.sum(K.abs(score_map))
        def mean_iou(target, output):
            iou = iou_coef(target, output, mode='iou', class_weights=None, binary=True, axis=-1)
            return K.sum(iou, axis=-1)
        def recall_metrics(target, output):
            return recall(target, output)
        def precision_metrics(target, output):
            return precision(target, output)
        
        return [weighted_accuracy, mean_iou] # , recall_metrics, precision_metrics]
    
    def train(self, train_dataset, val_dataset, epochs, config, 
              trainable_layers=".*", eval_datasets=None, **kwargs):
#         generator = lambda x, shuffle: DataLoader(
#             x, config=self.config, batch_size=config.BATCH_SIZE, shuffle=shuffle,)
        
#         if not isinstance(train_dataset, keras.utils.Sequence):
#             train_dataset = generator(train_dataset, shuffle=True)
        
#         if not isinstance(val_dataset, keras.utils.Sequence):
#             val_dataset = generator(val_dataset, shuffle=True)
        
#         if eval_datasets is not None:
#             for k in eval_datasets:
#                 if not isinstance(eval_datasets[k], keras.utils.Sequence):
#                     eval_datasets[k] = generator(eval_datasets[k], shuffle=True)

        super(self.__class__, self).train(
            train_dataset, val_dataset, epochs=epochs, config=config, 
            trainable_layers=trainable_layers, eval_datasets=eval_datasets, 
            **kwargs)
    
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
        logits = self.keras_model.predict([batch_images], batch_size=batch_size, verbose=0)
        scores = utils_keras.softmax(logits)
        labels = np.argmax(logits, axis=-1)
        
        return [_ for _ in labels], [_ for _ in scores]

    def inference_iterator(self, dataset, batch_size, image_info=lambda x: x.images, processor=None, verbose=0):
        """ Predict on a list of images with iterator. """            
#         if verbose:
#             print("Processing {} images".format(len(images)))
#             for image in images:
#                 print("image: " + image)
        
        ## Prepare/Start inference
        st_start = datetime.datetime.now()
        print("#####################################################")
        print("Inference start at: %s" % st_start.strftime('%Y-%m-%d %H:%M:%S'))
        print("\nModel Configurations: ")
        self.config.display()
        print("\nLoad weights from: %s" % self.weights_path)
        
        N = int(np.ceil(len(dataset) / batch_size))
        # N = len(dataset) // batch_size + 1
        # k = len(dataset) % self.config.BATCH_SIZE
        
        if image_info is not None:
            image_info = image_info(dataset)
        else:
            image_info = [None] * len(dataset)
        
        for i in range(N):
            s = i * batch_size
            e = min(s + batch_size, len(dataset))
            
            batch_data = np.stack([dataset[_][0] for _ in range(s, e)], axis=0)
            batch_info = [image_info[_] for _ in range(s, e)]
            
            if processor is not None:
                fn, args, kwargs = processor
                batch_data = fn(batch_data, *args, **kwargs)
            assert np.array_equal(batch_data.shape[1:], self.config.MODEL_INPUT_SHAPE), "processor failed to generate batch images with shape %s" % self.config.MODEL_INPUT_SHAPE
            batch_logits = self.keras_model.predict_on_batch(batch_data)
            batch_scores = utils_keras.softmax(batch_logits)
            batch_labels = np.argmax(batch_logits, axis=-1)
            
            for _ in zip(zip(batch_labels, batch_scores), batch_info):
                yield _
        
        ## End of inference
        st_end = datetime.datetime.now()
        print("\nInference ends at: %s" % st_end.strftime('%Y-%m-%d %H:%M:%S'))
        print("Total Run Time: %s" % (st_end - st_start))
        print("#####################################################")


class DataLoader(utils_keras.DataLoader):    
    def data_generator(self, indices):
        batch_images, batch_masks = [], []
        for i, idx in enumerate(indices):
            image, (masks, class_ids) = self.dataset[idx]
            masks, class_ids = np.stack(masks, axis=-1), np.array(class_ids)
            masks_dense = utils_keras.to_categorical(np.dot(masks, class_ids), self.config.NUM_CLASSES)
            
            if self.transform:
                masks_dense = self.transform(masks_dense, (image, masks), self.kwargs)
            
            batch_images.append(image)
            batch_masks.append(masks_dense)
        
        batch_images = np.array(self.pad_batch(batch_images))
        batch_masks = np.array(self.pad_batch(batch_masks))
            
        return batch_images, batch_masks
