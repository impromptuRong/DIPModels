""" Add validation metrics support to MaskRCNN class (mrcnn/model.py). """
import keras
import keras.backend as K
from keras.layers import Lambda
from keras.models import Model
from keras.utils import Sequence

import tensorflow as tf
import numpy as np
import datetime
import os
import re
import copy

from .mrcnn import model as model_o
from .mrcnn import utils as utils_o
from ..utils_g import utils_keras
from ..utils_g import keras_layers

def get_masks_true_and_pred(target_masks, target_class_ids, pred_masks):
    """ Extract ROIs and Masks for mask loss and evaluation index.
        
        target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32
        tensor with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])
    
    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)
    
    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)
                           
    return y_true, y_pred

def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """ Mask binary cross-entropy loss for the masks head.
        
        target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32
        tensor with values from 0 to 1.
    """
    y_true, y_pred = get_masks_true_and_pred(
        target_masks, target_class_ids, pred_masks)
    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss

def dice_coef_graph(target_masks, target_class_ids, pred_masks):
    """ Dice coefficient for masks head.
        
        target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32
        tensor with values from 0 to 1.
    """
    y_true, y_pred = get_masks_true_and_pred(
        target_masks, target_class_ids, pred_masks)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(tf.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    
    return (2.0 * intersection) / (union + intersection + 1e-8)

def iou_coef_graph(target_masks, target_class_ids, pred_masks):
    """ IOU coefficient/Jaccard for masks head.
        
        target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32
        tensor with values from 0 to 1.
        """
    y_true, y_pred = get_masks_true_and_pred(
        target_masks, target_class_ids, pred_masks)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(tf.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return intersection / (union + 1e-8)

def class_acc_graph(target_class_ids, pred_class_logits, class_weights=None):
    target_class_ids = K.reshape(target_class_ids, (-1,))
    pred_class_ids = K.reshape(pred_class_logits, (-1, pred_class_logits.get_shape()[-1]))
    
    y_true = K.cast(target_class_ids, 'int64')
    y_pred = K.cast(K.argmax(pred_class_ids, axis=-1), 'int64')
    if class_weights is not None:
        class_weights_l = tf.gather(class_weights, y_true)
        return K.sum(K.cast(K.equal(y_true, y_pred), K.floatx()) * class_weights_l)/K.sum(class_weights_l)
    else:
        return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))


############################################################
#  MaskRCNN Class
############################################################
## TODO: debug multi_gpu support. Both keras.utils.multi_gpu_model 
## and mrcnn/ParallelModel don't work. 
class MaskRCNN(utils_keras.KerasModel, model_o.MaskRCNN):
    """ Inherit from mrcnn/model.py MaskRCNN class.
        Add extra evaluation metrics and multi_gpu support.
        The actual Keras model is in the keras_model property.
    """
    def __init__(self, mode, config, model_dir=None):
        """ This class works a little bit different on model logs:
            mode: Either "training" or "inference"
            config: A Sub-class of the Config class
            model_dir: Directory to save training logs and trained weights
                if model_dir is specified, models will be saved into:
                log_dir=model_dir/yyyymmddThhmm and ignore pretrained weights path.
                In order to put output into the same folder as pretrained weights,
                use model_dir=None
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.weights_path = None
        self.model_name = config.NAME
        self.build_model(mode=mode)
    
    def build_model(self, mode, **kwargs):
        self.keras_model = self.build(mode=mode, config=self.config)
        # self.keras_model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
        self.custom_metrics = ["dice_coef", "iou_coef", "class_accuracy"]  ##, "weighted_class_accuracy"]
        self.add_metrics(self.custom_metrics)
        
        # Pre-defined layer regular expressions
        self.layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
            # Backbones
            "backbone": r"^((?!(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)).)*$",
        }
        
    def add_metrics(self, custom_metrics):
        """ Add dice coef and iou to output layers. """
        if self.mode != 'training':
            return
        
        # get the keras_model from MaskRCNN class
        use_multiGPU = hasattr(self.keras_model, "inner_model")
        model = (self.keras_model.inner_model if use_multiGPU 
                 else self.keras_model)
        
        # Extract required tensors
        target_mask, target_class_ids, mrcnn_mask =\
            model.get_layer('mrcnn_mask_loss').input
#         target_class_ids, mrcnn_class_logits, active_class_ids, class_weights =\
#             model.get_layer('mrcnn_class_loss').input
        target_class_ids, mrcnn_class_logits, active_class_ids =\
            model.get_layer('mrcnn_class_loss').input        
        # target_class_ids = K.reshape(target_class_ids, (-1,))
        
        # target_class_ids = tf.cast(target_class_ids, 'int64')
        
        # Find predictions of classes that are not in the dataset.
        #pred_class_ids = tf.argmax(mrcnn_class_logits, axis=-1)
        # TODO: Update this line to work with batch > 1. Right now it assumes all
        #       images in a batch have the same active_class_ids
        # pred_active = tf.gather(active_class_ids[0], pred_class_ids)
        
        # calculate dice_coef, iou_coef, classification accuracy
        custom_metrics_list = []
        if "dice_coef" in custom_metrics:
            dice_coef = Lambda(lambda x: dice_coef_graph(*x), name="dice_coef")(
                [target_mask, target_class_ids, mrcnn_mask])
            custom_metrics_list.append(dice_coef)
        if "iou_coef" in custom_metrics:
            iou_coef = Lambda(lambda x: iou_coef_graph(*x), name="iou_coef")(
                [target_mask, target_class_ids, mrcnn_mask])
            custom_metrics_list.append(iou_coef)
        if "class_accuracy" in custom_metrics:
            class_acc = Lambda(lambda x: class_acc_graph(*x), name="class_accuracy")(
                [target_class_ids, mrcnn_class_logits])
            custom_metrics_list.append(class_acc)
#         if "weighted_class_accuracy" in custom_metrics:
#             weighted_class_acc = Lambda(lambda x: class_acc_graph(*x), name="weighted_class_accuracy")(
#                 [target_class_ids, mrcnn_class_logits, class_weights])
#             custom_metrics_list.append(weighted_class_acc)
            
        # Rebuild keras_model with original inputs and outputs + iou/dice
        model = Model(model.inputs, model.outputs + custom_metrics_list, name='mask_rcnn')
        
        # use the mrcnn.parallel_model/ParallelModel for multi_gpu
        if use_multiGPU:
            from .mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, self.keras_model.gpu_count)
        self.keras_model = model
        return

    def compile_model(self, config):
        """ Compile model with loss and optimizers. 
            Function will use the self.compile function in original package. 
            To keep compatibility: the following hyper parameters will be changed: 
            1). self.config.GRADIENT_CLIP_NORM will be overwrite by config.GRADIENT_CLIPNORM if != default (0.0)
            2). self.config.LOSS_WEIGHTS will be overwrite by config.LOSS_WEIGHTS if != default (None)
            3). self.config.LEARNING_RATE will not be used, function will use config.LEARNING_RATE, 
                if self.config.LEARNING_RATE != config.LEARNING_RATE, function will give a warning.
            4). self.config.LEARNING_MOMENTUM will not be used, function will use config.OPTIMIZER[1]['momentum'],
                if self.config.LEARNING_MOMENTUM != config.OPTIMIZER[1]['momentum'], function will give a warning.
        """
        ## Set up trainable layers:
        print(config.TRAINABLE_LAYERS)
        self.set_trainable(config.TRAINABLE_LAYERS, self.keras_model)
        ## Organize hyperparameters for compatibility issue.
        if config.GRADIENT_CLIPNORM is not None:
            self.config.GRADIENT_CLIP_NORM = config.GRADIENT_CLIPNORM
        if config.LOSS_WEIGHTS is not None:
            self.config.LOSS_WEIGHTS = config.LOSS_WEIGHTS
        assert config.OPTIMIZER[0] == 'SGD', "Original package only support SGD optimizer. "
        if config.LEARNING_RATE != self.config.LEARNING_RATE:
            print("SGD will use start learning_rate=%s, instead of %s\n(Specify learning rate in training_config" % 
                  (config.LEARNING_RATE, self.config.LEARNING_RATE))
        if config.OPTIMIZER[1]['momentum'] != self.config.LEARNING_MOMENTUM:
            print("SGD will use momentum=%s, instead of %s\n(Specify momentum in training_config" % 
                  (config.OPTIMIZER[1]['momentum'], self.config.LEARNING_MOMENTUM))
        
        ## optimizer inside super.compile: 
        # optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, clipnorm=self.config.GRADIENT_CLIP_NORM)
        super(self.__class__, self).compile(learning_rate=config.LEARNING_RATE, 
                                            momentum=config.OPTIMIZER[1]['momentum'])
        
        # Add custom metrics: dice_coef, iou_coef, class_accuracy
        for name in self.custom_metrics:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(
                tf.reduce_mean(layer.output, keepdims=True))
    
    def train(self, *args, **kwargs):
        assert self.mode == "training", "Create model in training mode."
        super(self.__class__, self).train(*args, **kwargs)
    
#     def train(self, train_dataset, val_dataset, epochs, config, 
#               trainable_layers=".*", eval_datasets=None, **kwargs):
#         """ Use keras Sequence as data generator. 
#             layers (deprected in future): alias of trainable_layers
#             learning_rate (deprected in future): specify learning_rate in train_config instead
#             loss_weights (deprected in future): specify loss_weights in train_config instead
#         """
#         assert self.mode == "training", "Create model in training mode."
#         # Create data generators for train, val, test datasets
#         train_generator = DataLoader(train_dataset, self.config, shuffle=True,
#                                        batch_size=config.BATCH_SIZE)
#         val_generator = DataLoader(val_dataset, self.config, shuffle=True,
#                                      batch_size=config.BATCH_SIZE)
#         if eval_datasets is not None:
#             for k in eval_datasets:
#                 eval_datasets[k] = DataLoader(eval_datasets[k], self.config, shuffle=False,
#                                                 batch_size=config.BATCH_SIZE)
        
#         super(self.__class__, self).train(train_generator, val_generator, epochs, config, 
#                                           trainable_layers, eval_datasets, **kwargs)
    
    def inference(self, dataset, verbose=0):
        """ Standard inference on a dataset and return the full list of result. """
        return [r for r, _ in self.inference_iterator(dataset, image_info=None, verbose=verbose)]

    def inference_iterator(self, dataset, image_info=lambda x: x.images, processor=None, verbose=0):
        """ Inference on a dataset. Return an iterator for parallel post processing.

            dataset: utils_data.ImageDataset. Or any object that support indexing.
                     dataset[idx] should return a pair of (image, masks=None), 
            image_info: a function takes dataset as input to generate a list of 
                        the raw input information. image_info=None returns nothing.
            verbose: parse to detect.
            
            Yield a (result, image_info) for each image in dataset. 
            The image_info is the raw image_info srored in dataset.images
            The result is a dictionary which contains:
                rois: [N, (y1, x1, y2, x2)] detection bounding boxes
                class_ids: [N] int class IDs
                scores: [N] float probability scores for the class IDs
                masks: [H, W, N] instance binary masks
        """
        ## Prepare/Start inference
        st_start = datetime.datetime.now()
        print("#####################################################")
        print("Inference start at: %s" % st_start.strftime('%Y-%m-%d %H:%M:%S'))
        print("\nModel Configurations: ")
        self.config.display()
        print("\nLoad weights from: %s" % self.weights_path)
        ## mrcnn package require len(images) == self.config.BACH_SIZE, 
        N = len(dataset) // self.config.BATCH_SIZE
        k = len(dataset) % self.config.BATCH_SIZE
        
        if image_info is not None:
            image_info = image_info(dataset)
        else:
            image_info = [None] * len(dataset)
        
        ## Get default image_metas
        h, w, c = self.config.IMAGE_SHAPE
        image_meta = model_o.compose_image_meta(
            0, self.config.IMAGE_SHAPE, self.config.IMAGE_SHAPE, 
            [0, 0, h, w], 1, np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
        image_metas = np.tile(image_meta, (self.config.BATCH_SIZE, 1))
        
        for i in range(N):
            s = i * self.config.BATCH_SIZE
            e = s + self.config.BATCH_SIZE
            
            batch_data = [dataset[_][0] for _ in range(s, e)]
            batch_info = [image_info[_] for _ in range(s, e)]
            
            if processor is not None:
                fn, args, kwargs = processor
                batch_data = fn(batch_data, *args, **kwargs)
                assert np.array_equal(batch_data.shape[1:], self.config.IMAGE_SHAPE), "processor failed to generate batch images with shape %s" % self.config.IMAGE_SHAPE
                batch_result = super(self.__class__, self).detect_molded(batch_data, image_metas, verbose=verbose)
            else:
                batch_result = super(self.__class__, self).detect(batch_data, verbose=verbose)
            for _ in zip(batch_result, batch_info):
                yield _
        ## Pad blank image to the end of the dataset. 
        if k > 0:
            batch_data = [dataset[_][0] for _ in range(-k, 0)]
            batch_info = [image_info[_] for _ in range(-k, 0)]
            
            if processor is not None:
                fn, args, kwargs = processor
                batch_data = fn(batch_data, *args, **kwargs)
                padding = np.zeros((self.config.BATCH_SIZE - k,) + tuple(self.config.IMAGE_SHAPE))
                batch_data = np.vstack([batch_data, padding])
                assert np.array_equal(batch_data.shape[1:], self.config.IMAGE_SHAPE), "processor failed to generate batch images with shape %s" % self.config.IMAGE_SHAPE
                batch_result = super(self.__class__, self).detect_molded(batch_data, image_metas, verbose=verbose)
            else:
                batch_data = (batch_data + [np.zeros(self.config.IMAGE_SHAPE)] * (self.config.BATCH_SIZE - k))
                batch_result = super(self.__class__, self).detect(batch_data, verbose=verbose)
            
            for _ in zip(batch_result[:k], batch_info):
                yield _
        
        ## End of inference
        st_end = datetime.datetime.now()
        print("\nInference ends at: %s" % st_end.strftime('%Y-%m-%d %H:%M:%S'))
        print("Total Run Time: %s" % (st_end - st_start))
        print("#####################################################")


class DataLoader(Sequence):
    def __init__(self, dataset, config, shuffle=True, random_rois=0, 
                 batch_size=1, detection_targets=False):
        self.dataset = dataset
        self._len = len(self.dataset)
        self.indices = np.random.permutation(self._len)
        self.config = config
        self.random_rois = random_rois
        self.batch_size = batch_size
        self.shuffle = shuffle
        # detection_targets is only used for debug, so deprecated here.
        self.detection_targets = detection_targets
        
        if self.config.USE_MINI_MASK:
            self.gt_masks_shape = self.config.MINI_MASK_SHAPE
        else:
            self.gt_masks_shape = (self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1])
        

    def __len__(self):
        return int(np.ceil(1.0 * len(self.dataset) / self.batch_size))

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

    def data_generator(self, indices):
        # Anchors: [anchor_count, (y1, x1, y2, x2)]
        if hasattr(self.config, 'BACKBONE_SHAPES'):
            backbone_shapes = self.config.BACKBONE_SHAPES
        else:
            backbone_shapes = model_o.compute_backbone_shapes(self.config, self.config.IMAGE_SHAPE)
        anchors = utils_o.generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                   self.config.RPN_ANCHOR_RATIOS,
                                                   backbone_shapes,
                                                   self.config.BACKBONE_STRIDES,
                                                   self.config.RPN_ANCHOR_STRIDE)
        
        # Init batch arrays
        batch_image_meta = np.zeros(
            (self.batch_size,) + (self.config.IMAGE_META_SIZE,), dtype=np.float32)
        batch_rpn_match = np.zeros(
            [self.batch_size, anchors.shape[0], 1], dtype=np.int32)
        batch_rpn_bbox = np.zeros(
            [self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=np.float32)
        batch_images = np.zeros(
            (self.batch_size,) + tuple(self.config.IMAGE_SHAPE), dtype=np.float32)
        batch_gt_class_ids = np.zeros(
            (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
        batch_gt_boxes = np.zeros(
            (self.batch_size, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)
        batch_gt_masks = np.zeros(
            (self.batch_size, self.gt_masks_shape[0], self.gt_masks_shape[1],
             self.config.MAX_GT_INSTANCES), dtype=np.bool)
        
        if self.random_rois:
            batch_rpn_rois = np.zeros(
                (batch_size, self.random_rois, 4), dtype=np.int32)
            if self.detection_targets:
                batch_rois = np.zeros(
                    (batch_size,) + (self.config.TRAIN_ROIS_PER_IMAGE, 4), dtype=np.int32)
                batch_mrcnn_class_ids = np.zeros(
                    (batch_size,) + (self.config.TRAIN_ROIS_PER_IMAGE,), dtype=np.int32)
                batch_mrcnn_bbox = np.zeros(
                    (batch_size,) + (self.config.TRAIN_ROIS_PER_IMAGE, self.config.NUM_CLASSES, 4), dtype=np.float32)
                batch_mrcnn_mask = np.zeros(
                    (batch_size,) + (config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], 
                                     config.MASK_SHAPE[1], config.NUM_CLASSES), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            # Get GT bounding boxes and masks for image.
            image, image_meta, gt_class_ids, gt_boxes, gt_masks =\
                load_image_gt(self.dataset, self.config, idx,  
                              use_mini_mask=self.config.USE_MINI_MASK)
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue
            
            # RPN Targets
            rpn_match, rpn_bbox = model_o.build_rpn_targets(image.shape, anchors, 
                                                            gt_class_ids, gt_boxes, self.config)
            
            # Mask R-CNN Targets
            if self.random_rois:
                rpn_rois = model_o.generate_random_rois(
                    image.shape, random_rois, gt_class_ids, gt_boxes)
                if self.detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        model_o.build_detection_targets(
                        rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)
            
            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[i] = image_meta
            batch_rpn_match[i] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[i] = rpn_bbox
            batch_images[i] = model_o.mold_image(image.astype(np.float32), self.config)
            batch_gt_class_ids[i, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[i, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[i, :, :, :gt_masks.shape[-1]] = gt_masks
            if self.random_rois:
                batch_rpn_rois[i] = rpn_rois
                if detection_targets:
                    batch_rois[i] = rois
                    batch_mrcnn_class_ids[i] = mrcnn_class_ids
                    batch_mrcnn_bbox[i] = mrcnn_bbox
                    batch_mrcnn_mask[i] = mrcnn_mask

        inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
        outputs = []

        if self.random_rois:
            inputs.extend([batch_rpn_rois])
            if self.detection_targets:
                inputs.extend([batch_rois])
                # Keras requires that output and targets have the same number of dimensions
                batch_mrcnn_class_ids = np.expand_dims(
                    batch_mrcnn_class_ids, -1)
                outputs.extend(
                    [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

        return inputs, outputs


def load_image_gt(dataset, config, idx, use_mini_mask=False):
    """ Load and return ground truth data for an image (image, mask, bounding boxes).
        (Removed augment, augmentation from original load_image_gt)
        
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image, (masks, class_ids) = dataset[idx]
    class_ids = np.array(class_ids)
    # assert len(masks) == len(class_ids), "The size of masks should be same as labels!"
    
    ## In case masks is [], generate a (h, w, 0) boolean mask
    if len(masks):
        mask = np.moveaxis(np.array(masks), 0, -1)
    else:
        mask = np.zeros(shape=tuple(config.IMAGE_SHAPE[:-1]) + (0,), dtype=bool)
    assert mask.ndim == len(config.IMAGE_SHAPE), "Boolean mask has invalid shape!"
    assert mask.shape[-1] == len(class_ids), "Mask channels is inconsistent with class_ids (%d != %d)!" % (mask.shape[-1], len(class_ids))
    
    original_shape = image.shape
    image, window, scale, padding, crop = utils_o.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils_o.resize_mask(mask, scale, padding, crop)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils_o.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    rss = dataset.images[idx]["source"]
    active_class_ids = np.array(dataset.get_labels_from_source(rss),
                                dtype=np.int32)

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils_o.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = model_o.compose_image_meta(idx, original_shape, image.shape,
                                            window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask
