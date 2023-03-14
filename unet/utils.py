import time
import numpy as np
import skimage
import skimage.morphology
from scipy import ndimage

from ..utils_g.utils_image import *


## consider moving the following support functions into keras_layers
# def np_dice_coef(y_true, y_pred):
#     tr = y_true.flatten()
#     pr = y_pred.flatten()

#     return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)


# def precision(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)

#     true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred_f, 0, 1)))

#     return true_positives / (predicted_positives + K.epsilon())


# def recall(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)

#     true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))

#     return true_positives / (possible_positives + K.epsilon())


# def f1_score(y_true, y_pred):
#     return 2. / (1. / recall(y_true, y_pred) + 1. / precision(y_true, y_pred))


# def dice_coef_single(y_true, y_pred, smooth=1):
#     intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#     union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
#     return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
    
#     prod = y_true_f * y_pred_f
#     intersection = K.sum(prod)
    
#     numer = (2. * intersection + smooth)
#     denom = (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
#     return numer / denom


# def soft_dice(target, output):
#     output = K.softmax(output)
#     numerator = 2. * K.sum(target * output, axis=-1)
#     denominator = K.sum(K.square(target) + K.square(output), axis=-1)
#     return 1 - K.mean(numerator / (denominator + K.epsilon()))
        

# def dice_bce_loss(y_true, y_pred):
#     from keras.losses import binary_crossentropy
#     return binary_crossentropy(y_true, y_pred) + (1.-dice_coef(y_true, y_pred))


def unet_border_weights(masks, w0=1, sigma=4, r=3):
    ## ref : https://www.kaggle.com/piotrczapla/tensorflow-u-net-starter-lb-0-34/notebook
    n_masks = masks.shape[-1]
    inner_masks = np.stack([skimage.morphology.binary_erosion(masks[:,:,x], skimage.morphology.disk(r)) 
                            for x in range(n_masks)], axis=-1)
    outer_masks = np.stack([skimage.morphology.binary_dilation(masks[:,:,x], skimage.morphology.disk(r)) 
                            for x in range(n_masks)], axis=-1)
    border = np.logical_xor(outer_masks, inner_masks)

    # calculate weight for important pixels
    distances = np.array([ndimage.distance_transform_edt(border[:,:,x] == 0) 
                          for x in range(n_masks)])
    shortest_dist = np.sort(distances, axis=0)
    # distance to the border of the nearest cell
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)
    d2 = np.zeros(d1.shape)
    weights = w0 * np.exp(-(d1+d2)**2/(2*sigma**2)).astype(np.float32)
    #weights[weights < 1e-4] = 0.
    
    # Set pixel inside inner_masks (positive) to w0
    weights = np.where(np.any(inner_masks, axis=-1), w0, weights)
    return weights


def deeplab_border_weights(masks, w0=1, r=3):
    n_masks = masks.shape[-1]
    inner_masks = np.stack([skimage.morphology.binary_erosion(masks[:,:,x], skimage.morphology.disk(r)) 
                            for x in range(n_masks)], axis=-1)
    outer_masks = np.stack([skimage.morphology.binary_dilation(masks[:,:,x], skimage.morphology.disk(r)) 
                            for x in range(n_masks)], axis=-1) - 1
    weights = inner_masks + outer_masks
    weights[weights == -1] = -1./np.sum(outer_masks)
    weights[weights == 1] = 1./np.sum(inner_masks)
    return weights


def unet_class_weights(masks, class_ids):
    n_masks = masks.shape[-1]
    radius = [np.sqrt(np.sum(masks[...,i])) for i in range(n_masks)]
    weights = np.zeros((max(class_ids),))
    for r, class_id in zip(radius, class_ids):
        weights[class_id-1] += r
    return weights/np.sum(weights)


## Re-write this function, unet_class_weights should be set globally:
def median_frequency_balancing(masks_list, labels_list, num_classes=None):
    """ Median frequency balancing. 
        w[c] = median_freq/freq(c)
    """
    if num_classes is None:
        num_classes = np.max(np.unique(labels)) + 1
    freq = np.zeros(shape=(num_classes,))
    for masks, labels in zip(masks_list, labels_list):
        for mask, label in zip(masks, labels):
            freq[label] += np.sum(mask)/mask.size()
    return np.median(freq)/freq


def transform_masks(inputs, raw_data, **kwargs):
    bg_masks, cl_masks = inputs
    image, masks = raw_data
    bg_score = seg_utils.unet_border_weights(masks, **kwargs['border_weights'])
    
    n_classes = cl_masks.shape[-1]
    cl_score = seg_utils.unet_class_weights(masks, n_classes)
    
    return bg_masks * bg_score, cl_masks * cl_scores


def display_dataset(dataset, indices=None, results=None, result_type='masks', display_fn=None, 
                    titles=['Raw Image', 'Raw Masks', 'Processed Image', 'Processed Masks', 'Predicted Masks'], 
                    display_panels=None, **kwargs):
    """ Display the images, masks (and inference results) in a dataset. 
        dataset: a utils_data.ImageDataset object.
        indices (optional): a list of numbers, np.random.choice(len(dataset), 2) etc. default = 'all'
        results (optional): display detection result if given.
    """
    if indices is None:
        indices = range(len(dataset))
    image_p_in_range = kwargs.setdefault('in_range', (0., 1.))
    
    records = []
    for idx in indices:
        image_info = dataset.images[idx]
        image_path, masks_path = image_info['data']
        
        if 'image_reader' in image_info['kwargs']:
            image_raw = image_info['kwargs']['image_reader'](image_path)
        else:
            image_raw = skimage.io.imread(image_path)
        
        if 'masks_reader' in image_info['kwargs']:
            masks_raw = image_info['kwargs']['masks_reader'](masks_path) if masks_path is not None else None
        else:
            masks_raw = skimage.io.imread(masks_path) if masks_path is not None else None

        start = time.time()
        image_p, masks_p = dataset[idx]
        end = time.time()
        
        plot_lists = [display_image(image_raw, titles[0]), 
                      display_image(masks_raw, titles[1]) if masks_raw is not None else None,
                      display_image(image_p, titles[2], mean=dataset.kwargs.get('mean', 0.), 
                                    std=dataset.kwargs.get('std', 1.), in_range=image_p_in_range), 
                     ]
        if masks_p is not None:
            if len(masks_p) == 2:
                x = display_masks(masks_p[0], titles[3], labels=masks_p[1], 
                                  label_to_val=dataset.kwargs['label_to_val'])
            else:
                x = display_image(masks_p, titles[3])
            plot_lists.append(x)
        
        if results is not None:
            assert result_type in ['image', 'masks_with_bg', 'masks_without_bg']
            res = results[idx]
            if result_type == 'image':
                x = display_image(res, titles[-1])
            elif result_type == 'masks_with_bg':
                x = display_masks(res[..., 1:], titles[-1], label_to_val=dataset.kwargs['label_to_val'])
            else:
                x = display_masks(res, titles[-1], label_to_val=dataset.kwargs['label_to_val'])
            plot_lists.append(x)
        
        ## Output images, results and other stats
        print("Processing image: %s, %s" % (image_info['image_id'], end-start))
        if display_fn is not None:
            sr = display_fn(image_info, image_raw, masks_raw, image_p, masks_p, 
                            res if results is not None else None, **kwargs)
            records.append((image_info['image_id'], sr))
        
        if display_panels is not None:
            plot_lists = [plot_lists[_] for _ in display_panels]
        multiplot(plot_lists)
        
        plt.show()

    return records


def display_data_loader(data_loader):
    """ Display an instance of model.DataLoader class. 
        Function is used to display batch inputs and outputs yield
        from data_loader. Debug only.
    """
    for batch_images, batch_masks in data_loader:
        print('batch_images: %s' % image_stats(batch_images))
        
        if batch_masks is not None:
            print('batch_masks: %s' % image_stats(batch_masks))
        break
    
    for i in range(data_loader.batch_size):
        plots = list()
        plots.append(display_image(batch_images[i], 'image %s' % i))
        for k in range(data_loader.config.NUM_CLASSES):
            plots.append(display_image(batch_masks[i][..., k], 'masks %s - %s' % (i, k)))
    
        multiplot(plots, nrow=1, ncol=len(plots))
        
        plt.show()
