""" Inherit original Mask R-CNN visualize.py file. """
import skimage.io
import time

from .mrcnn.visualize import *
from .mrcnn.utils import expand_mask
from ..utils_g.utils_image import *


def display_dataset(dataset, indices=None, results=None, visualize_results=False):
    """ Display the images, masks (and inference results) in a dataset. 
        dataset: a utils_data.ImageDataset object.
        indices (optional): a list of numbers, np.random.choice(len(dataset), 2) etc. default = 'all'
        results (optional): display detection result if given.
        visualize_results (optional): whether to call visualize.display_instances in original mrcnn.
    """
    if indices is None:
        indices = range(len(dataset))
    for idx in indices:
        image_info = dataset.images[idx]
        try:
            image_raw, masks_raw = dataset.load_item(idx)
        except:
            image_path, masks_path = image_info['data']
            image_raw = rgba2rgb(skimage.io.imread(image_path))
            masks_raw = rgba2rgb(skimage.io.imread(masks_path)) if masks_path is not None else None

        start = time.time()
        image_p, masks_p = dataset[idx]
        end = time.time()
        
        plot_lists = []
        plot_lists.append(display_image(image_raw, 'Raw Image', channel_axis=-1))
        if masks_raw is not None:
            plot_lists.append(display_image(masks_raw, 'Raw Masks', channel_axis=-1))
        plot_lists.append(display_image(image_p, 'Processed Image', channel_axis=-1, in_range=(0., 1.),
                                        mean=dataset.kwargs.get('mean', 0.), std=dataset.kwargs.get('std', 1.)))
        
        if masks_p is not None:
            try:
                x = display_image(masks_p, 'Processed Masks', channel_axis=-1)
            except:
                x = display_masks(masks_p[0], 'Processed Masks', labels=masks_p[1], 
                                  label_to_val=dataset.kwargs['label_to_val'])                
            plot_lists.append(x)
        
        if results is not None:
            res = results[idx]
            plot_lists.append(display_masks(res['masks'], 'Predicted Masks', labels=res['class_ids'], 
                                            label_to_val=dataset.kwargs['label_to_val']))
        
        print("Processing image: %s [%s], %s" % (image_info['image_id'], idx, end - start))
        multiplot(plot_lists)
        plt.show()
        
        if results is not None and visualize_results:
            res = results[idx]
            img = display_image(image_p, 'Processed Image', channel_axis=-1, in_range=(0., 1.),
                                mean=dataset.kwargs['mean'], std=dataset.kwargs['std'])[0]
            display_instances(img_as('uint8')(img), res['rois'], res['masks'], res['class_ids'], 
                              list(dataset.class_info.columns), res['scores'])
            plt.show()


def display_data_loader(data_loader):
    """ Display an instance of model.DataSequence class. 
        Function is used to display batch inputs and outputs yield
        from data_loader. Debug only.
    """
    for inputs, outputs in data_loader:
        batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = inputs
        print('batch_images: %s' % image_stats(batch_images))
        print('batch_image_meta: %s' % image_stats(batch_image_meta))
        print('batch_rpn_match: %s' % image_stats(batch_rpn_match))
        print('batch_rpn_bbox: %s' % image_stats(batch_rpn_bbox))
        print('batch_gt_class_ids: %s' % image_stats(batch_gt_class_ids))
        print('batch_gt_boxes: %s' % image_stats(batch_gt_boxes))
        print('batch_gt_masks: %s' % image_stats(batch_gt_masks))
        break
    
    plots = {'image': [], 'masks': []}
    for i in range(data_loader.batch_size):
        non_zero_boxes = np.sum(batch_gt_masks[i], (0, 1)) > 0
        non_zero_masks = np.sum(batch_gt_boxes[i], 1) > 0
        assert np.all(non_zero_boxes == non_zero_masks)

        batch_masks = batch_gt_masks[i][..., non_zero_masks]
        if data_loader.config.USE_MINI_MASK:
            batch_masks = expand_mask(batch_gt_boxes[i, non_zero_boxes, ...], batch_masks, batch_images[i].shape)
        plots['image'].append(display_image(batch_images[i], 'Batch Image %s' % i, channel_axis=-1))
        plots['masks'].append(display_masks(batch_masks, 'Batch Masks %s' % i))
    multiplot(plots['image'] + plots['masks'], nrow=2, ncol=data_loader.batch_size)

    