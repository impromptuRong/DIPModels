import numpy as np
import skimage
from scipy import ndimage

## Made it support 3d image
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
    
    # Set all positive to 1
    #labels = utils_misc.masks_to_image(masks, labels=None)
    #weights[labels > 0] = 1
    return weights

def unet_class_weights(masks, class_ids):
    n_masks = masks.shape[-1]
    radius = [np.sqrt(np.sum(masks[:,:,x])) for x in range(n_masks)]
    weights = np.zeros((max(class_ids),))
    for r, class_id in zip(radius, class_ids):
        weights[class_id-1] += r
    return weights/np.sum(weights)

