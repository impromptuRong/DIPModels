""" Inherit original Mask R-CNN utils.py file. """
from .mrcnn.utils import *

def filter_res_with_nms(res, nms_threshold=0.3, remove_overlapping=True):
    """ Trim over-lapping region in detection results. """
    h, w, N = res['masks'].shape
    masks = np.zeros((h, w), dtype=np.bool)
    keep = []
    for i in range(N):
        area = np.sum(res['masks'][..., i])
        intersect = np.logical_and(res['masks'][..., i], masks)
        keep.append(np.sum(intersect)/area < nms_threshold)
        if keep[-1]:
            if remove_overlapping:
                res['masks'][intersect, i] = False
            masks = np.logical_or(res['masks'][..., i], masks)
    
    return {'rois': res['rois'][keep, :], 
            'class_ids': res['class_ids'][keep], 
            'masks': res['masks'][..., keep], 
            'scores': res['scores'][keep]}


def detect_single_image_with_flipping(predict_func, x, types,
                                      concat_d, **kwargs):
    """ Predict result by combining results from horizontal/vertical flipping.
        
        For each image A, consider A, A[:,::-1], A[::-1,:], A[::-1, ::-1]
        and their transpose. The function combine the prediction result of
        the above 8 images. Currently requires predict_func to organize results
        into numpy array, or numpy array nested in tuple and dictionary.
        
        Arguments:
            predict_func: predict function used to predict a single image/batch.
            x: image
            types: one of ('scalar', 'image', 'coord', 'boxes'), output will
                   be concat along batch dimension (axis=0) by default.
            concat_d: User specified concat dimension.
            kwargs: other parameters passed to predict_func
        
        Return:
            An object which has same structure as predict_func by combining all
            results from flipping.
        
        Examples:
            For Mask_RCNN: totally N masks
                rois: bounding box [x1, y1, x2, y2], shape=(N, 4)
                class_ids: class_ids(all 1 in our case), shape=(N,)
                scores: scores for mask, shape=(N,)
                masks: single mask image, shape=(x.height, x.width, N)
            Call:
                detect_single_image_with_flipping(model.detect, image,
                    types=dict(class_ids='scalar', scores='scalar',
                               masks='image', rois='boxes'),
                    concat_d=dict(class_ids=0, scores=0, masks=-1, rois=0))
    """
    h, w = x.shape[0], x.shape[1]
    batch_list = [x, np.transpose(x, (1, 0, 2)),
                  x[:, ::-1, :], np.transpose(x[:, ::-1, :], (1, 0, 2)),
                  x[::-1, :, :], np.transpose(x[::-1, :, :], (1, 0, 2)),
                  x[::-1, ::-1, :], np.transpose(x[::-1, ::-1, :], (1, 0, 2))]
    result_list = [predict_func([_], **kwargs)[0] for _ in batch_list]
    
    if not result_list:
        return None

    def _invert(x, flag):
        if flag == 'image':
            return [x[0], np.transpose(x[1], (1, 0, 2)),
                    x[2][:, ::-1, :], np.transpose(x[3], (1, 0, 2))[:, ::-1, :],
                    x[4][::-1, :, :], np.transpose(x[5], (1, 0, 2))[::-1, :, :],
                    x[6][::-1, ::-1, :],
                    np.transpose(x[7], (1, 0, 2))[::-1, ::-1, :]]
        elif flag == 'boxes':
            return [x[0], x[1][:,[1,0,3,2]],
                    x[2][:,[0,3,2,1]] * np.array([1, -1, 1, -1]) + np.array([0, w-1, 0, w-1]),
                    x[3][:,[1,2,3,0]] * np.array([1, -1, 1, -1]) + np.array([0, w-1, 0, w-1]),
                    x[4][:,[2,1,0,3]] * np.array([-1, 1, -1, 1]) + np.array([h-1, 0, h-1, 0]),
                    x[5][:,[3,0,1,2]] * np.array([-1, 1, -1, 1]) + np.array([h-1, 0, h-1, 0]),
                    -x[6][:,[2,3,0,1]] + np.array([h-1, w-1, h-1, w-1]),
                    -x[7][:,[3,2,1,0]] + np.array([h-1, w-1, h-1, w-1])]
        elif flag == 'coord':
            pass
        else:
            return x

    ## Revert image and coordinate back to origin
    if isinstance(result_list[0], (dict)):
        keys = result_list[0].keys()
        res = dict()
        for k in keys:
            tmp = _invert([_[k] for _ in result_list], flag=types[k])
            res[k] = np.concatenate(tmp, axis=concat_d[k])
        return res
    elif isinstance(result_list[0], (list, tuple)):
        N = len(result_list[0])
        res = [None] * N
        for k in range(N):
            tmp = _invert([_[k] for _ in result_list], flag=types[k])
            res[k] = np.concatenate(tmp, axis=concat_d[k])
        return res
    else:
        return np.concatenate(result_list, axis=concat_d)

