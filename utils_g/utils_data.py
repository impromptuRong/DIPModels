import numpy as np
import pandas as pd
import datetime
import pickle
from sklearn.model_selection import train_test_split


def save_to_pickle(filename, x):
    with open(filename, 'wb') as output:
        pickle.dump(x, output, pickle.HIGHEST_PROTOCOL)
    return


def train_val_split(dataset, validation_split=0.3, random_split=True):
    if validation_split > 0 and validation_split < 1:
        # split is a ratio:
        validation_split = int(len(dataset) * validation_split)
    
    indices = list(range(len(dataset))) # start with all the indices in training set
    if random_split:
        valid_idx = np.random.choice(indices, size=validation_split, replace=False)
        train_idx = np.array(list(set(indices) - set(valid_idx)))
    else:
        valid_idx, train_idx = indices[:validation_split], indices[validation_split:]
    
    return train_idx, valid_idx


def generate_image_info(image_id, file_name, image_size, 
                        date_captured=datetime.datetime.utcnow().isoformat(' '),
                        license_id=1, coco_url="", flickr_url=""):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[1],
        "height": image_size[0],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url,
    }


def generate_annotation_info(ann=None, **kwargs):
    """ generate annotation info from bbox, label and mask. 
        {"id" : int, "image_id" : int, "category_id" : int, 
         "segmentation" : RLE or [polygon], "area" : float, 
         "bbox" : [x,y,width,height], "iscrowd" : 0 or 1}
    """
    res = {"id": None, "image_id": None, "category_id": None, 
           "segmentation": None, "area": None, 
           "bbox": None, "iscrowd": None}
    dtypes = {"id": int, "image_id": int, "category_id": int, 
              "area": float, "bbox": list}
    res.update(ann)
    res.update(kwargs)
    ## TODO: Add a type and size check here
    assert res["iscrowd"] in [0, 1]
    if res["iscrowd"] == 1:
        assert isinstance(res["segmentation"], dict)
        assert "counts" in res["segmentation"]
        assert "size" in res["segmentation"]
    else:
        assert isinstance(res["segmentation"], list)
        for x in res["segmentation"]:
            assert len(x) > 4
    for k in dtypes:
        assert isinstance(res[k], dtypes[k])
    
    return res


def convert_to_coco(dataset, description='coco data', labels=None, root_path=None, **kwargs):
    """ convert images, masks, bounding boxes, labels into coco_detection format. 
        Coco-detection data format: http://cocodataset.org/#format-data
        This function will take the outputs from self.__getitem__(idx)
        So pre-define self.__getitem__(idx) and return (image, annotations, ...)
    Arguments:
        dataset: an ImageDataset with self.__getitem__(idx) that returns (image, annotations, ...)
            image: a numpy array image. (limited to rgb image by coco)
            annotations:
                masks: a list of: boolean array, rle, polygons.
                labels: a list or np.array
                bboxes: the bounding box with format (x1, y1, x2, y2)
            for mask image, utils_image.split_masks can directly generate masks + labels + bboxes
            for list of binary mask, use binary_mask_to_coco_annotation

        labels (None): a list of names, optional if categories is provided in kwargs
        root_path (None): the root folder of dataset. default will use the raw "image_path"
        kwargs: info, licenses, categories
    """
    dt = datetime.datetime.utcnow()
    default_info = {
        "description": description,
        "year": dt.year,
        "contributor": os.getuid(),
        "date_created":dt.isoformat(' '),
    }
    INFO = kwargs.setdefault('info', default_info)

    default_licenses = {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
    LICENSES = kwargs.setdefault('licenses', default_licenses)

    CATEGORIES = kwargs.setdefault('categories', None)
    if CATEGORIES is None:
        labels = dataset.labels if labels is None else labels
        CATEGORIES = [{'id': idx, 'name': str(_), 'supercategory': 'default',} for idx, _ in enumerate(labels)]

    ## Basic Coco-Detection format
    res = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    ## add instances to annotation file
    object_idx = 1
    for image_idx, (image_info, items) in enumerate(zip(dataset.images, dataset), 1):
        image, annotations = items[0], items[1]
        ## image_notes
        image_id = image_info["image_id"]
        image_path = image_info["data"][0] if root_path is None else root_path.format(image_id=image_id)
        image_notes = generate_image_info(image_idx, image_path, image.shape[:-1])
        res["images"].append(image_notes)

        ## annotations_notes
        for ann in annotations:
            masks_notes = generate_annotation_info(ann, id=object_idx, image_id=image_idx)
            res["annotations"].append(masks_notes)
            object_idx += 1

    return res


class ImageDataset(object):
    """ Dataset class (modified from MaskRCNN Dataset)
        Inherit this class for different tasks. 
        Compatible with keras_dataloader, pytorch_dataloader,
    """
    def __init__(self, labels=[''], source=[''], processor=None, add_bg=True, **kwargs):
        """ labels: the labels in the dataset.
            source: name of the dataset
            processor: a function for preprocessing.
            kwargs: global args for processor
        """
        # map image_id -> X, y, scc
        self.images = list()
        self.indices = dict()
        self.processor = processor
        self.kwargs = kwargs
        self.class_info = pd.DataFrame(False, index=source, columns=labels, dtype=bool)
        if add_bg:
            self.class_info.insert(0, 'bg', True)
    
    @property
    def labels(self):
        return self.class_info.columns.values.tolist()
    
    @property
    def source(self):
        return self.class_info.index.values.tolist()
    
    def load_from(self, iterator, source='', labels=None, **kwargs):
        if labels is None:
            labels = self.class_info.columns.values
        self.class_info.loc[source, labels] = True
        for image_id, data, image_kwargs in iterator:
            ## update parameters with the following sequences: 
            ## Dataset class kwargs (self.kwargs), load_from fn kwargs, iterator specified image_kwargs
            args = dict()
            args.update(self.kwargs)
            args.update(kwargs)
            args.update(image_kwargs)
            self.add_image(image_id, data, source, **args)
    
    def add_image(self, image_id, data, source, **kwargs):
        """ Add image information to Dataset.
            image_id: image_id should be unique.
            data: tuple of (X, y) or similar (X_path, y_path) etc.
                  implement __getitem__ for further io. y=None for inference.
            kwargs: other information, including parameters for processor.
        """
        assert image_id not in self.indices, "image_id already exists"
        self.images.append({"image_id": image_id, "data": data, "source": source, "kwargs": kwargs})
        self.indices[image_id] = len(self.images) - 1
    
    def update_attributes(self, image_ids=None, **kwargs):
        """ Update attributes in Dataset. 
            If image_ids is None, new values will be add to self.kwargs. (global)
            Otherwise, new values will be add to self.images[idx]['kwargs'] (for each image)
        """
        if image_ids is None:
            self.kwargs.update(kwargs)
        else:
            for _ in image_ids:
                idx = self.indicex[_]
                self.images[idx]['kwargs'].update(kwargs)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """ Take the image index and return the data (dataset[idx]). 
            Apply processor here.
        """
        raise NotImplementedError("Subclass must implement this function")
    
    def load_item(self, idx):
        raise NotImplementedError("Subclass must implement this function")
    
    def get_labels_from_source(self, source):
        """ Get labels from a given source. 
            (replace get_source_class_ids (load_image_gt) in matterplot mrcnn package.)
            Return a boolean list x, len(x) = self.class_info.ncol
        """
        return self.class_info.loc[source].tolist()
    
    def get_source_with_labels(self, labels):
        """ Get source contains a given label. 
            Return a boolean list x, len(x) = self.class_info.nrow
        """
        return self.class_info.loc[:, labels].tolist()
    
    def convert_to_coco(self, description="coco data", labels=None, root_path=None, **kwargs):
        return convert_to_coco(self, description, labels=labels, root_path=root_path, **kwargs)


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

