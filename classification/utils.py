import time
import numpy as np

from ..utils_g.utils_image import *


def softmax(logits):
    probs = np.exp(logits - np.amax(logits, axis=-1, keepdims=True))
    return probs/np.sum(probs, axis=-1, keepdims=True)


def display_dataset(dataset, indices=None, results=None, display_fn=None, 
                    titles=['Raw Image', 'Processed Image'], display_panels=None, **kwargs):
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
        image_path, y = image_info['data']
        
        if 'image_reader' in image_info['kwargs']:
            image_raw = image_info['kwargs']['image_reader'](image_path)
        else:
            image_raw = skimage.io.imread(image_path)
        
        start = time.time()
        image_p, masks_p = dataset[idx]
        end = time.time()
        
        plot_lists = [display_image(image_raw, titles[0], cmap='gray'), 
                      display_image(image_p, titles[1], mean=dataset.kwargs.get('mean', 0.), 
                                    std=dataset.kwargs.get('std', 1.), in_range=image_p_in_range, cmap='gray'), 
                     ]
        
        ## Output images, results and other stats
        print("Processing image: %s, %s" % (image_info['image_id'], end-start))
        print(y)
        if display_fn is not None:
            sr = display_fn(image_info, image_raw, image_p, 
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
    for batch_images, batch_y in data_loader:
        print('batch_images: %s' % image_stats(batch_images))
        
        if batch_y is not None:
            print('batch_y:')
            print(batch_y)
        break
    
    for i in range(data_loader.batch_size):
        plots = list()
        plots.append(display_image(batch_images[i], 'image %s' % i))
        multiplot(plots, nrow=1, ncol=len(plots))
        
        plt.show()


def display_data_sequence(data_sequence, classes):
    """ Display an instance of model.DataSequence class. 
        Function is used to display batch inputs and outputs yield
        from data_sequence. Debug only.
    """
    for inputs, outputs in data_sequence:
        batch_images = inputs[0]
        print('batch_images: %s' % image_stats(batch_images))
        for name, y in zip(classes, outputs):
            print('batch_%s: %s' % (name, y.shape))
        
        plots = list()
        for i in range(len(batch_images)):
            img = batch_images[i]
            img_tag = list()
            for name, y in zip(classes, outputs):
                tmp = y[i]
                idx = np.nonzero(tmp)
                y_tag = ','.join(['%s_%s=%s' % (name, b, a) for a, b in zip(tmp[idx], idx[0].tolist())])
                img_tag.append(y_tag)
            img_tag = ' '.join(img_tag)
            plots.append(display_image(img, 'batch image %s: \n%s' % (i, img_tag)))
        multiplot(plots, nrow=int((len(plots)+3)//4), ncol=4)
        break


