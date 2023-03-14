import os
import re
import numpy as np
import pandas as pd


def load_masks_from_dir(folder, labels, pattern=r"(.png$)|(.jpg$)"):
    """ Merge all mask files in a folder into one numpy array. """
    masks = []
    for file in os.listdir(folder):
        if not file.startswith('.') and re.search(pattern, file):
            x = skimage.io.imread(os.path.join(folder, file))
            if x.ndim[-1] == 4:
                x = utils_image.rgba2rgb(x, binary_alpha=True)
            masks.append(utils_image.label_masks(x, labels))
    return np.stack(masks, axis=-1)


def run_length_encode(masks):
    h, w, d = masks.shape
    res = []
    for i in range(d):
        bs = np.where(masks[:,:,i].T.flatten())[0] + i * h * w
        
        rle = []
        prev = -2
        for b in bs:
            if (b > prev + 1):
                rle.extend((b + 1, 0))
            rle[-1] += 1
            prev = b

        #https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
        #if len(rle)!=0 and rle[-1]+rle[-2] == x.size:
        #    rle[-2] = rle[-2] -1  #print('xxx')
        res.append(rle)
    # rle = ' '.join([str(r) for r in rle])
    return res


def run_length_decode(rles, h, w, fill_value=1):
    res = []
    # rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for x in rles:
        mask = np.zeros((h * w), np.uint8)
        rle = np.array(x).reshape(-1, 2)
        for r in rle:
            start = (r[0] - 1) % (w * h)
            end = start + r[1]
            mask[start : end] = fill_value
        res.append(mask.reshape(w, h).T)
    
    return np.stack(res, axis=-1)


def get_masks_from_rles(df_rles, image_id, h, w, fill_value=1):
    if isinstance(df_rles, str):
        df_rles = pd.read_csv(df_rles, index_col="ImageId")
    res = []
    for rle in df_rles.loc[image_id, "EncodedPixels"].tolist():
        res.append([int(x) for x in rle.split(' ')])
    print(len(res))
    return run_length_decode(res, h, w, fill_value)


#### TODO: Rewrite this function ####
def save_rles_to_file(masks_dict, filename):
    """ Write masks into files with rle_encoding
    
    Augment:
        masks_dict: a dictionary with ImageId as key and masks matrix as value
        filename: output filename
    """
    with open(filename, "w") as myfile:
        myfile.write("ImageId, EncodedPixels\n")
        for image_id, masks in masks_dict.iteritems():
            print(image_id)
            RLE = rle_encoding(masks[:, :, j])[0]
            myfile.write(image_id + "," + " ".join([str(k) for k in RLE]) + "\n")

## May not need this function anymore
#def excel_to_csv(excel_file, csv_file):
#    """ Transfer excel file to csv file """
#    wb = xlrd.open_workbook(excel_file)
#    sh = wb.sheet_by_name('Sheet1')
#    out_file = open(csv_file, 'w')
#    wr = csv.writer(out_file, quoting=csv.QUOTE_ALL)

#    for rownum in range(sh.nrows):
#        wr.writerow(sh.row_values(rownum))

#    out_file.close()
