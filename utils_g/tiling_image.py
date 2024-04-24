import os
import cv2
import json
import time

import skimage
import tifffile
import argparse
import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tifffile import TiffFile
Image.MAX_IMAGE_PIXELS = None


TIFF_PARAMS = {
    'tile': (1, 256, 256), 
    'photometric': 'RGB',
    # 'compress': True,
    # 'compression': 'zlib', # compression=('jpeg', 95),  # None RGBA, requires imagecodecs
    'compression': 'jpeg',
    # 'compressionargs': 
}


def get_page(image_file, level=0):
    with open(image_file, 'rb') as fp:
        fh = TiffFile(fp)
        return fh.pages[0].asarray()


def filter_image_file(x, exts=['.png', '.jpeg', '.jpg', '.tif', '.tiff', '.ndpi']):
    filename = os.path.split(x)[-1]
    ext = os.path.splitext(x)[1].lower()

    return (not filename.startswith('.')) and (ext in exts)


def folder_iterator(folder, keep_fn=None):
    file_idx = -1
    for root, dirs, files in os.walk(folder):
        for file in files:
            if keep_fn is not None and not keep_fn(file):
                continue
            file_idx += 1
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, folder)

            yield file_idx, rel_path, file_path


def wsi_imwrite(image, filename, header, slide_info, tiff_params, bigtiff=False, scales=None, **kwargs):
    w0, h0 = image.shape[1], image.shape[0]
    if len(image.shape) == 2:  # single channel
        if 'photometric' not in tiff_params:
            tiff_params['photometric'] = 'MINISBLACK'
        assert tiff_params['photometric'].upper() in ['MINISBLACK', 'MINISWHITE']
    else:
        default_photometric = {3: 'RGB', 4: 'RGBA'}[image.shape[-1]]
        if 'photometric' not in tiff_params:
            tiff_params['photometric'] = default_photometric

    tile_w, tile_h = tiff_params['tile'][-2:]   # (None/depth, w, h)
    print(image.shape, tiff_params)
    # mpp = slide_info.get('mpp', 0.25)
    now = datetime.datetime.now()

    if scales is None:
        scales = []
        w, h, scale = w0, h0, 1
        while w > 512 or h > 512:
            scale *= 2
            scales.append(scale)
            w, h = w0 // scale, h0 // scale
    print(scales)

    with tifffile.TiffWriter(filename, bigtiff=bigtiff) as tif:
        info_message = ', '.join(f'{k}={v}' for k, v in slide_info.items())
        descp = f"{header}\n{w0}x{h0} ({tile_w}x{tile_h}) {tiff_params['photometric']}|{info_message}|{now.strftime('Date = %Y-%m-%d|Time = %H:%M:%S')}"
        for k, v in kwargs.items():
            descp += f'|{k} = {v}'
        # resolution=(mpp * 1e-4, mpp * 1e-4, 'CENTIMETER')
        tif.write(image, metadata=None, description=descp, subfiletype=0, **tiff_params,)  # subifds=len(scales), 

        for scale in scales:
            w, h = w0 // scale, h0 // scale
            image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            descp = f"{w0}x{h0} ({tile_w}x{tile_h}) -> {w}x{h} {tiff_params['photometric']}"
            # tile = (page.tilewidth, page.tilelength) if page.tilewidth > 0 and page.tilelength > 0 else None
            # resolution=(mpp * 1e-4 * w0/w, mpp * 1e-4 * h0/h, 'CENTIMETER')
            tif.write(image, metadata=None, description=descp, subfiletype=1, **tiff_params,)


def main(args):
    if os.path.isdir(args.data_path):
        keep_fn = lambda x: filter_image_file(x)
        slide_files = list(folder_iterator(args.data_path, keep_fn))
    else:
        rel_path = os.path.basename(args.data_path)
        slide_files = [(0, rel_path, args.data_path)]
    print(f"Inputs: {args.data_path} ({len(slide_files)} files observed). ")
    print(f"Outputs: {args.output_dir}")
    print("==============================")
    
    if args.image_details:
        if args.image_details.endswith('.json'):
            with open(args.image_details) as f:
                image_details = json.load(f)
        elif args.image_details.endswith('.csv'):
            image_details = pd.read_csv(args.image_details, index_col='image_id')
            # image_details['image_id'] = image_details['image_id'].apply(str)
            image_details = image_details.to_json(orient="index")
    else:
        image_details = {}
    
    if args.meta_info:
        with open(args.meta_info) as f:
            meta_info = json.load(f)
    else:
        meta_info = {"mpp":0.25, "mag":"40x"}

    for file_idx, rel_path, slide_file in slide_files:
        output_dir = os.path.join(args.output_dir, os.path.dirname(rel_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        slide_id = os.path.splitext(os.path.basename(slide_file))[0]
        res_file = os.path.join(output_dir, f"{slide_id}.tiff")
        slide_info = {**meta_info, **image_details.get(slide_id, {})}
#         if meta_data:
#             slide_info = meta_data.loc[meta_data['image_id'] == slide_id, ].to_dict(orient='records')[0]
#         else:
#             slide_info = meta_data

        print("==============================")
        if not os.path.exists(res_file):
            t0 = time.time()
            print(slide_id, slide_info)
            # try:
            if filter_image_file(slide_file, exts=['.tif', '.tiff', '.ndpi']):
                image = get_page(slide_file, level=0)
            else:
                image = np.array(Image.open(slide_file), dtype=np.uint8)
            print(f"Load in image with shape: {image.shape}")
#             except Exception as e:
#                 print(f"Failed to read file: {slide_file}.")
#                 print(e)
#                 continue

            try:
                img_file = os.path.join(output_dir, f"{slide_id}.tiff")
                wsi_imwrite(image, img_file, header=args.header, 
                            slide_info=slide_info, bigtiff=args.bigtiff,
                            tiff_params=TIFF_PARAMS,)
            except Exception as e:
                print(f"Failed to write file: {img_file}.")
                print(e)
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert simgle page image into pyramid tiff file', add_help=True)
    parser.add_argument('--data_path', required=True, type=str, help="Input filename or directory.")
    parser.add_argument('--image_details', default=None, type=str, help="Image details in a csv/json file.")
    parser.add_argument('--header', default=f'Tiled Image', type=str, help="Header information.")
    parser.add_argument('--meta_info', default=None, type=str, help="Meta data write to image (mag, mpp, etc).")
    parser.add_argument('--output_dir', default=None, type=str, help="Output folder.")
    parser.add_argument('--bigtiff', action='store_true', help="BigTiff format.")

    args = parser.parse_args()
    if args.output_dir is None:
        ct = datetime.datetime.now()
        args.output_dir = f"./slide_results/{ct.strftime('%Y%m%dT%H%M%S')}"

    main(args)
