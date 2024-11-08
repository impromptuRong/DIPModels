import os
import cv2
import copy
import json
import time
import numbers

import tifffile
import argparse
import datetime

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from utils_image import rgba2rgb

TIFF_PARAMS = {
    'tile': (1, 256, 256), 
    'photometric': 'RGB',
    # 'compress': True,
    # 'compression': 'zlib', # compression=('jpeg', 95),  # None RGBA, requires imagecodecs
    'compression': 'jpeg',
    'compressionargs': {'level': 95},
    # 'compression': 'zlib',
    # 'compressionargs': {'level': 8},
}


def to_ascii(text):
    return text.encode('ascii', 'ignore').decode('ascii')


def get_level_images(image_file, level=None):
    series = 0
    with tifffile.TiffFile(image_file) as tif:
        descp = tif.pages[0].description
        axis_map = {name: idx for idx, name in enumerate(tif.series[series].dims)}
        order = [axis_map['height'], axis_map['width']]
        order += [x for x in range(len(axis_map)) if x not in order]

        pages_list = tif.series[series].levels        
        if isinstance(level, numbers.Number):
            level = [level]
        if level is None:
            level = [x for x in range(len(pages_list))]
        level = [x % len(pages_list) for x in level]

        pages = [pages_list[idx].asarray().transpose(order) for idx in level]

        return pages, descp


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


def get_tiff_params(image, tiff_params):
    tiff_params = copy.deepcopy(tiff_params)
    if len(image.shape) == 2:  # single channel
        if 'photometric' not in tiff_params:
            tiff_params['photometric'] = 'MINISBLACK'
        assert tiff_params['photometric'].upper() in ['MINISBLACK', 'MINISWHITE']
    else:
        default_photometric = {3: 'RGB', 4: 'RGBA'}[image.shape[-1]]
        if 'photometric' not in tiff_params:
            tiff_params['photometric'] = default_photometric
    
    return tiff_params


def get_default_description(page, header, tiff_params, **kwargs):
    params = get_tiff_params(page, tiff_params)
    
    tile_w, tile_h = params['tile'][-2:]
    w0, h0 = page.shape[1], page.shape[0]
    now = datetime.datetime.now()

    descp = f"{header}\n{w0}x{h0} ({tile_w}x{tile_h}) {params['photometric']}|{now.strftime('Date = %Y-%m-%d|Time = %H:%M:%S')}"
    for k, v in kwargs.items():
        descp += f'|{k} = {v}'

    return descp


def wsi_imwrite(images, filename, description=None, scale=0.5, 
                tiff_params=TIFF_PARAMS, bigtiff=False,):
    if isinstance(images, list):
        image_list = [_ for _ in images]
    else:
        image = images
        image_list = [image]
        while image.shape[1] > 512 or image.shape[0] > 512:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            image_list.append(image)

    image_sizes = [(x.shape[1], x.shape[0]) for x in image_list]
    w0, h0 = image_sizes[0]
    tile_w, tile_h = tiff_params['tile'][-2:]   # (None/depth, w, h)
    with tifffile.TiffWriter(filename, bigtiff=bigtiff) as tif:
        for page_idx, page in enumerate(image_list):
            params = get_tiff_params(page, tiff_params)
            w, h = image_sizes[page_idx]
            if page_idx == 0:
                subfiletype = 0
                descp = description
            else:
                subfiletype = 1
                descp = f"{w0}x{h0} ({tile_w}x{tile_h}) -> {w}x{h} {params['photometric']}"
            print(f"{page.shape}", end=" ")
            tif.write(page, metadata=None, description=to_ascii(descp), 
                      subfiletype=subfiletype, **params,)
            # mpp = slide_info.get('mpp', 0.25)
            # resolution=(mpp * 1e-4, mpp * 1e-4, 'CENTIMETER')
            # subifds=N_levels-1,
            # tile = (page.tilewidth, page.tilelength) if page.tilewidth > 0 and page.tilelength > 0 else None
            # resolution=(mpp * 1e-4 * w0/w, mpp * 1e-4 * h0/h, 'CENTIMETER')
        print("Success!")


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
        meta_info = {"mpp": 0.25, "mag": "40x"}

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
                images, description = get_level_images(slide_file, level=None)
                # description from original file doesn't work with current writter (ome issue)
                description = get_default_description(images[0], args.header, TIFF_PARAMS, **slide_info)
            else:
                images = rgba2rgb(np.array(Image.open(slide_file), dtype=np.uint8))
                description = get_default_description(images, args.header, TIFF_PARAMS, **slide_info)
            print(f"Load in images with shapes: {[_.shape for _ in images]}")
            print(f"Image description: {description}")
#             except Exception as e:
#                 print(f"Failed to read file: {slide_file}.")
#                 print(e)
#                 continue

            try:
                output_file = os.path.join(output_dir, f"{slide_id}.tiff")
                wsi_imwrite(images, output_file, 
                            description=description,
                            scale=args.scale,
                            tiff_params=TIFF_PARAMS,
                            bigtiff=args.bigtiff,)
            except Exception as e:
                print(f"Failed to write file: {output_file}.")
                print(e)
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert simgle page image into pyramid tiff file', add_help=True)
    parser.add_argument('--data_path', required=True, type=str, help="Input filename or directory.")
    parser.add_argument('--image_details', default=None, type=str, help="Image details in a csv/json file.")
    parser.add_argument('--header', default=f'Tiled Image', type=str, help="Header information.")
    parser.add_argument('--meta_info', default=None, type=str, help="Meta data write to image (mag, mpp, etc).")
    parser.add_argument('--output_dir', default=None, type=str, help="Output folder.")
    parser.add_argument('--scale', default=0.5, type=float, help="Downsampling scale for pyramid tiff, default=0.5.")
    parser.add_argument('--bigtiff', action='store_true', help="BigTiff format.")

    args = parser.parse_args()
    if args.output_dir is None:
        ct = datetime.datetime.now()
        args.output_dir = f"./slide_results/{ct.strftime('%Y%m%dT%H%M%S')}"

    main(args)
