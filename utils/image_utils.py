#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
import json
import cupy as cp
import numpy as np
from tqdm import tqdm
import rawpy
import cv2
import sys 

# constants
_EXIF_KEYS = (
    'BlackLevel',
    'WhiteLevel',
    'AsShotNeutral',
    'ColorMatrix2',
    'NoiseProfile',
)

# Color conversion from reference illuminant XYZ to RGB color space.
# See http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html.
_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])

 
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# FORK: defining function to extract EXIF metadata from image file
#       metadata should be stored in JSON format, in the same directory as corresponding image files, with the same filename as the image files
#
# params:
#       path    ->  Path to image filename
def fetchEXIF(path):  
    # load EXIF data from file
    root, _ = os.path.splitext(path)
    path = root + ".json"
    exif_data = None
    with open(path, 'r') as json_file:
        exif_data = json.load(json_file)[0]

    return exif_data

def bilinear_demosaic(bayer):
    # define inner functions
    def reshape_quads(*planes):
        """Reshape pixels from four input images to make tiled 2x2 quads."""
        planes = cp.stack(planes, -1)
        shape = planes.shape[:-1]

        # Create [2, 2] arrays out of 4 channels.
        zup = planes.reshape(shape + (2, 2,))
        
        # Transpose so that x-axis dimensions come before y-axis dimensions.
        zup = cp.transpose(zup, (0, 2, 1, 3))
        
        # Reshape to 2D.
        zup = zup.reshape((shape[0] * 2, shape[1] * 2))
        return zup

    def bilinear_upsample(z):
        """2x bilinear image upsample."""
        # Using np.roll makes the right and bottom edges wrap around. The raw image
        # data has a few garbage columns/rows at the edges that must be discarded
        # anyway, so this does not matter in practice.
        # Horizontally interpolated values.
        zx = .5 * (z + cp.roll(z, -1, axis=-1))
        # Vertically interpolated values.
        zy = .5 * (z + cp.roll(z, -1, axis=-2))
        # Diagonally interpolated values.
        zxy = .5 * (zx + cp.roll(zx, -1, axis=-2))
        return reshape_quads(z, zx, zy, zxy)

    def upsample_green(g1, g2):
        """Special 2x upsample from the two green channels."""
        z = cp.zeros_like(g1)
        z = reshape_quads(z, g1, g2, z)
        alt = 0

        # Grab the 4 directly adjacent neighbors in a "cross" pattern.
        for i in range(4):
            axis = -1 - (i // 2)
            roll = -1 + 2 * (i % 2)
            alt = alt + .25 * cp.roll(z, roll, axis=axis)
            
        # For observed pixels, alt = 0, and for unobserved pixels, alt = avg(cross),
        # so alt + z will have every pixel filled in.
        return alt + z

    # assign bayer pattern to red, green and blue variables
    r, g1, g2, b = [bayer[(i//2)::2, (i%2)::2] for i in range(4)]

    # upsample red components
    r = bilinear_upsample(r)

    # upsample blue components
    # Flip in x and y before and after calling upsample, as bilinear_upsample
    # assumes that the samples are at the top-left corner of the 2x2 sample.
    b = bilinear_upsample(b[::-1, ::-1])[::-1, ::-1]

    # upsample green components
    g = upsample_green(g1, g2)
    rgb = cp.stack([r, g, b], -1)

    return rgb


def loadRawImages(path_to_folder, downsample=True, img_idx=None):
    # prepare logger
    raw_dataset = []
    folder = os.listdir(path_to_folder)
    for file in folder:
        if file.endswith(".DNG") or file.endswith(".dng"):
            raw_dataset.append(file)

    progress = tqdm(len(raw_dataset), desc="Demosaicing and downscaling RAW Images...", unit="/" + str(len(raw_dataset)) + " images")

    raw_images = {}
    bayer_masks = {}

    # hard coded paqrams for quick testing
    blackLevel = cp.array([[528.0]]) # need to be formatted like this to avoid overflows on certain rgb components at demosaic
    whiteLevel = cp.array([[4095.0]])

    image0 = None
    exifs = []

    # iterate through all image files in dataset 
    for i, raw_file in enumerate(raw_dataset):
        path = os.path.join(path_to_folder, raw_file)

        if img_idx is not None and i != img_idx:
            raw_images[path] = None
            bayer_masks[path] = None
            continue

        # fetch metadata
        exifs.append(fetchEXIF(path))

        # load raw image
        raw = cp.array(rawpy.imread(path).raw_image).astype(cp.float32)
        raw = ((raw - blackLevel)/ (whiteLevel - blackLevel)).astype(cp.float32)
        raw = cp.clip(raw, 0, 1)
        

        # bilinear demosaic
        raw_demosaic = bilinear_demosaic(raw)

        image0 = raw_demosaic if i == 0 or img_idx is not None else image0

        # get bayer mask for image
        pix_x, pix_y = np.meshgrid(np.arange(raw_demosaic.shape[1]), np.arange(raw_demosaic.shape[0]))
        bayer_mask = pixels_to_bayer_mask(pix_x, pix_y)

        if downsample:
            # resize image for memory management
            scale = raw_demosaic.shape[1] / 1600
            resolution = (int(raw_demosaic.shape[1] / scale), int(raw_demosaic.shape[0] / scale))
            raw_demosaic = cv2.resize(cp.asnumpy(raw_demosaic), resolution)
            

            # resize bayer mask
            resolution = (int(bayer_mask.shape[1] / scale), int(bayer_mask.shape[0] / scale))
            bayer_mask = cv2.resize(bayer_mask, resolution)

        raw_images[path] = cp.asnumpy(raw_demosaic)
        bayer_masks[path] = bayer_mask

        # update progress logger
        progress.update(1)


    # use first image as reference for gamma mapping
    meta = {}
    exif = exifs[0]
    # Convert from array of dicts (exifs) to dict of arrays (meta).
    for key in _EXIF_KEYS:
        exif_value = exif.get(key)
        if exif_value is None:
            continue
        # Values can be a single int or float...
        if isinstance(exif_value, int) or isinstance(exif_value, float):
            vals = [x[key] for x in exifs]
        # Or a string of numbers with ' ' between.
        elif isinstance(exif_value, str):
            vals = [[float(z) for z in x[key].split(' ')] for x in exifs]
        meta[key] = np.squeeze(np.array(vals))
    # Shutter speed is a special case, a string written like 1/N.
    meta['ShutterSpeed'] = np.fromiter(
        (1. / float(exif['ShutterSpeed'].split('/')[1]) for exif in exifs), float)

    # Create raw-to-sRGB color transform matrices. Pipeline is:
    # cam space -> white balanced cam space ("camwb") -> XYZ space -> RGB space.
    # 'AsShotNeutral' is an RGB triplet representing how pure white would measure
    # on the sensor, so dividing by these numbers corrects the white balance.
    whitebalance = meta['AsShotNeutral'].reshape(-1, 3)
    cam2camwb = np.array([np.diag(1. / x) for x in whitebalance])
    # ColorMatrix2 converts from XYZ color space to "reference illuminant" (white
    # balanced) camera space.
    xyz2camwb = meta['ColorMatrix2'].reshape(-1, 3, 3)
    rgb2camwb = xyz2camwb @ _RGB2XYZ
    # We normalize the rows of the full color correction matrix, as is done in
    # https://github.com/AbdoKamel/simple-camera-pipeline.
    rgb2camwb /= rgb2camwb.sum(axis=-1, keepdims=True)
    # Combining color correction with white balance gives the entire transform.
    cam2rgb = np.linalg.inv(rgb2camwb) @ cam2camwb
    meta['cam2rgb'] = cam2rgb

    # get exposure value of first image for gamma mapping at render-time
    image0 = cp.asnumpy(image0)
    image0 = image0 @ meta['cam2rgb'][0].T
    exposure = np.percentile(image0, 97)
    meta['exposure'] = exposure

    progress.close()
    return raw_images, meta, bayer_masks


''' Applies a Bayer CFA mask to an RGB image. This function is adapted from multinerf/internal/raw-utils.py
    This function takes pixel coordinates and determines what color channel the pixel belongs to based on a Bayer pattern.

    params:
        pix_x       ->  Pixel x-coordinates
        pix_y       ->  Pixel y-coordinates
'''
def pixels_to_bayer_mask(pix_x, pix_y):
    # Red is top left (0, 0).
    r = (pix_x % 2 == 0) * (pix_y % 2 == 0)
    # Green is top right (0, 1) and bottom left (1, 0).
    g = (pix_x % 2 == 1) * (pix_y % 2 == 0) + (pix_x % 2 == 0) * (pix_y % 2 == 1)
    # Blue is bottom right (1, 1).
    b = (pix_x % 2 == 1) * (pix_y % 2 == 1)
    bayer_mask =  np.stack([r, g, b], -1).astype(np.float32)
    return bayer_mask   


