# gaussian-splatting FORK script for raw-splats
#   Description:    render images from raw-splats model and process them through minimal pipeline, designed by multinerf
#   Author:         Anthony Bird


import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import numpy as np


def linear_to_srgb(linear, eps=None):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        # set to machine epsilon 
        #   - smallest positive number that can be represented by floating point on machine
        eps = np.finfo(np.float32).eps
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * np.maximum(eps, linear)**(5 / 12) - 11) / 200

    srgb_final = np.where(linear <= 0.0031308, srgb0, srgb1)

    return srgb_final


# processes the given image
#   params:
#       img         : float32       ->  image to process, float32 [0-1]
#       gt_image    : float32       ->  ground truth image [0-1]
#       blackLevel  : numpy.ndarray ->  black level of gt_image
#       whiteLevel  : numpy.ndarray ->  white level of gt_image
def post_process(img, blackLevel, whiteLevel, exposure, cam2rgb):
    if img.shape[-1] != 3:
        raise ValueError(f'raw.shape[-1] is {img.shape[-1]}, expected 3')
    if cam2rgb.shape != (3, 3):
        raise ValueError(f'camtorgb.shape is {cam2rgb.shape}, expected (3, 3)')
    # Convert from camera color space to standard linear RGB color space.
    rgb_linear = np.matmul(img, cam2rgb.T)
    if exposure is None:
        exposure = np.percentile(rgb_linear, 97)

    # "Expose" image by mapping the input exposure level to white and clipping.
    rgb_linear_scaled = np.clip(rgb_linear / exposure, 0, 1)

    # Apply sRGB gamma curve to serve as a simple tonemap.
    srgb = linear_to_srgb(rgb_linear_scaled)
    srgb = (srgb * 65535).astype(np.uint16)

    return srgb





def render_set(model_path, name, iteration, views, gaussians, pipeline, background, metadata = None, img_idx = 0):
    renders = []
    gt_images = [] 

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        renders.append(render(view, gaussians, pipeline, background)["render"])
        gt_images.append(view.original_image)
    
    rendered_img =  renders[img_idx].detach().cpu().permute(1, 2, 0)
    rendered_img = rendered_img.numpy()
    gt_img = gt_images[img_idx].detach().cpu().permute(1, 2, 0)
    gt_img = gt_img.numpy()

    # set arbitrary black and white levels for testing
    if metadata is not None:
        blackLevel = metadata['BlackLevel'].reshape(-1, 1, 1)
        whiteLevel = metadata['WhiteLevel'].reshape(-1, 1, 1)
        render_post = post_process(rendered_img, blackLevel[img_idx], whiteLevel[img_idx], metadata['exposure'], metadata['cam2rgb'][img_idx])
        gt_post = post_process(gt_img, blackLevel[img_idx], whiteLevel[img_idx], metadata['exposure'], metadata['cam2rgb'][img_idx])

        # display processed render and gt
        render_post = cv2.cvtColor(render_post, cv2.COLOR_BGR2RGB)
        gt_post = cv2.cvtColor(gt_post, cv2.COLOR_BGR2RGB)
        cv2.imshow('rendering', render_post)
        cv2.imshow('gt', gt_post)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
               
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, img_idx : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene.metadata, img_idx)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scene.metadata, img_idx)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--img", default=0, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.img)