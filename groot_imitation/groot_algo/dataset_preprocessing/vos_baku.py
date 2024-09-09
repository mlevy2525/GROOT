"""Get all the masks for the demonstration dataset"""
import h5py
import cv2
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import imageio

from groot_imitation.groot_algo.xmem_tracker import XMemTracker
from groot_imitation.groot_algo.misc_utils import get_annotation_path, get_first_frame_annotation, VideoWriter
from groot_imitation.groot_algo.o3d_modules import convert_convention

import pickle

from groot_imitation.groot_algo.dataset_preprocessing.new_instance import NewInstance

def dataset_vos_annotation(cfg, xmem_tracker, verbose=False):
    """This is the case where we only focus on manipulation one specific-instance."""

    first_frame, first_frame_annotation = get_first_frame_annotation(cfg.annotation_folder)
    if cfg.new_instance_idx != -1:
        new_instance = NewInstance(cfg, first_frame, first_frame_annotation)

    with open(cfg.dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    count = 0
    dataset_masks = []
    if cfg.save_video:
        overlay_images = []
    for demo in tqdm(dataset['observations']):
        xmem_tracker.clear_memory()
        images = demo[cfg.pixel_key]
        # Image should be in RGB format
        image_list = []
        if cfg.new_instance_idx == -1 or count % cfg.new_instance_idx != 0 or count == 0:
            image_list.append(first_frame)

        for image in images:
            if cfg.is_real_robot:
                image_list.append(image[:,:,::-1])
            else:
                image_list.append(image)

        if count % cfg.new_instance_idx == 0 and count != 0 and cfg.new_instance_idx != -1:
            first_frame = cv2.resize(image_list[0], (first_frame_annotation.shape[1], first_frame_annotation.shape[0]), interpolation=cv2.INTER_AREA)
            first_frame_annotation = new_instance.get_new_instance_annotation(first_frame)

        first_frame_annotation = first_frame_annotation
        image_list = [cv2.resize(image, (first_frame_annotation.shape[0], first_frame_annotation.shape[1]), interpolation=cv2.INTER_AREA) for image in image_list]
        masks = xmem_tracker.track_video(image_list, first_frame_annotation)

        if verbose:
            print(len(image_list), len(masks))

        dataset_masks.append(np.stack(masks[1:], axis=0))
        if cfg.save_video:
            for rgb_img, mask in zip(image_list, masks):
                colored_mask = Image.fromarray(mask)
                colored_mask.putpalette(xmem_tracker.palette)
                colored_mask = np.array(colored_mask.convert("RGB"))
                overlay_img = cv2.addWeighted(rgb_img, 0.7, colored_mask, 1.0, 0)

                overlay_images.append(overlay_img)
        count += 1

    if cfg.save_video:
        imageio.mimsave(f"{cfg.annotation_folder}/annotation_video.mp4", overlay_images, fps=40)

    pickle.dump(dataset_masks, open(f"{cfg.annotation_folder}/dataset_masks.pkl", "wb"))