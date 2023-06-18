"""
TEMP test and exploration script
"""
import os
import inspect
from typing import List, Optional, Union
from collections import defaultdict

import cv2
import numpy as np
import torch

import PIL
from PIL import Image
import gradio as gr
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from diffusers import StableDiffusionInpaintPipeline

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm


FILE_PATH = os.path.realpath(__file__)
ASSET_PATH = os.path.join(os.path.split(os.path.dirname(FILE_PATH))[0], 'assets')
IMAGE_PATH = os.path.join(ASSET_PATH, 'horses.png')
DEVICE = 'cuda'


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def closest_number(n, m=8, q=0):
    """ Obtains closest number to n that is divisble by m 
        q allows for a larger number to be returned
        """
    return int(n/m) * (m+q)


def draw_panoptic_segmentation(model, segmentation, segments_info):
    # get the used color map
    # viridis = cm.get_cmap('viridis', torch.max(segmentation))
    viridis = mpl.colormaps.get_cmap('viridis').resampled(torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))

    # remove legend for better viz
    # ax.legend(handles=handles)
    plt.show()


if __name__ == '__main__':
    print(os.path.exists(IMAGE_PATH))

    # read image and resize to an 8x8 divisible number
    image = Image.open(IMAGE_PATH)
    W, H = image.size
    print(W, H)
    W, H = closest_number(W/2), closest_number(H/2)
    image = image.resize((W, H))
    print(image.size)

    # get segmentation model
    print("getting segmentation models")
    seg_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")

    print("performing segmentation")
    # prepare image for input into segmentation model
    inputs = seg_processor(image, return_tensors="pt")

    with torch.no_grad():
        seg_outputs = seg_model(**inputs)

    # prost process segmentation results
    seg_prediction = seg_processor.post_process_panoptic_segmentation(seg_outputs, target_sizes=[image.size[::-1]])[0]


    # display segementation prediction
    draw_panoptic_segmentation(seg_model, **seg_prediction)

    # get segmentation labels
    segment_labels = {}

    for segment in seg_prediction['segments_info']:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = seg_model.config.id2label[segment_label_id]

        segment_labels.update({segment_id : segment_label})

    # print(segment_labels)

    # obtain target selections
    tgts = []
    for k,v in segment_labels.items():
        if v in ['horse']:
            tgts.append(k)

    # print(tgts)

    print("Obtaining inpainting mask")
    # mask = np.array([(seg_prediction['segmentation'] == t).numpy() for t in tgts]).sum(axis=0).astype(np.uint8)*255

    # or use an inverse mask
    mask = np.logical_not(np.array([(
        seg_prediction['segmentation'] == t).numpy() for t in tgts]).sum(axis=0)).astype(np.uint8)*255

    plt.imshow(mask)
    plt.show()

    # get Stable Diffusion model for image inpainting
    model_path = "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(DEVICE)

    # edit the image
    print("Editing the image!")
    

    prompt = "astronauts in space"

    guidance_scale=17.5
    num_samples = 3
    generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

    images = pipe(
        prompt=prompt,
        image=image,
        mask_image=Image.fromarray(mask), # ensure mask is same type as image
        height=H,
        width=W,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images

    

    # insert initial image in the list so we can compare side by side
    images.insert(0, image)

    
    # image_grid(images, 1, num_samples + 1)

    _, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(images[0])
    ax[0, 1].imshow(images[1])
    ax[1, 0].imshow(images[2])
    ax[1, 1].imshow(images[3])
    plt.show()

     

        









