#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import time
import argparse

import cv2
import numpy as np

import torch

from decode import decode_multi_pose
from util.model import load_model
from util.img_util import load_image
from util.draw_util import draw_complete

# Load and parse arguments.
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', default = 'mobilenet_v1_101',
                help = "Model: The model which you want to use for detection.")
ap.add_argument('-i', '--image', default = ['test_imgs/federertennis.jpg', 'test_imgs/person_bench.jpg'],
                help = "Image: Which images you want to detect on.")
args = vars(ap.parse_args())

# Load Model.
model = load_model(args['model'])
output_stride = model.output_stride

# Determine if there are images to detect.
if args['image'] is None:
   print("No images to detect. Exiting.")
   sys.exit(0)

images = []
if isinstance(args['image'], str):
   images.append(args['image'])
elif isinstance(args['image'], (tuple, list)):
   for image in args['image']:
      images.append(image)
else:
   raise ValueError("Unknown format for list of images. Please try again.")

# Detect pose on images.
for image in images:
   # Pick image filepath for future saving.
   file, extension = os.path.splitext(image)
   file = os.path.basename(file)

   input_image, draw_image, output_scale = load_image(image)

   with torch.no_grad():
      input_image = torch.Tensor(input_image)

      # Input image into network.
      heatmaps, offsets, displacement_fwd, displacment_bwd = model(input_image)

      # Decode pose scores/coordinates from network outputs.
      pose_scores, keypoint_scores, keypoint_coords = decode_multi_pose(
         heatmaps.squeeze(0),
         offsets.squeeze(0),
         displacement_fwd.squeeze(0),
         displacment_bwd.squeeze(0),
         output_stride,
         min_pose_score = 0.25
      )

   # Adjust coordinates by image resolution scale.
   keypoint_coords *= output_scale

   # Draw pose keypoints and skeleton from scores.
   draw_image = draw_complete(
      draw_image, pose_scores, keypoint_scores, keypoint_coords,
      min_pose_score = 0.25, min_part_score = 0.25
   )

   # Save image detections.
   savedir = os.path.join(os.path.dirname(__file__), 'modded')
   cv2.imwrite(os.path.join(savedir or '', f'{file}-detect{extension}'), draw_image)







