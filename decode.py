#!/usr/bin/env python3
import os
import warnings

import cv2
import numpy as np
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.keypoints import Keypoints

# Initialize Constant Class
kp = Keypoints()

# The next two methods are for single-pose detection.

def to_keypoint(edge_id, source_keypoint, target_keypoint_id, scores, offsets, output_stride, displacements):
   # Convert to score & keypoint.
   try:
      height = scores.shape[1]; width = scores.shape[2]
   except AttributeError:
      raise AttributeError("You have not provided a valid argument for 'scores', as it does not have an attribute 'shape'.")
   except Exception as e:
      raise e

   source_keypoint_indices = np.clip(
      np.round(source_keypoint / output_stride), a_min = 0, a_max = [height - 1, width - 1]).astype(np.int32)

   displaced_point = source_keypoint + displacements[edge_id, source_keypoint_indices[0], source_keypoint_indices[1]]

   displaced_point_indices = np.clip(
      np.round(displaced_point / output_stride), a_min = 0, a_max = [height - 1, width - 1]).astype(np.int32)

   score = scores[target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]]

   image_coord = displaced_point_indices * output_stride + offsets[
      target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]]

   return score, image_coord

def decode_single_pose(root_score, root_id, image_coord, scores, offsets, output_stride, displacements_fwd, displacements_bwd):
   # Decode MobileNet output into single pose.
   num_parts = scores.shape[0]
   num_edges = len(kp.parent_child_parts)

   instance_keypoint_scores = np.zeros(num_parts)
   instance_keypoint_coords = np.zeros((num_parts, 2))
   instance_keypoint_scores[root_id] = root_score
   instance_keypoint_coords[root_id] = image_coord

   try:
      for edge in reversed(range(num_edges)):
         target_keypoint_id, source_keypoint_id = kp.parent_child_parts[edge]
         if instance_keypoint_scores[source_keypoint_id] > 0.0 and instance_keypoint_scores[target_keypoint_id] == 0.0:
            try:
               score, coords = to_keypoint(edge, instance_keypoint_coords[source_keypoint_id], target_keypoint_id,
                                           scores, offsets, output_stride, displacements_bwd)
               instance_keypoint_scores[target_keypoint_id] = score
               instance_keypoint_coords[target_keypoint_id] = coords
            except Exception as e:
               raise e
   except ValueError as ve:
      raise ve
   except Exception as e:
      raise e

   try:
      for edge in range(num_edges):
         source_keypoint_id, target_keypoint_id = kp.parent_child_parts[edge]
         try:
            if instance_keypoint_scores[source_keypoint_id] > 0.0 and instance_keypoint_scores[target_keypoint_id] == 0.0:
               score, coords = to_keypoint(edge, instance_keypoint_coords[source_keypoint_id], target_keypoint_id,
                                           scores, offsets, output_stride, displacements_fwd)
               instance_keypoint_scores[target_keypoint_id] = score
               instance_keypoint_coords[target_keypoint_id] = coords
         except Exception as e:
            raise e
   except ValueError as ve:
      raise ve
   except Exception as e:
      raise e

   return instance_keypoint_scores, instance_keypoint_coords

# The following methods are for multi-pose detection.

def within_nms_radius(pose_coords, squared_nms_radius, point):
    if not pose_coords.shape[0]:
        return False
    return np.any(np.sum((pose_coords - point) ** 2, axis=1) <= squared_nms_radius)

def get_instance_score(pose_coords, nms_radius, keypoint_scores, keypoint_coords):
   # Get instance score of each pose.
   try:
      if pose_coords.shape[0]:
         # Non-max-suppression calculations.
         score = np.sum((pose_coords - keypoint_coords) ** 2, axis = 2) > nms_radius ** 2
         non_overlapped = np.sum(keypoint_scores[np.all(score, axis = 0)])
      else:
         non_overlapped = np.sum(keypoint_scores)
   except Exception as e:
      raise e
   else:
      return non_overlapped / len(keypoint_scores)

def build_window_with_score(score_thresh, local_max_radius, scores):
   # Determine parts.
   parts = []
   try:
      lmd = 2 * local_max_radius + 1
      try:
         max_vals = F.max_pool2d(scores, lmd, stride = 1, padding = 1)
         max_loc = (scores == max_vals) & (scores > score_thresh)
         max_loc_indx = max_loc.nonzero()
         scores_vector = scores[max_loc]
         sort_indx = torch.argsort(scores_vector, descending = True)
      except Exception as e:
         raise e
   except ValueError as ve:
      raise ve
   except Exception as e:
      raise e

   return scores_vector[sort_indx], max_loc_indx[sort_indx]

def decode_multi_pose(scores, offsets, d_fwd, d_bwd, output_stride, max_pose_detections = 10,
                      score_thresh = 0.5, nms_radius = 20, min_pose_score = 0.5):
   # Decode MobileNet output into multiple poses.
   try:
      part_scores, part_indx = build_window_with_score(score_thresh, 1, scores)
      part_scores = part_scores.cpu().numpy()
      part_indx = part_indx.cpu().numpy()
      scores = scores.cpu().numpy()
   except Exception as e:
      raise e

   try:
      # Reshape and transpose all of the different inputs to necessary specifications.
      height, width = scores.shape[1], scores.shape[2]
      offsets = offsets.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
      displacements_fwd = d_fwd.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
      displacements_bwd = d_bwd.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
   except ValueError as ve:
      raise ve
   except Exception as e:
      raise e

   count = 0
   pose_scores = np.zeros(max_pose_detections)
   keypoint_scores = np.zeros((max_pose_detections, len(kp)))
   keypoint_coords = np.zeros((max_pose_detections, len(kp), 2))

   for root_score, (root_id, root_coord_y, root_coord_x) in zip(part_scores, part_indx):
      root_coord = np.array([root_coord_y, root_coord_x])
      image_coords = root_coord * output_stride + offsets[root_id, root_coord_y, root_coord_x]

      try:
         # Perform non-max-suppression calculations.
         if within_nms_radius(keypoint_coords[:count, root_id, ...], nms_radius ** 2, image_coords):
            continue
         try:
            inst_keypoint_scores, inst_keypoint_coords = decode_single_pose(
               root_score, root_id, image_coords, scores, offsets, output_stride, displacements_fwd, displacements_bwd)
            pose_score = get_instance_score(
               keypoint_coords[:count, ...], nms_radius ** 2, inst_keypoint_scores, inst_keypoint_coords)
         except Exception as e:
            raise e
         else:
            if min_pose_score == 0. or pose_score >= min_pose_score:
               pose_scores[count] = pose_score
               keypoint_scores[count, ...] = inst_keypoint_scores
               keypoint_coords[count, ...] = inst_keypoint_coords
               count += 1
            if count >= max_pose_detections:
               break
      except Exception as e:
         raise e

   return pose_scores, keypoint_scores, keypoint_coords


