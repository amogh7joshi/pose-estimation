#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import warnings

import cv2
import numpy as np

from util.keypoints import Keypoints

# Initialize Constant Class.
kp = Keypoints()

def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence = 0.1):
   # Acquire adjacent keypoints.
   new_points = []
   for left, right in kp.connected_keypoint_ids:
      if keypoint_scores[left] < min_confidence:
         continue
      if keypoint_scores[right] < min_confidence:
         continue
      new_points.append(np.array([
         keypoint_coords[left][::-1], keypoint_coords[right][::-1]
      ]).astype(np.int32))
   return new_points

def draw_keypoints(image, scores, keypoint_scores, keypoint_coords, min_pose_conf = 0.5, min_part_conf = 0.5):
   # Verify Confidences.
   if min_pose_conf <= 0.15:
      warnings.warn(f"You have chosen a very low value for min_pose_conf. Consider setting it >= 0.5.")
   if min_part_conf <= 0.15:
      warnings.warn(f"You have chosen a very low value for min_part_conf. Consider setting it >= 0.5")

   # Select Keypoints.
   keypoints = []
   for indx, score in enumerate(scores):
      if score < min_pose_conf:
         continue
      for k_s, k_c in zip(keypoint_scores[indx, :], keypoint_coords[indx, :, :]):
         if k_s < min_part_conf:
            continue
         keypoints.append(cv2.KeyPoint(k_c[1], k_c[0], 10. * k_s))

   # Draw Keypoints
   image = cv2.drawKeypoints(image, keypoints, outImage = np.array([]))
   return image

def draw_skeleton(image, scores, keypoint_scores, keypoint_coords, min_pose_conf = 0.5, min_part_conf = 0.5):
   # Verify Confidences.
   if min_pose_conf <= 0.15:
      warnings.warn(f"You have chosen a very low value for min_pose_conf. Consider setting it >= 0.5.")
   if min_part_conf <= 0.15:
      warnings.warn(f"You have chosen a very low value for min_part_conf. Consider setting it >= 0.5")

   # Get adjacent keypoints.
   adjacent_points = []
   for indx, score in enumerate(scores):
      if score < min_pose_conf:
         continue
      new_keypoints = get_adjacent_keypoints(keypoint_scores[indx:, :], keypoint_coords[indx, :, :], min_part_conf)
      adjacent_points.extend(new_keypoints)

   # Draw Skeleton
   image = cv2.polylines(image, adjacent_points, isClosed = False, color = (0, 255, 255))

   return image

def draw_complete(image, scores, keypoint_scores, keypoint_coords, min_pose_score = 0.5, min_part_score = 0.5):
   # Convenience method to draw both keypoints and skeleton.
   keypoints = []; adjacent_keypoints = []
   for indx, score in enumerate(scores):
      if score < min_pose_score:
         continue
      new_points = get_adjacent_keypoints(keypoint_scores[indx, ...], keypoint_coords[indx, ...], min_part_score)
      adjacent_keypoints.extend(new_points)

      for k_s, k_c in zip(keypoint_scores[indx, ...], keypoint_coords[indx, ...]):
         if k_s < min_part_score:
            continue
         keypoints.append(cv2.KeyPoint(k_c[1], k_c[0], k_s * 10.))

   if keypoints:
      # Draw Keypoints.
      image = cv2.drawKeypoints(
         image, keypoints, outImage = np.array([]), color = (0, 255, 255),
         flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
      )

   # Draw Skeleton.
   image = cv2.polylines(image, adjacent_keypoints, isClosed = False, color = (0, 255, 255))

   return image


