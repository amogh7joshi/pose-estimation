#!/usr/bin/env python3
import abc

class Keypoints(object):
   '''
   Constant class containing lists of different keypoint correlations.
   '''
   def __init__(self):
      self.keypoints = [
         'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
         'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
         'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle']
      self.connected_keypoints = [
         ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
         ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
         ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
         ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
         ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
         ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")]
      self.keypoint_chain = [
         ("nose", "leftEye"), ("leftEye", "leftEar"), ("nose", "rightEye"),
         ("rightEye", "rightEar"), ("nose", "leftShoulder"),
         ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
         ("leftShoulder", "leftHip"), ("leftHip", "leftKnee"),
         ("leftKnee", "leftAnkle"), ("nose", "rightShoulder"),
         ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
         ("rightShoulder", "rightHip"), ("rightHip", "rightKnee"),
         ("rightKnee", "rightAnkle")]
      self.keypoint_channels = [
         'left_face', 'right_face', 'right_upper_leg_front',
         'right_lower_leg_back', 'right_upper_leg_back',
         'left_lower_leg_front', 'left_upper_leg_front',
         'left_upper_leg_back', 'left_lower_leg_back',
         'right_feet', 'right_lower_leg_front', 'left_feet',
         'torso_front', 'torso_back', 'right_upper_arm_front',
         'right_upper_arm_back', 'right_lower_arm_back',
         'left_lower_arm_front', 'left_upper_arm_front',
         'left_upper_arm_back', 'left_lower_arm_back',
         'right_hand', 'right_lower_arm_front', 'left_hand']
      self.keypoint_ids = {name: id for id, name in enumerate(self._keypoints)}
      self.connected_keypoint_ids = [
         (self._keypoint_ids[a], self._keypoint_ids[b]) for a, b, in self._connected_keypoints]
      self.parent_child_parts = [
         (self._keypoint_ids[a], self._keypoint_ids[b]) for a, b in self._keypoint_chain]

   def __len__(self):
      return len(self._keypoints)

   def __getitem__(self, item):
      if not isinstance(item, int):
         raise ValueError("That is not a valid index.")
      elif item >= len(self) or item < 0:
         raise ValueError(f"Your index is not within the range 0-{len(self)}.")
      return self._keypoints[item]

   def __setitem__(self, key, value):
      raise ValueError("You cannot change the value of an index from outside the class.")

   @property
   def keypoints(self):
      return self._keypoints

   @property
   def connected_keypoints(self):
      return self._connected_keypoints

   @property
   def keypoint_chain(self):
      return self._keypoint_chain

   @property
   def keypoint_channels(self):
      return self._keypoint_channels

   @property
   def keypoint_ids(self):
      return self._keypoint_ids

   @property
   def connected_keypoint_ids(self):
      return self._connected_keypoint_ids

   @property
   def parent_child_parts(self):
      return self._parent_child_parts

   @keypoints.setter
   def keypoints(self, value):
      self._keypoints = value

   @connected_keypoints.setter
   def connected_keypoints(self, value):
      self._connected_keypoints = value

   @keypoint_chain.setter
   def keypoint_chain(self, value):
      self._keypoint_chain = value

   @keypoint_channels.setter
   def keypoint_channels(self, value):
      self._keypoint_channels = value

   @keypoint_ids.setter
   def keypoint_ids(self, value):
      self._keypoint_ids = value

   @connected_keypoint_ids.setter
   def connected_keypoint_ids(self, value):
      self._connected_keypoint_ids = value

   @parent_child_parts.setter
   def parent_child_parts(self, value):
      self._parent_child_parts = value