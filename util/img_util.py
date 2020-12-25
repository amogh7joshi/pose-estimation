import os
import sys

import cv2
import numpy as np

def load_image(image_path, resize = (None, None), **kwargs):
   _resize = None
   # Parse Kwargs.
   if len(kwargs) > 2:
      raise ValueError("You have provided too many keyword arguments.")
   for kwarg in kwargs:
      if kwarg not in ['scale_factor', 'output_stride']:
         raise ValueError("You have provided an invalid keyword argument. Valid keyword arguments are: scale_factor, output_stride")
   if resize:
      if not isinstance(resize, (list, tuple, set, int)):
         raise TypeError("You have provided an invalid value for the resize argument.")
      if isinstance(resize, int):
         _resize = (resize, resize)
      else:
         if len(resize) > 2:
            raise ValueError("You can only resize an image by two dimensions.")
         else:
            _resize = resize

   # Verify And Load Image (a lot of verifications).
   try:
      image = cv2.imread(image_path)
      if _resize:
         image_path = cv2.resize(image_path, dsize=_resize, interpolation=cv2.INTER_LINEAR)
      if image is None:
         raise SystemError
      try:
         return process_input(image, **kwargs)
      except TypeError:
         raise TypeError("You are trying to convert two incompatible types.")
      except Exception as e:
         raise e
   except SystemError:
      try:
         if _resize:
            image_path = cv2.resize(image_path, dsize = _resize, interpolation = cv2.INTER_LINEAR)
         return process_input(image_path, **kwargs)
      except Exception as e:
         raise e
   except cv2.error:
      raise Exception(f"The image at {image_path} does not exist or has issues.")
   except Exception as e:
      raise e

# Helper function for processing after image loading.
def process_input(image, scale_factor = 1.0, output_stride = 16):
   # Determine a valid resolution.
   try:
      height = (int(image.shape[0] * scale_factor) // output_stride) * output_stride + 1
      width = (int(image.shape[1] * scale_factor) // output_stride) * output_stride + 1
      scale = np.array([image.shape[0] / height, image.shape[1] / width])
   except ZeroDivisionError:
      raise ZeroDivisionError(f"You have provided an invalid value for scale_factor: {scale_factor}")
   except Exception as e:
      raise e

   # Process.
   final_image = cv2.resize(image, (width, height), interpolation = cv2.INTER_LINEAR)
   final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
   final_image = final_image.astype(float)
   final_image = final_image * (2.0 / 255.0) - 1.0
   final_image = final_image.transpose((2, 0, 1)).reshape(1, 3, height, width)
   return final_image, image, scale

