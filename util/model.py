#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import torch

from architecture.mobilenetv1 import MobileNetV1

def load_model(model_id, output_stride = 16, model_dir = None):
   # Load model architecture with weights.
   if model_dir:
      try: # Verify model and path.
         model_path = os.path.join(model_dir, f'{model_id}.pth')
         load_dict = torch.load(model_path)
      except FileNotFoundError:
         raise FileNotFoundError(f"The file {os.path.join(model_dir, f'{model_id}.pth')} does not exist.")
      except Exception as e:
         raise e
   else:
      model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'{model_id}.pth')
      try:
         load_dict = torch.load(model_path)
      except FileNotFoundError:
         raise FileNotFoundError(f"The saved model for {model_id} does not exist. Please convert the model and try again.")
      except Exception as e:
         raise e

   # Load and return model.
   model = MobileNetV1(model_id, output_stride = output_stride)
   model.load_state_dict(load_dict)

   return model



