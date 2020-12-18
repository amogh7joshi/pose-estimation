#!/usr/bin/env python3
import os
import time
import json

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.modules import Convolution, SeparableConv

def model_config(config_file = None) -> list:
   # Verify and load architectures from file.
   if config_file:
      try:
         with open(config_file, 'r') as file:
            architectures = json.load(file)
      except FileNotFoundError:
         raise FileNotFoundError(f"The file {config_file} does not exist.")
      except Exception as e:
         raise e
   else:
      config_file = os.path.join(os.path.dirname(__file__), 'architectures.json')
      try:
         with open(config_file, 'r') as file:
            architectures = json.load(file)
      except FileNotFoundError:
         raise FileNotFoundError(f"The file {config_file} is missing.")
      except Exception as e:
         raise e

   # Determine architectures.
   existing_models = architectures['MODELS']
   try:
      mobilenet_v1_050_arch = architectures['mobilenet-v1-050-arch']
      mobilenet_v1_075_arch = architectures['mobilenet-v1-075-arch']
      mobilenet_v1_100_arch = architectures['mobilenet-v1-100-arch']
   except KeyError as ke:
      raise ke
   except TypeError as te:
      raise te
   except Exception as e:
      raise e

   # Return existing models and architectures.
   return [existing_models, mobilenet_v1_050_arch, mobilenet_v1_075_arch, mobilenet_v1_100_arch]

class MobileNetV1(nn.Module):
   def __init__(self, model, output_stride = 16):
      # Verify model and acquire architecture.
      super(MobileNetV1, self).__init__()
      self.assert_model_exists(model)
      self.architecture = self.acquire_model(model)
      self.output_stride = output_stride

      # Setup Model Architecture.
      conv_strided = self.convert_strided_layers(self.architecture, output_stride)
      conv_list = [(
         'conv%d' % c['blockID'], c['type'](
         c['input'], c['output'], 3, c['stride'], c['rate']
         )) for c in conv_strided]
      final_layer = conv_strided[-1]['output']

      # Complete Main Model Architecture.
      self.main_model = nn.Sequential(OrderedDict(conv_list))
      self.heatmap = nn.Conv2d(final_layer, 17, 1, 1)
      self.offset = nn.Conv2d(final_layer, 34, 1, 1)
      self.displacement_fwd = nn.Conv2d(final_layer, 32, 1, 1)
      self.displacement_bwd = nn.Conv2d(final_layer, 32, 1, 1)

   @staticmethod
   def assert_model_exists(model):
      # Verify that the model is valid.
      verified_models, _, _, _ = model_config()
      if model not in verified_models:
         raise ValueError(f"The model {model} is not a valid model: {verified_models}.")

   def acquire_model(self, model):
      # Acquire model architecture.
      global load_arch
      _, arch_50, arch_75, arch_100 = model_config()
      try:
         if model == 'mobilenet_v1_050':
            load_arch = arch_50
         elif model == 'mobilenet_v1_075':
            load_arch = arch_75
         elif model == 'mobilenet_v1_100' or model == 'mobilenet_v1_101':
            load_arch = arch_100
         else:
            self.assert_model_exists(model)
      except Exception as e:
         raise e

      # Parse Layers.
      try:
         for layer in load_arch:
            layer[0] = self.parse_layer(layer[0])
      except ValueError as ve:
         raise ve
      except Exception as e:
         raise e

      return load_arch

   @staticmethod
   def parse_layer(layer):
      # Verify that layer is accurate.
      if layer == 'Convolution':
         return Convolution
      elif layer == 'SeparableConv':
         return SeparableConv
      else:
         raise ValueError(f"The layer {layer} is not a valid layer. Please check the architectures.json file.")

   @staticmethod
   def convert_strided_layers(architecture, output_strides):
      stride = 1
      rate = 1
      layer_info = []

      # Create information dictionary about strides and rate for future model construction.
      for indx, layer in enumerate(architecture):
         type = layer[0]
         input = layer[1]
         output = layer[2]
         strides = layer[3]

         if stride == output_strides:
            layer_stride = 1
            layer_rate = rate
            rate *= strides
         else:
            layer_stride = strides
            layer_rate = 1
            stride *= strides

         layer_info.append({
            'blockID': indx,
            'type': type,
            'input': input,
            'output': output,
            'stride': layer_stride,
            'rate': layer_rate,
            'outputStride': stride
         })

      return layer_info

   def forward(self, x):
      x = self.main_model(x)
      heatmap = torch.sigmoid(self.heatmap(x))
      offset = self.offset(x)
      displacement_fwd = self.displacement_fwd(x)
      displacement_bwd = self.displacement_bwd(x)

      return heatmap, offset, displacement_fwd, displacement_bwd



