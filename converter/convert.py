#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import json
import struct
import argparse

import numpy as np

import torch

from architecture.mobilenetv1 import MobileNetV1

def tensorflow_to_torch_name(tf_name):
   # Convert tensorflow layer names into pytorch layer names.
   tf_name = tf_name.lower()
   tf_split = tf_name.split('/')
   tf_layer_split = tf_split[1].split('_')
   tf_var_type = tf_split[2]

   if tf_var_type in ['weights', 'depthwise_weights']:
      post = '.weight'
   elif tf_var_type in ['biases']:
      post = '.bias'
   else:
      post = ''

   if tf_layer_split[0] == 'conv2d':
      torch_name = 'main_model.conv' + tf_layer_split[1]
      if len(tf_layer_split) > 2:
         torch_name += '.' + tf_layer_split[2]
      else:
         torch_name += '.convolution'
      torch_name += post
   else:
      if tf_layer_split[0] in ['offset', 'displacement', 'heatmap'] and tf_layer_split[-1] == '2':
         torch_name = '_'.join(tf_layer_split[:-1])
         torch_name += post
      else:
         torch_name = ''

   return torch_name

def load_variables(model, save_dir = None):
   # Verify and load model.
   if save_dir:
      try:
         with open(os.path.join(save_dir, 'manifest.json'), 'r') as file:
            variables = json.load(file)
      except FileNotFoundError:
         raise FileNotFoundError(f"The file {os.path.join(save_dir, 'manifest.json')} does not exist.")
      except Exception as e:
         raise e
   else:
      save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'weights', model)
      save_path = os.path.join(save_dir, 'manifest.json')
      try:
         with open(save_path, 'r') as file:
            variables = json.load(file)
      except FileNotFoundError:
         if model not in ['mobilenet_v1_050', 'mobilenet_v1_075', 'mobilenet_v1_100', 'mobilenet_v1_101']:
            raise ValueError(f"The model {model} is not a valid model.")
         else:
            raise FileNotFoundError(f"The file {save_path} was not found.")
      except Exception as e:
         raise e

   # Process variables.
   model_dict = dict()
   for var in variables:
      torch_name = tensorflow_to_torch_name(var)
      if not torch_name: continue
      filename = variables[var]['filename']

      # Open and process file.
      try:
         with open(os.path.join(save_dir, 'info', filename), 'rb') as bytefile:
            stream = bytefile.read()
            var_bytes = str(int(len(stream) / struct.calcsize('f'))) + 'f'
            v = struct.unpack(var_bytes, stream)
            v = np.array(v, dtype = np.float32)

            shape = variables[var]['shape']
            if len(shape) == 4:
               if 'depthwise' in filename:
                  to_transpose = (2, 3, 0, 1)
               else:
                  to_transpose = (3, 2, 0, 1)
               v = np.reshape(v, shape).transpose(to_transpose)
            model_dict[torch_name] = torch.Tensor(v)
      except FileNotFoundError:
         raise FileNotFoundError(f"The file {os.path.join(save_dir, 'info', filename)} was not found.")
      except Exception as e:
         raise e

   return model_dict

def perform_conversion(model_id, output_stride = 16, save_dir = None):
   # Perform model conversion.
   if not save_dir: # Verify and create save directory if necessary.
      save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
      if not os.path.exists(save_dir):
         os.makedirs(save_dir)

   # Load and convert.
   model_state_dict = load_variables(model_id)
   model = MobileNetV1(model_id, output_stride = output_stride)
   model.load_state_dict(model_state_dict)
   checkpoint_path = os.path.join(save_dir, f'{model_id}.pth')
   if os.path.exists(checkpoint_path): # Remove existing model.
      os.remove(checkpoint_path)
   torch.save(model.state_dict(), checkpoint_path)

   print(f"Saved model to {checkpoint_path}.")

if __name__ == '__main__':
   # Get model argument.
   ap = argparse.ArgumentParser()
   ap.add_argument('-m', '--model', default = 'mobilenet_v1_101',
                   help = "Model: The model which you want to convert. (Must be preprocessed).")
   args = vars(ap.parse_args())

   # Process.
   perform_conversion(args['model'])



