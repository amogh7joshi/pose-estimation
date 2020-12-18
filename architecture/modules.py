#!/usr/bin/env python3
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

# Modules & functions used in the main MobileNet v1.

class Convolution(nn.Module):
   def __init__(self, input, output, kernel_size = 3, strides = 1, dilation = 1):
      # General convolution layer.
      super(Convolution, self).__init__()
      self.convolution = nn.Conv2d(
         input, output, kernel_size, strides,
         padding = self._pad(kernel_size, strides, dilation), dilation = dilation
      )

   @staticmethod
   def _pad(kernel_size, stride, dilation):
      return ((stride - 1) + dilation * (kernel_size - 1)) // 2

   def forward(self, x):
      return F.relu6(self.convolution(x))

class SeparableConv(nn.Module):
   def __init__(self, input, output, kernel_size = 3, strides = 1, dilation = 1):
      super(SeparableConv, self).__init__()
      self.depthwise = nn.Conv2d(
         input, input, kernel_size, strides,
         self._pad(kernel_size, strides, dilation), dilation = dilation, groups = input
      )
      self.pointwise = nn.Conv2d(
         input, output, 1, 1
      )

   @staticmethod
   def _pad(kernel_size, stride, dilation):
      return ((stride - 1) + dilation * (kernel_size - 1)) // 2

   def forward(self, x):
      x = F.relu6(self.depthwise(x))
      x = F.relu6(self.pointwise(x))
      return x


def convert_strided_layers(architecture, output_strides):
   stride = 1; rate = 1
   layer_info = []

   # Create information dictionary about strides and rate for future model construction.
   for indx, layer in enumerate(architecture):
      type = layer[0]
      strides = layer[1]

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
         'stride': layer_stride,
         'rate': layer_rate,
         'outputStride': stride
      })

   return layer_info

