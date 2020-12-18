#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import time
import json
import urllib.request
import argparse

STORAGE_DIR = 'https://storage.googleapis.com/tfjs-models/weights/posenet/'

def download_file(base_dir = None, model = None, mode = 'clear') -> None:
   if model is None: raise ValueError("You have not provided a model to download.")
   if base_dir is None: base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'weights')
   if not os.path.isdir(os.path.join(base_dir, model)):
      os.mkdir(os.path.join(base_dir, model))
   if not os.path.isdir(os.path.join(base_dir, model, 'info')):
      os.mkdir(os.path.join(base_dir, model, 'info'))
   savedir = os.path.join(base_dir, model, 'info')

   # Check whether provided model is valid.
   if model not in ['mobilenet_v1_050', 'mobilenet_v1_075', 'mobilenet_v1_100', 'mobilenet_v1_101']:
      raise ValueError("That is not a valid model type.")

   # Download JSON file containing model information.
   manifest_source = os.path.join(STORAGE_DIR, model, 'manifest.json')
   response = urllib.request.urlopen(manifest_source)
   data = response.read()

   if mode == 'renew':
      if os.path.isfile(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/weights', model, 'manifest.json')):
         print(
            f"Skipping \'manifest.json\', as {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/weights', model, 'manifest.json')} already exists.")
      else:
         with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/weights', model, 'manifest.json'),
                   'bw+') as manifest_save:
            manifest_save.write(data)
         print(
            f"Saved 'manifest.json' to {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/weights', model, 'manifest.json')}")
   elif mode == 'clear':
      with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/weights', model, 'manifest.json'),
                'bw+') as manifest_save:
         manifest_save.write(data)
      print(
         f"Saved 'manifest.json' to {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/weights', model, 'manifest.json')}")
   else:
      raise ValueError("That is not a valid mode.")

   time.sleep(2) # Allow time for manifest.json to manifest (no pun intended =D).

   # Download model.
   model_source = os.path.join(STORAGE_DIR, model)

   with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/weights', model, 'manifest.json'),
             encoding = 'utf-8') as file:
      json_data = json.load(file)

      for key in json_data:
         section = json_data[key]
         filename = section['filename']
         response = urllib.request.urlopen(os.path.join(model_source, filename))
         data = response.read()

         if mode == 'renew':
            if os.path.isfile(os.path.join(savedir, str(filename).lower())):
               print(f'Skipping {filename.lower()}, as {os.path.join(savedir, str(filename).lower())}, already exists.')
               continue
         elif mode == 'clear':
            pass
         else:
            raise ValueError("That is not a valid mode.")

         with open(os.path.join(savedir, str(filename).lower()), 'bw+') as save_file:
            save_file.write(data)

         print(f"Saved {filename} to {os.path.join(savedir, str(filename).lower())}")


if __name__ == '__main__':
   # Mode to run by, which model to download.
   ap = argparse.ArgumentParser()
   ap.add_argument('-m', '--mode', default='renew',
                   help="Mode: [renew] to only download nonexistent files, [clear] to clear and re-download all files.")
   ap.add_argument('-o', '--model', default='mobilenet_v1_101',
                   help="Model: Which model you want to download. ")
   args = vars(ap.parse_args())

   # Process.
   download_file(model=args['model'], mode=args['mode'])

   