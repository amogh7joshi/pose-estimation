#!/usr/bin/env bash

# Read model which user wishes to convert.
echo "What model do you want to process?"
read -r model

_contains () {  # Check if space-separated list $1 contains line $2
  echo "$1" | tr ' ' '\n' | grep -F -x -q "$2"
}

# List containing valid model types.
declare list=(
  'mobilenet_v1_050'
  'mobilenet_v1_075'
  'mobilenet_v1_100'
  'mobilenet_v1_101'
)

# Determine whether user provided valid model.
if ! _contains "${list}" "${model}"; then
  echo "The model \"${model}\" is not a valid model. Aborting."
  exit 1
fi

# Determine whether model weights directory is empty.
DIR=./data/weights/$model/info
if [ "$(ls -A $DIR)" ]; then

  echo "Existing weights files found in directory ${DIR}. Overwrite them? [y|n]"
  read -r override
  if [ $override = "y" ]; then
    mode="clear"
  elif [ $override = "n" ]; then
    mode="renew"
  else
    echo "Invalid option \"${override}\". Aborting."
    exit 1
  fi
fi

# Download weight files.
python3 converter/download.py -m $mode -o $model

echo "Model ${model} weight files downloaded."
sleep 2

# Convert weight files to PyTorch model.
python3 converter/convert.py -m $model

echo "Model ${model} conversion complete."
exit 0


