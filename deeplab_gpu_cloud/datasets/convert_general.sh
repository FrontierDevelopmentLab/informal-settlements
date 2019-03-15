#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to preprocess general datasets which don't require further preprocessing.
# The images have to be in JPEG and the segmentation masks in PNG format with continuous
# annotation values from 0 up to the number of available classes.
#
# Usage:
#   sh ./convert_general.sh
#

if [[ "$#" -ne 1 ]] ; then
    echo 'Usage: sh ./convert_general.sh <DATASET>'
    exit 1
fi

# Set the dataset
DATASET=$1

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/${DATASET}"
mkdir -p "${WORK_DIR}"


# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

echo "Current_DIR $(pwd)"
echo "WORK_DIR ${WORK_DIR}"
echo "OUTPUT_DIR ${OUTPUT_DIR}"

IMAGE_FOLDER="${WORK_DIR}/images"
SEMANTIC_SEG_FOLDER="${WORK_DIR}/labels"
LIST_FOLDER="${WORK_DIR}"

echo "Converting ${DATASET} dataset..."
python ./build_general_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
