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
# This script is used to run local test on a dataset. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab_gpu_cloud directory.
#   sh ./local_test.sh <DATASET> <NUM_TRAIN_ITERATIONS> <INITIAL_LEARNING_RATE>
#
#

# Simple way to slip in basic parameters in a docker container (for example in a cloud environment)
if [[ "$#" -ne 5 ]] ; then
    echo 'Usage: sh ./local_test.sh <DATASET> <NUM_CLASSES> <NUM_TRAIN_ITERATIONS> <INITIAL_LEARNING_RATE> <NUM_GPU>'
    exit 1
fi

DATASET=$1
NUM_CLASSES=$2
NUM_TRAIN_ITERATIONS=$3
INITIAL_LEARNING_RATE=$4
NUM_GPU=$5

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab_gpu_cloud"
DATASET_DIR="datasets"
EXP_DIR="/results"

DATASET_TF_RECORD="${WORK_DIR}/${DATASET_DIR}/${DATASET}/tfrecord"

# Run model_test first to make sure the PYTHONPATH is correctly set.
python "${WORK_DIR}"/model_test.py -v

# Go to datasets folder and convert the dataset.

cd "${WORK_DIR}/${DATASET_DIR}"
sh convert_general.sh "${DATASET}"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${EXP_DIR}/${DATASET_DIR}/${DATASET}/init_models"
TRAIN_LOGDIR="${EXP_DIR}/${DATASET_DIR}/${DATASET}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${EXP_DIR}/${DATASET_DIR}/${DATASET}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${EXP_DIR}/${DATASET_DIR}/${DATASET}/${EXP_FOLDER}/vis"
EXPORT_DIR="${EXP_DIR}/${DATASET_DIR}/${DATASET}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}" --no-same-owner
cd "${CURRENT_DIR}"

# Train the model.
# initialize_last_layer=false only if finetuning on another dataset with a different number of classes
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --num_clones=${NUM_GPU} \
  --train_split="train" \
  --base_learning_rate="${INITIAL_LEARNING_RATE}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=$((4*${NUM_GPU})) \
  --training_number_of_steps="${NUM_TRAIN_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --initialize_last_layer=false \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset="${DATASET}" \
  --dataset_dir="${DATASET_TF_RECORD}"

# Run evaluation. This performs eval over the full val split.
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=513 \
  --eval_crop_size=513 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset="${DATASET}" \
  --dataset_dir="${DATASET_TF_RECORD}" \
  --max_number_of_evaluations=1

# Visualize the results.
#python "${WORK_DIR}"/vis.py \
#  --logtostderr \
#  --vis_split="val" \
#  --model_variant="xception_65" \
#  --atrous_rates=6 \
#  --atrous_rates=12 \
#  --atrous_rates=18 \
#  --output_stride=16 \
#  --decoder_output_stride=4 \
#  --vis_crop_size=513 \
#  --vis_crop_size=513 \
#  --checkpoint_dir="${TRAIN_LOGDIR}" \
#  --vis_logdir="${VIS_LOGDIR}" \
#  --dataset="${DATASET}" \
#  --dataset_dir="${DATASET_TF_RECORD}" \
#  --max_number_of_iterations=1

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=${NUM_CLASSES} \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
