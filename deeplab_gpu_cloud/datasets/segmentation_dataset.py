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

"""Provides data from semantic segmentation datasets.

The SegmentationDataset class provides both images and annotations (semantic
segmentation and/or instance segmentation) for TensorFlow. Currently, we
support the following datasets:

1. PASCAL VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

PASCAL VOC 2012 semantic segmentation dataset annotates 20 foreground objects
(e.g., bike, person, and so on) and leaves all the other semantic classes as
one background class. The dataset contains 1464, 1449, and 1456 annotated
images for the training, validation and test respectively.

2. Cityscapes dataset (https://www.cityscapes-dataset.com)

The Cityscapes dataset contains 19 semantic labels (such as road, person, car,
and so on) for urban street scenes.

3. ADE20K dataset (http://groups.csail.mit.edu/vision/datasets/ADE20K)

The ADE20K dataset contains 150 semantic labels both urban street scenes and
indoor scenes.

4. Informal Settlements dataset

The Informal Settlements dataset contains 1617 VHR satellite images with semantic binary labels
for informal settlements (e.g., slums, unplanned settlements, poor neighbourhoods).

References:
  M. Everingham, S. M. A. Eslami, L. V. Gool, C. K. I. Williams, J. Winn,
  and A. Zisserman, The pascal visual object classes challenge a retrospective.
  IJCV, 2014.

  M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
  U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban
  scene understanding," In Proc. of CVPR, 2016.

  B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, A. Torralba, "Scene Parsing
  through ADE20K dataset", In Proc. of CVPR, 2017.
"""
import collections
import os.path
import tensorflow as tf

slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    ['splits_to_sizes',   # Splits of the dataset into training, val, and test.
     'num_classes',   # Number of semantic classes, including the background
                      # class (if exists). For example, there are 20
                      # foreground classes + 1 background class in the PASCAL
                      # VOC 2012 dataset. Thus, we set num_classes=21.
     'ignore_label',  # Ignore label value.
    ]
)

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 2975,
        'val': 500,
    },
    num_classes=19,
    ignore_label=255,
)

_PASCAL_VOC_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1464,
        'train_aug': 10582,
        'trainval': 2913,
        'val': 1449,
    },
    num_classes=21,
    ignore_label=255,
)

# These number (i.e., 'train'/'test') seems to have to be hard coded
# You are required to figure it out for your training/testing example.
_ADE20K_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 20210,  # num of samples in images/training
        'val': 2000,  # num of samples in images/validation
    },
    num_classes=151,
    ignore_label=0,
)

_INF_SET_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1294,  # num of samples in images/training
        'val': 323,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_LOWER_512_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 58,  # num of samples in images/training
        'val': 14,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_KIBERA_512_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 250,  # num of samples in images/training
        'val': 62,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_EL_DAIEN_512_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1987,  # num of samples in images/training
        'val': 497,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_Al_GENEINA_512_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1479,  # num of samples in images/training
        'val': 370,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_KIBERA_AND_LOWER_512_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 308,  # num of samples in images/training
        'val': 76,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_ALL_512_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 3774,  # num of samples in images/training
        'val': 943,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_ALL_WITHOUT_SUDAN_512_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 367,  # num of samples in images/training
        'val': 91,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_ALL_WITHOUT_SMALL_CITIES_512_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 3774,  # num of samples in images/training
        'val': 943,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_LOWER_256_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 288,  # num of samples in images/training
        'val': 72,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_KIBERA_256_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1018,  # num of samples in images/training
        'val': 254,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_EL_DAIEN_256_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 8035,  # num of samples in images/training
        'val': 2009,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_Al_GENEINA_256_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 5986,  # num of samples in images/training
        'val': 1496,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_KIBERA_AND_LOWER_256_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1306,  # num of samples in images/training
        'val': 326,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_ALL_256_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 15597,  # num of samples in images/training
        'val': 3898,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_ALL_WITHOUT_SUDAN_256_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1576,  # num of samples in images/training
        'val': 393,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_ALL_WITHOUT_SMALL_CITIES_256_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 15327,  # num of samples in images/training
        'val': 3831,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_MEDELLIN_256_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 94,  # num of samples in images/training
        'val': 23,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_MAKOKO_256_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 176,  # num of samples in images/training
        'val': 44,  # num of samples in images/validation
    },
    num_classes=2,
    ignore_label=255, 
)

_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'inf_set': _INF_SET_INFORMATION,
    'el_daien_512': _EL_DAIEN_512_INFORMATION,
    'al_geneina_512': _Al_GENEINA_512_INFORMATION,
    'kibera_512':_KIBERA_512_INFORMATION,
    'lower_512':_LOWER_512_INFORMATION,
    'all_512':_ALL_512_INFORMATION,
    'all_without_sudan_512':_ALL_WITHOUT_SUDAN_512_INFORMATION,
    'all_without_small_cities_512':_ALL_WITHOUT_SMALL_CITIES_512_INFORMATION,
    'kibera_and_lower_512':_KIBERA_AND_LOWER_512_INFORMATION,
    'el_daien_256': _EL_DAIEN_256_INFORMATION,
    'al_geneina_256': _Al_GENEINA_256_INFORMATION,
    'kibera_256':_KIBERA_256_INFORMATION,
    'lower_256':_LOWER_256_INFORMATION,
    'all_256':_ALL_256_INFORMATION,
    'all_without_sudan_256':_ALL_WITHOUT_SUDAN_256_INFORMATION,
    'all_without_small_cities_256':_ALL_WITHOUT_SMALL_CITIES_256_INFORMATION,
    'kibera_and_lower_256':_KIBERA_AND_LOWER_256_INFORMATION,
    'medellin_256':_MEDELLIN_256_INFORMATION,
    'makoko_256':_MAKOKO_256_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'


def get_cityscapes_dataset_name():
  return 'cityscapes'


def get_dataset(dataset_name, split_name, dataset_dir):
  """Gets an instance of slim Dataset.

  Args:
    dataset_name: Dataset name.
    split_name: A train/val Split name.
    dataset_dir: The directory of the dataset sources.

  Returns:
    An instance of slim Dataset.

  Raises:
    ValueError: if the dataset_name or split_name is not recognized.
  """
  if dataset_name not in _DATASETS_INFORMATION:
    raise ValueError('The specified dataset %s is not supported yet.' % dataset_name)

  splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

  if split_name not in splits_to_sizes:
    raise ValueError('data split name %s not recognized' % split_name)

  # Prepare the variables for different datasets.
  num_classes = _DATASETS_INFORMATION[dataset_name].num_classes
  ignore_label = _DATASETS_INFORMATION[dataset_name].ignore_label

  file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Specify how the TF-Examples are decoded.
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/height': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/segmentation/class/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/segmentation/class/format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),
  }
  items_to_handlers = {
      'image': tfexample_decoder.Image(
          image_key='image/encoded',
          format_key='image/format',
          channels=3),
      'image_name': tfexample_decoder.Tensor('image/filename'),
      'height': tfexample_decoder.Tensor('image/height'),
      'width': tfexample_decoder.Tensor('image/width'),
      'labels_class': tfexample_decoder.Image(
          image_key='image/segmentation/class/encoded',
          format_key='image/segmentation/class/format',
          channels=1),
  }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=splits_to_sizes[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      ignore_label=ignore_label,
      num_classes=num_classes,
      name=dataset_name,
      multi_label=True)
