#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Supported by BiCubes project (HFRI grant 3943)

"""
This script processes satellite imagery data to extract features and compute indices such as NDVI, NDWI, and NDBI.
It reads input images, normalizes them, and extracts features based on provided band descriptions.
The script also handles auxiliary features and saves the processed data for further analysis.

Dependencies:
- numpy
- osgeo (gdal)
- glob
- natsort
- os
- tqdm
- optparse
- sys
"""

import numpy as np
from osgeo import gdal
from osgeo.gdal_array import DatasetReadAsArray
from glob import glob
import natsort
import os
from tqdm import tqdm
import optparse
import sys

class OptionParser(optparse.OptionParser):
  """
  Custom option parser to handle required arguments.
  """
  def check_required(self, opt):
      """
      Check if a required option is provided.
      """
      option = self.get_option(opt)
      # Assumes the option's 'default' is set to None!
      if getattr(self.values, option.dest) is None:
          self.error("{} option is not supplied. {} is required for this script!".format(option, option))

def getFeatures(data_image, labels_image):
  """
  Extract features and labels from the input images.

  Parameters:
  - data_image: numpy array, the data image to extract features from.
  - labels_image: numpy array, the labels image for ground truth.

  Returns:
  - feature_table: numpy array, the extracted features.
  - labels_vector: numpy array, the corresponding labels.
  """
  # Get image rows and cols
  rows = labels_image.shape[0]
  cols = labels_image.shape[1]
  # Reshape from 3d image to 2d matrix
  data_all = np.reshape(data_image, (rows*cols, -1))
  del data_image
  # Reshape from 2d image to 1d vector
  labels_all = np.reshape(labels_image, (rows*cols, ))
  del labels_image

  # These are the positions with known ground truth
  groundtruth_positions = labels_all > 0
  # Ignore values that we have no ground truth for
  feature_table = data_all[groundtruth_positions, :]
  del data_all
  # Ignore values that we have no ground truth for
  labels_vector = labels_all[groundtruth_positions]

  # First feature id must be 0
  labels_vector = labels_vector - 1.0

  # Ignore NaN values
  # Where are there NaN values in our data
  nan_positions = np.isnan(feature_table)
  # Which are the rows where we have no NaN values
  no_nan_rows = np.sum(nan_positions, axis=1) == 0
  # Only keep rows without NaN
  feature_table = feature_table[no_nan_rows, :]
  labels_vector = labels_vector[no_nan_rows]

  # Labels are integer
  labels_vector.astype(np.uint8)

  return feature_table, labels_vector

def read(filename):
  """
  Read a geospatial image file and return its data and metadata.

  Parameters:
  - filename: str, the path to the image file.

  Returns:
  - imdata: numpy array, the image data.
  - geoTransform: tuple, geotransformation parameters.
  - proj: str, projection information.
  - drv_name: str, driver name (file type).
  """
  # Open file
  print(f"Will read {filename}")
  dataset = gdal.Open(filename, gdal.GA_ReadOnly)

  # Read image
  imdata = DatasetReadAsArray(dataset)

  # If there are multiple bands in the dataset
  # (meaning: if there is a third dimension on imdata)
  if len(imdata.shape) == 3:
      imdata = np.swapaxes(imdata, 0, 1)
      imdata = np.swapaxes(imdata, 1, 2)

  # Get the Driver name (filetype)
  drv_name = dataset.GetDriver().ShortName

  # Get Georeference info
  geoTransform = dataset.GetGeoTransform()
  proj = dataset.GetProjection()

  # Clear variable
  dataset = None

  return imdata, geoTransform, proj, drv_name

def readSubsetAndNormalize(filename, description_subset, NORM_FACTOR=10000):
  """
  Read a subset of bands from a geospatial image file and normalize them.

  Parameters:
  - filename: str, the path to the image file.
  - description_subset: list of str, descriptions of the bands to be read.
  - NORM_FACTOR: float, normalization factor.

  Returns:
  - imdata: numpy array, the normalized image data.
  - geoTransform: tuple, geotransformation parameters.
  - proj: str, projection information.
  - drv_name: str, driver name (file type).
  """
  # Open file
  dataset = gdal.Open(filename, gdal.GA_ReadOnly)
  input_bands = dataset.RasterCount
  cols = dataset.RasterXSize
  rows = dataset.RasterYSize
  output_bands = len(description_subset)

  imdata = np.zeros((rows, cols, output_bands), dtype=np.float32)
  for b in range(input_bands):
      band_index = b + 1
      band_object = dataset.GetRasterBand(band_index)
      current_description = band_object.GetDescription()
      for v, valid_description in enumerate(description_subset):
          if current_description == valid_description:
              band_array = band_object.ReadAsArray()
              band_array = band_array / NORM_FACTOR
              band_array[band_array < 0] = 0
              band_array[band_array > 1] = 1
              imdata[:, :, v] = band_array
      band_object = None

  drv_name = dataset.GetDriver().ShortName  # Get the Driver name (filetype)
  geoTransform = dataset.GetGeoTransform()  # Get Georeference info
  proj = dataset.GetProjection()
  dataset = None  # Clear variable

  return imdata, geoTransform, proj, drv_name

def computeNormalizedDifIndex(b1, b2):
  """
  Compute a normalized difference index between two bands.

  Parameters:
  - b1: numpy array, first band.
  - b2: numpy array, second band.

  Returns:
  - index: numpy array, the computed normalized difference index.
  """
  rows, cols = b1.shape
  # initialize by lowest possible value in a normalized index (-1)
  index = (-1) * np.ones((rows, cols), dtype=np.float32)
  # only divide nonzeros else -1
  np.divide(b1 - b2, b1 + b2, out=index, where=(b1 + b2) != 0)
  index = (index + 1) / 2  # Normalize to 0 - 1
  index[index < 0] = 0
  index[index > 1] = 1

  return index

def getBandIndexFromDescriptions(band_description, description_subset):
  """
  Get the index of a band description from a subset.

  Parameters:
  - band_description: str, the band description to find.
  - description_subset: list of str, the subset of descriptions.

  Returns:
  - index: int, the index of the band description in the subset.
  """
  for index, current_decription in enumerate(description_subset):
      if band_description == current_decription:
          return index

def tile_callback(option, opt, value, parser):
  """
  Callback function to handle comma-separated list options.
  """
  setattr(parser.values, option.dest, value.split(','))

def glob_and_check(path, pattern):
  """
  Find files matching a pattern in a directory and ensure only one file matches.

  Parameters:
  - path: str, the directory path.
  - pattern: str, the glob pattern to match files.

  Returns:
  - output: str, the matched file path.
  """
  os.chdir(path)
  files = glob(pattern)
  num_files = len(files)
  if num_files > 1:
      print(files)
      print('All of the above files match the input pattern. Please use a more specific pattern or rearrange files in different directories')
      sys.exit(-1)
  elif num_files == 1:
      output = files[0]
      print('Accessing', output)
      return output

if len(sys.argv) == 1:
  prog = os.path.basename(sys.argv[0])
  print(sys.argv[0] + ' [options]')
  print("Run: python3 ", prog, " --help")
  sys.exit(-1)
else:
  usage = "usage: %prog [options]"
  parser = OptionParser(usage=usage)
  # Building parameters
  parser.add_option("-i", dest="input_path", action="store", type="string", help="Path to input images.", default=None)
  parser.add_option("-t", dest="tile_name", action="store", type="string", help="Sentinel-2 tile name in format NNCCC e.g. 34SEG.", default=None)
  parser.add_option("-r", dest="reference_data_path", action="store", type="string", help="Path to reference data. Reference data names must follow the pattern '*'+tile_name+'*_train.tif' and '*'+tile_name+'*_test.tif' ", default=None)
  parser.add_option("-o", dest="output_path", action="store", type="string", help="Output path for this experiment's results. Will be created if it does not exist. Features will be saved in 'features' dir.", default=None)
  # optional
  parser.add_option("-p", dest="input_pattern", action="store", type="string", help="Optional. Glob pattern all input images adhere to. If not provided, all tif files in directory will be used.", default='*.tif')
  parser.add_option("-d", dest="band_description_subset", action="callback", type="string", help="Band descriptions subseting data in input path matching input pattern. If multiple separate by comma. Default value: B2,B3,B4,B5,B8,B11,B12", default=['B2','B3','B4','B5','B8','B11','B12'], callback=tile_callback)
  parser.add_option("-s", dest="selected_indices_names", action="callback", type="string", help="Optional. Select indices to compute from those implemented (NDVI, NDWI, NDBI). If multiple separate by comma. Will be computed. Descriptions for needed input bands must be in format B3,B4 etc. Default value: NDVI,NDWI,NDBI", default=['NDVI','NDWI','NDBI'], callback=tile_callback)
  parser.add_option("-a", dest="auxiliary_features_paths", action="callback", type="string", help="Optional. Path/s to additional features. If multiple separate by comma. Each of these paths must have a single file corresponding to the current tile.", default=None, callback=tile_callback)
  parser.add_option("-l", dest="auxiliary_features_labels", action="callback", type="string", help="Optional. Label name/s for additional features, used for output .npy files. If multiple separate by comma.", default=None, callback=tile_callback)
  parser.add_option("-n", dest="auxiliary_features_normfactor", action="callback", type="string", help="Optional. Normalization factor/s by which to normalize auxiliary features to 0-1. Any values smaller than 0 or larger than 1 after division by normfactor will be CLIPPED to 0 or 1 respectively.", default=None, callback=tile_callback)
  parser.add_option("-f", dest="timeseries_feature_name", action="store", type="string", help="Optional. Part of the output .npy name, relevant to timeseries features. Default value: ts ", default='ts')
  
  (options, args) = parser.parse_args()
  # Checking required arguments for the script
  parser.check_required("-i")
  parser.check_required("-t")
  parser.check_required("-r")
  parser.check_required("-o")

input_path = options.input_path
tile_name = options.tile_name
band_description_subset = options.band_description_subset
reference_data_path = options.reference_data_path
output_path = options.output_path

input_pattern = options.input_pattern
auxiliary_features_paths = options.auxiliary_features_paths
auxiliary_features_labels = options.auxiliary_features_labels
auxiliary_features_normfactor = options.auxiliary_features_normfactor
selected_indices_names = options.selected_indices_names
timeseries_feature_name = options.timeseries_feature_name

# create output path if it does not exist
if not os.path.exists(output_path):
  os.mkdir(output_path)
# create features path if it does not exist
features_path = os.path.join(output_path, 'features')
if not os.path.exists(features_path):
  os.mkdir(features_path)

os.chdir(input_path)
files = natsort.natsorted(glob(input_pattern))  # sort data images

gt_train_name = glob_and_check(reference_data_path, '*' + tile_name + '*_train.tif')
gt_test_name = glob_and_check(reference_data_path, '*' + tile_name + '*_test.tif')
print(f"reference_data_path: {reference_data_path}")
print(f"tile_name: {tile_name}")
print(f"gt_train_name: {gt_train_name}")
labels_train_image, geoTransform, proj, drv_name = read(gt_train_name)
labels_test_image, geoTransform, proj, drv_name = read(gt_test_name)

# get band indices
b3_index = getBandIndexFromDescriptions('B3', band_description_subset)
b4_index = getBandIndexFromDescriptions('B4', band_description_subset)
b8_index = getBandIndexFromDescriptions('B8', band_description_subset)
b11_index = getBandIndexFromDescriptions('B11', band_description_subset)

total_feature_count = (len(band_description_subset) + len(selected_indices_names)) * len(files)
os.chdir(input_path)
for d, f in enumerate(tqdm(files)):
  im, gtr, proj, drv = readSubsetAndNormalize(f, band_description_subset)  # Read subset and normalize data image
  if 'NDVI' in selected_indices_names:
      ndvi = computeNormalizedDifIndex(im[:, :, b8_index], im[:, :, b4_index])  # NDVI B08, B04
      im = np.dstack((im, ndvi))  # stack
  if 'NDWI' in selected_indices_names:
      ndwi = computeNormalizedDifIndex(im[:, :, b3_index], im[:, :, b8_index])  # NDWI B03, B08
      im = np.dstack((im, ndwi))  # stack
  if 'NDBI' in selected_indices_names:
      ndbi = computeNormalizedDifIndex(im[:, :, b11_index], im[:, :, b8_index])  # NDBI B11, B08
      im = np.dstack((im, ndbi))  # stack
  # Create Feature Table and Labels Vector
  train_FT, train_LV = getFeatures(im, labels_train_image)
  test_FT, test_LV = getFeatures(im, labels_test_image)
  if d == 0:  # initialize
      timeseries_train_FT = np.copy(train_FT)
      timeseries_test_FT = np.copy(test_FT)
  else:  # stack
      timeseries_train_FT = np.hstack((timeseries_train_FT, train_FT))
      timeseries_test_FT = np.hstack((timeseries_test_FT, test_FT))

print('Saving timeseries features for', tile_name)
print(timeseries_train_FT)
print(timeseries_test_FT)

# Save npy files
os.chdir(features_path)
np.save('_'.join([tile_name, timeseries_feature_name, 'train', 'FT.npy']), timeseries_train_FT)
np.save('_'.join([tile_name, 'train', 'LV.npy']), train_LV.astype(np.uint8))
np.save('_'.join([tile_name, timeseries_feature_name, 'test', 'FT.npy']), timeseries_test_FT)
np.save('_'.join([tile_name, 'test', 'LV.npy']), test_LV.astype(np.uint8))

# Get auxiliary features
if auxiliary_features_paths is not None:
  for e, auxiliary_feature_path in enumerate(auxiliary_features_paths):
      if auxiliary_features_normfactor is None:
          print('Please provide normalization factor for', auxiliary_feature_path)
          sys.exit(-1)
      os.chdir(auxiliary_feature_path)
      print('Processing auxiliary feature', auxiliary_features_labels[e], 'for', tile_name, 'from', auxiliary_feature_path)
      print('Normalization factor is', auxiliary_features_normfactor[e])
      auxiliary_feature_filename = glob_and_check(auxiliary_feature_path, '*' + tile_name + '*')
      auxiliary_feature_image, geoTransform, proj, drv_name = read(auxiliary_feature_filename)
      aux_train_FT, train_LV = getFeatures(auxiliary_feature_image, labels_train_image)
      aux_test_FT, test_LV = getFeatures(auxiliary_feature_image, labels_test_image)
      aux_train_FT = aux_train_FT / int(auxiliary_features_normfactor[e])
      aux_test_FT = aux_test_FT / int(auxiliary_features_normfactor[e])
      aux_train_FT[aux_train_FT > 1] = 1
      aux_test_FT[aux_test_FT > 1] = 1
      aux_train_FT[aux_train_FT < 0] = 0
      aux_test_FT[aux_test_FT < 0] = 0
      os.chdir(features_path)
      np.save('_'.join([tile_name, auxiliary_features_labels[e], 'train', 'FT.npy']), aux_train_FT.astype(np.float32))
      np.save('_'.join([tile_name, auxiliary_features_labels[e], 'test', 'FT.npy']), aux_test_FT.astype(np.float32))
      print(aux_train_FT)
      print(aux_test_FT)
