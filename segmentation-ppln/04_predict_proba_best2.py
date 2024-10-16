#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Supported by BiCubes project (HFRI grant 3943)

"""
This script processes satellite imagery data to classify and segment images using a pre-trained classifier.
It reads input images, computes indices, and applies a Random Forest classifier to generate prediction maps.

Dependencies:
- numpy
- osgeo (gdal)
- tqdm
- sklearn
- joblib
- natsort
- optparse
- sys
"""

import numpy as np
from osgeo import gdal
from osgeo.gdal_array import DatasetReadAsArray
from tqdm import tqdm
from glob import glob
from sklearn import preprocessing
import math
import os
import joblib
import natsort
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
  dataset = gdal.Open(filename, gdal.GA_ReadOnly)

  # Read image
  imdata = DatasetReadAsArray(dataset)

  # If there are multiple bands in the dataset
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
      if band_description in current_decription:
          return index

def read_partial(filename, i, parts):
  """
  Read a part of a geospatial image file.

  Parameters:
  - filename: str, the path to the image file.
  - i: int, the part index to read.
  - parts: int, the total number of parts to split the image into.

  Returns:
  - data_part: numpy array, the image data for the specified part.
  """
  # Open dataset for reading
  dataset_object = gdal.Open(filename, gdal.GA_ReadOnly)
  # Get dimensions
  cols = dataset_object.RasterXSize
  rows = dataset_object.RasterYSize
  bands = dataset_object.RasterCount
  # Split in parts
  row_step = rows // parts
  row_residual = rows - (row_step * parts)

  band_object = dataset_object.GetRasterBand(1)  # First band object
  band_part = band_object.ReadAsArray(0, row_step * i, cols, row_step)  # Load first band part
  np_dtype = band_part.dtype  # Acquire numpy dtype

  if i != parts-1:  # All iterations except last
      data_part = np.zeros((row_step, cols, bands), dtype=np_dtype)
  else:  # Last iteration
      data_part = np.zeros((row_step + row_residual, cols, bands), dtype=np_dtype)

  if bands == 1:
      band_object = dataset_object.GetRasterBand(1)
      if i != parts-1:  # All iterations except last
          data_part = band_object.ReadAsArray(0, row_step * i, cols, row_step)
      else:  # Last iteration
          data_part = band_object.ReadAsArray(0, row_step * i, cols, row_step + row_residual)
  else:
      for b in range(bands):
          band_index = b + 1
          band_object = dataset_object.GetRasterBand(band_index)
          if i != parts-1:  # All iterations except last
              data_part[:, :, b] = band_object.ReadAsArray(0, row_step * i, cols, row_step)
          else:  # Last iteration
              data_part[:, :, b] = band_object.ReadAsArray(0, row_step * i, cols, row_step + row_residual)

  dataset_object = None
  return data_part

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
  # Initialize by lowest possible value in a normalized index (-1)
  index = (-1) * np.ones((rows, cols), dtype=np.float32)
  # Only divide nonzeros else -1
  np.divide(b1 - b2, b1 + b2, out=index, where=(b1 + b2) != 0)
  index = (index + 1) / 2  # Normalize to 0 - 1
  index[index < 0] = 0
  index[index > 1] = 1

  return index

def readPartialSubsetAndNormalize(filename, i, parts, description_subset, selected_indices_names=['NDVI', 'NDWI', 'NDBI'], NORM_FACTOR=10000):
  """
  Read a subset of bands from a geospatial image file, normalize them, and compute indices.

  Parameters:
  - filename: str, the path to the image file.
  - i: int, the part index to read.
  - parts: int, the total number of parts to split the image into.
  - description_subset: list of str, descriptions of the bands to be read.
  - selected_indices_names: list of str, names of indices to compute.
  - NORM_FACTOR: float, normalization factor.

  Returns:
  - data_part: numpy array, the normalized image data with computed indices.
  """
  # Open file
  dataset_object = gdal.Open(filename, gdal.GA_ReadOnly)
  input_bands = dataset_object.RasterCount
  cols = dataset_object.RasterXSize
  rows = dataset_object.RasterYSize
  output_bands = len(description_subset)
  # Split in parts
  row_step = rows // parts
  row_residual = rows - (row_step * parts)
  if i != parts-1:  # All iterations except last
      data_part = np.zeros((row_step, cols, output_bands), dtype=np.float32)
  else:  # Last iteration
      data_part = np.zeros((row_step + row_residual, cols, output_bands), dtype=np.float32)
  for b in range(input_bands):
      band_index = b + 1
      band_object = dataset_object.GetRasterBand(band_index)
      current_description = band_object.GetDescription()
      for v, valid_description in enumerate(description_subset):
          if current_description == valid_description:
              band_object = dataset_object.GetRasterBand(band_index)
              if i != parts-1:  # All iterations except last
                  data_part[:, :, v] = band_object.ReadAsArray(0, row_step * i, cols, row_step)
              else:  # Last iteration
                  data_part[:, :, v] = band_object.ReadAsArray(0, row_step * i, cols, row_step + row_residual)
  dataset_object = None
  data_part = data_part / NORM_FACTOR
  data_part[data_part < 0] = 0
  data_part[data_part > 1] = 1
  # Get band indices
  b3_index = getBandIndexFromDescriptions('B3', band_description_subset)
  b4_index = getBandIndexFromDescriptions('B4', band_description_subset)
  b8_index = getBandIndexFromDescriptions('B8', band_description_subset)
  b11_index = getBandIndexFromDescriptions('B11', band_description_subset)
      
  if 'NDVI' in selected_indices_names:
      ndvi = computeNormalizedDifIndex(data_part[:, :, b8_index], data_part[:, :, b4_index])  # NDVI B08, B04
      data_part = np.dstack((data_part, ndvi))  # Stack
  if 'NDWI' in selected_indices_names:
      ndwi = computeNormalizedDifIndex(data_part[:, :, b3_index], data_part[:, :, b8_index])  # NDWI B03, B08
      data_part = np.dstack((data_part, ndwi))  # Stack
  if 'NDBI' in selected_indices_names:
      ndbi = computeNormalizedDifIndex(data_part[:, :, b11_index], data_part[:, :, b8_index])  # NDBI B11, B08
      data_part = np.dstack((data_part, ndbi))  # Stack
  return data_part

def write_partial(fullpath, imdata, i, row_step):
  """
  Write a part of an image to a geospatial file.

  Parameters:
  - fullpath: str, the path to the output file.
  - imdata: numpy array, the image data to write.
  - i: int, the part index to write.
  - row_step: int, the number of rows in each part.
  """
  imshape = imdata.shape
  outDataset = gdal.Open(fullpath, 1)
  if np.size(imshape) == 3:
      bands = imshape[2]
      for b in range(bands):
          outBand = outDataset.GetRasterBand(b+1)
          outBand.WriteArray(imdata[:, :, b], 0, row_step * i)
          outBand = None
  elif np.size(imshape) == 2:
      outBand = outDataset.GetRasterBand(1)
      outBand.WriteArray(imdata, 0, row_step * i)
      outBand = None
  outDataset = None
      
  return

def tile_callback(option, opt, value, parser):
  """
  Callback function to handle comma-separated list options.
  """
  setattr(parser.values, option.dest, value.split(','))

def temporal_image(input_image, input_features_per_date, temporal_feature_count):
  """
  Compute temporal statistics for each feature in the input image.

  Parameters:
  - input_image: numpy array, the input image data.
  - input_features_per_date: int, number of features per date.
  - temporal_feature_count: int, number of temporal features to compute.

  Returns:
  - temporal_image: numpy array, the computed temporal features.
  """
  # Use a more convenient variable name
  f = input_features_per_date

  # Image dimensions
  rows, cols, depth = input_image.shape
  
  # If applicable, remove extra features (DEM, CLC, etc.)
  extra_features = depth % input_features_per_date
  if extra_features > 0:
      input_image = input_image[:, :, :-extra_features]

  # Initialize temporal image
  temporal_image = np.zeros((rows, cols, f * temporal_feature_count), dtype=np.float32)

  k = 0
  i = 0
  while i < f:
      # Calculate min per band
      temporal_image[:, :, k] = np.nanmin(input_image[:, :, i::f], axis=2)
      k += 1
      # Calculate percentiles per band
      temporal_image[:, :, k:k+5] = np.moveaxis(np.percentile(input_image[:, :, i::f], q=[10, 25, 50, 75, 90], axis=2), 0, -1)  # Moving first axis to last position, because percentile output has different axes
      k += 5
      # Calculate max per band
      temporal_image[:, :, k] = np.nanmax(input_image[:, :, i::f], axis=2)
      k += 1
      # Calculate mean per band
      temporal_image[:, :, k] = np.nanmean(input_image[:, :, i::f], axis=2)
      k += 1
      # Calculate std per band
      temporal_image[:, :, k] = np.nanstd(input_image[:, :, i::f], axis=2)
      k += 1
      i += 1

  return temporal_image

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
  parser.add_option("-t", dest="tile_name", action="store", type="string", help="Sentinel-2 tile name in format NNCCC e.g. 34SEG.", default=None)
  parser.add_option("-i", dest="input_path", action="store", type="string", help="Path to input images.", default=None)
  parser.add_option("-o", dest="output_path", action="store", type="string", help="Output path for this experiment's results. Will be created if it does not exist. Features will be saved in 'features' dir.", default=None)
  parser.add_option("-c", dest="classifier_pkl_filename", action="store", type="string", help="Optional. Classifier exact .pkl filename, must be present in 'models' directory of output_path. If not provided a new classifier will be trained. ", default=None)
  parser.add_option("-f", dest="feature_type", action="store", type="string", help="Feature type(ts,tf).", default=None)
  parser.add_option("--sea", dest="sea_mask_path", action="store", type="string", help="Path to stored sea masks.", default=None)
  
  # Optional
  parser.add_option("-p", dest="input_pattern", action="store", type="string", help="Optional. Glob pattern all input images adhere to. If not provided, all tif files in directory will be used.", default='*.tif')
  parser.add_option("-b", dest="band_description_subset", action="callback", type="string", help="Optional. Band descriptions subseting data in input path matching input pattern. If multiple separate by comma. Default value: B2,B3,B4,B5,B8,B11,B12", default=['B2','B3','B4','B5','B8','B11','B12'], callback=tile_callback)
  parser.add_option("-s", dest="selected_indices_names", action="callback", type="string", help="Optional. Select indices to compute from those implemented (NDVI, NDWI, NDBI). If multiple separate by comma. Will be computed. Descriptions for needed input bands must be in format B3,B4 etc. Default value: NDVI,NDWI,NDBI", default=['NDVI','NDWI','NDBI'], callback=tile_callback)
  parser.add_option("-d", dest="stats_descriptions", action="callback", type="string", help="Optional. Band descriptions subseting data in input path matching input pattern. If multiple separate by comma. Default value: min,10%,25%,median,75%,90%,max,mean,std", default=['min', '10%', '25%', 'median', '75%', '90%', 'max', 'mean', 'std'], callback=tile_callback)
  parser.add_option("-a", dest="auxiliary_features_paths", action="callback", type="string", help="Optional. Path/s to additional features. If multiple separate by comma. Each of these paths must have a single file corresponding to the current tile.", default=None, callback=tile_callback)
  parser.add_option("-n", dest="auxiliary_features_normfactor", action="callback", type="string", help="Optional. Normalization factor/s by which to normalize auxiliary features to 0-1. Any values smaller than 0 or larger than 1 after division by normfactor will be CLIPPED to 0 or 1 respectively.", default=None, callback=tile_callback)
  
  parser.add_option("--ts", dest="timeseries_feature_name", action="store", type="string", help="Optional. Part of the output .npy name, relevant to timeseries features. Default value: ts ", default='ts')
  parser.add_option("--tf", dest="temporal_feature_name", action="store", type="string", help="Optional. Part of the output .npy name, relevant to temporal features. Default value: tf ", default='tf')
  
  (options, args) = parser.parse_args()
  # Checking required arguments for the script
  parser.check_required("-t")
  parser.check_required("-i")
  parser.check_required("-o")
  parser.check_required("-c")
  parser.check_required("-f")

tile_name = options.tile_name
input_path = options.input_path
output_path = options.output_path
classifier_pkl_filename = options.classifier_pkl_filename
feature_type = options.feature_type
sea_mask_path = options.sea_mask_path

input_pattern = options.input_pattern
band_description_subset = options.band_description_subset
selected_indices_names = options.selected_indices_names
stats_descriptions = options.stats_descriptions
auxiliary_features_paths = options.auxiliary_features_paths
auxiliary_features_normfactor = options.auxiliary_features_normfactor
timeseries_feature_name = options.timeseries_feature_name
temporal_feature_name = options.temporal_feature_name

OUTPUT_SEA_VALUE = 0

# Folders with input images
classifier_path = os.path.join(output_path, 'models')
maps_path = os.path.join(output_path, 'maps')
temporal_feature_count = len(stats_descriptions)

# Get files
os.chdir(input_path)
files = natsort.natsorted(glob(options.input_pattern))  # Sort data images
print(files)
path = os.path.normpath(input_path)

# Get georeference from first file
dataset = gdal.Open(files[0], gdal.GA_ReadOnly)
drv = dataset.GetDriver().ShortName
gtr = dataset.GetGeoTransform()
proj = dataset.GetProjection()
cols = dataset.RasterXSize
rows = dataset.RasterYSize
dataset = None

# Load classifier
os.chdir(classifier_path)
clf_RF = joblib.load(classifier_pkl_filename)

# Create label encoder (to translate class codes in classifier, to label codes in map)
unique_labels = clf_RF.classes_
le = preprocessing.LabelEncoder()
le.fit(unique_labels + 1)
print(le.classes_)

# Initialize output map
if not os.path.exists(maps_path):
  os.mkdir(maps_path)
os.chdir(maps_path)
savename = '_'.join(['Prediction_Probabilities_Map', tile_name, 'clf', classifier_pkl_filename.split('.')[0]]) + '.tif'  # Output name
driver = gdal.GetDriverByName(drv)
outDataset = driver.Create(savename, cols, rows, 4, gdal.GDT_Byte)  # Output map is uint8
outDataset.SetGeoTransform(gtr)
outDataset.SetProjection(proj)
outDataset = None

# Handle input partially
fpd = len(band_description_subset) + len(selected_indices_names)
total_feature_count = fpd * len(files)  # Excluding DEM, will be appended last
parts = 45  # Split in parts
row_step = rows // parts
row_residual = rows - (row_step * parts)

SEA_THRESHOLD = 0.15

# Get auxiliary features
if auxiliary_features_paths != None:
  aux_count = len(auxiliary_features_paths)
  aux_stack = np.zeros((rows, cols, aux_count), dtype=np.float32)
  for a, auxiliary_feature_path in enumerate(auxiliary_features_paths):
      if auxiliary_features_normfactor == None:
          print('Please provide normalization factor for', auxiliary_feature_path)
          sys.exit(-1)
      os.chdir(auxiliary_feature_path)
      aux_filename = glob_and_check(auxiliary_feature_path, '*' + tile_name + '*')
      aux_stack[:, :, a] = read(aux_filename)[0] / int(auxiliary_features_normfactor[a])
      
for i in tqdm(range(parts)):
  # Image part range & count
  p1 = cols * row_step * i
  if i != parts-1:  # All iterations except last
      p2 = cols * row_step * (i+1)
  else:  # Last iteration
      p2 = cols * rows
  pixel_count = p2 - p1
  
  if i != parts-1:  # All iterations except last
      data_part = np.zeros((row_step, cols, total_feature_count), dtype=np.float32)
  else:  # Last iteration
      data_part = np.zeros((row_step + row_residual, cols, total_feature_count), dtype=np.float32)    
      
  # Load sea mask
  os.chdir(sea_mask_path)
  sea_mask_filename = glob_and_check(sea_mask_path, '*' + tile_name + '*')
  sea_mask_part = read_partial(sea_mask_filename, i, parts)
  sea_mask_per_pixel = np.reshape(sea_mask_part == 0, (pixel_count, 1))         
  sea_count = np.sum(sea_mask_per_pixel)
  del sea_mask_part

  if sea_count == pixel_count:  # ALL SEA
      print('This part of the image is all sea.')  
      preds = np.zeros((pixel_count, 1), dtype=np.uint8)
      predicted_labels = np.reshape(preds, (-1, cols))
      probs = 100 * np.ones((pixel_count, 1), dtype=np.uint8)
      sec_preds = preds
      sec_probs = probs
      probabilities = np.reshape(probs, (-1, cols))        
      os.chdir(maps_path)
      output_stacked = np.dstack((predicted_labels, probabilities))
      write_partial(savename, output_stacked.astype(np.uint8), i, row_step)
  else:  # LOAD DATA, CREATE TIMESERIES DATA PART
      os.chdir(input_path)
      # Get timeseries features
      for d, f in enumerate(files):
          data_part[:, :, d*fpd:d*fpd+fpd] = readPartialSubsetAndNormalize(f, i, parts, band_description_subset, selected_indices_names)   
      # Get temporal features
      if feature_type == temporal_feature_name:
          data_part = temporal_image(data_part, fpd, temporal_feature_count) 
      # Stack auxiliary features part
      if auxiliary_features_paths != None:
          if i != parts-1:  # All iterations except last
              data_part = np.dstack((data_part, aux_stack[i*row_step:(i+1)*row_step, :, :]))
          else:  # Last iteration
              data_part = np.dstack((data_part, aux_stack[i*row_step:(i+1)*row_step+row_residual, :, :]))
      
      # Handle NaN values
      # Where are there NaN values in our data
      data_nan_positions = np.isnan(data_part)
      data_inf_positions = np.isinf(data_part)
      # Replace NaN with a very high value
      data_part[data_nan_positions] = 65535
      data_part[data_inf_positions] = 65535
      data_nan_positions = data_nan_positions[:, :, 0]
      data_inf_positions = data_inf_positions[:, :, 0]
          
      # Resize 2d array to 1d array
      data_per_pixel = np.reshape(data_part, (pixel_count, -1))
      feature_count = data_per_pixel.shape[1]
      del data_part
      
      if sea_count == 0:  # ALL LAND
          print('This part of the image is all land.') 
          proba = clf_RF.predict_proba(data_per_pixel)
          sort_pos = np.argsort(proba, axis=1)
          max_pos = sort_pos[:, -1]
          sec_max_pos = sort_pos[:, -2]
          preds = le.inverse_transform(max_pos)
          sec_preds = le.inverse_transform(sec_max_pos)
          sort_val = np.sort(proba, axis=1)
          probs = 100 * sort_val[:, -1]
          sec_probs = 100 * sort_val[:, -2]
          del data_per_pixel, max_pos
          
      elif sea_count < SEA_THRESHOLD * pixel_count:  # MIXED LAND AND SEA (SEA less than SEA THRESHOLD * pixel count )
          print(np.round(100 * sea_count / pixel_count, 2), ' % Sea in this image. Case 1.') 
          proba = clf_RF.predict_proba(data_per_pixel)
          sort_pos = np.argsort(proba, axis=1)
          max_pos = sort_pos[:, -1]
          sec_max_pos = sort_pos[:, -2]
          preds = le.inverse_transform(max_pos)
          sec_preds = le.inverse_transform(sec_max_pos)
          sort_val = np.sort(proba, axis=1)
          probs = 100 * sort_val[:, -1]
          sec_probs = 100 * sort_val[:, -2]

          preds = np.reshape(preds, (pixel_count, 1))
          preds = np.where(sea_mask_per_pixel == True, 0, preds)
          probs = np.reshape(probs, (pixel_count, 1))
          probs = np.where(sea_mask_per_pixel == True, 100, probs)
          sec_preds = np.reshape(sec_preds, (pixel_count, 1))
          sec_preds = np.where(sea_mask_per_pixel == True, 0, sec_preds)
          sec_probs = np.reshape(sec_probs, (pixel_count, 1))
          sec_probs = np.where(sea_mask_per_pixel == True, 100, sec_probs)
          del data_per_pixel, max_pos, sec_max_pos, sort_val
          
      else:  # MIXED LAND AND SEA ((SEA more than SEA THRESHOLD * pixel count )
          print(np.round(100 * sea_count / pixel_count, 2), ' % Sea in this image. Case 2.') 
          land_data_per_pixel = []
          for p in range(pixel_count):
              if sea_mask_per_pixel[p] == False:
                  land_data_per_pixel.append(data_per_pixel[p, :])
          mixed_output = []
          proba_output = []
          sec_mixed_output = []
          sec_proba_output = []

          land_data_per_pixel = np.asarray(land_data_per_pixel)
          proba = clf_RF.predict_proba(land_data_per_pixel) 

          sort_pos = np.argsort(proba, axis=1)
          max_pos = sort_pos[:, -1]
          sec_max_pos = sort_pos[:, -2]
          preds = le.inverse_transform(max_pos)
          sec_preds = le.inverse_transform(sec_max_pos)
          sort_val = np.sort(proba, axis=1)
          probs = 100 * sort_val[:, -1]
          sec_probs = 100 * sort_val[:, -2]

          c = 0            
          for p in range(pixel_count):
              if sea_mask_per_pixel[p] == False:
                  mixed_output.append(preds[c])
                  sec_mixed_output.append(sec_preds[c])
                  proba_output.append(probs[c])
                  sec_proba_output.append(sec_probs[c])
                  c += 1
              else:
                  mixed_output.append(OUTPUT_SEA_VALUE)  # Insert no data values
                  sec_mixed_output.append(OUTPUT_SEA_VALUE)  # Insert no data values
                  proba_output.append(100)  # Insert no data values
                  sec_proba_output.append(100)
          probs = np.array(proba_output, dtype=np.uint8)
          preds = np.array(mixed_output, dtype=np.uint8)
          sec_probs = np.array(sec_proba_output, dtype=np.uint8)
          sec_preds = np.array(sec_mixed_output, dtype=np.uint8)
          del data_per_pixel, max_pos, sec_max_pos, land_data_per_pixel, mixed_output

      predicted_labels = np.reshape(preds, (-1, cols))
      probabilities = np.reshape(probs, (-1, cols))
      sec_predicted_labels = np.reshape(sec_preds, (-1, cols))
      sec_probabilities = np.reshape(sec_probs, (-1, cols))
      os.chdir(maps_path)
      try:
          predicted_labels[data_nan_positions] = 0
          predicted_labels[data_inf_positions] = 0
          sec_predicted_labels[data_nan_positions] = 0
          sec_predicted_labels[data_inf_positions] = 0
          output_stacked = np.dstack((
              predicted_labels,
              probabilities,
              sec_predicted_labels,
              sec_probabilities,
          ))
          write_partial(savename, output_stacked.astype(np.uint8), i, row_step)
      except:
          predicted_labels[data_nan_positions] = math.nan
          predicted_labels[data_inf_positions] = math.nan
          sec_predicted_labels[data_nan_positions] = math.nan
          sec_predicted_labels[data_inf_positions] = math.nan
          output_stacked = np.dstack((
              predicted_labels,
              probabilities,
              sec_predicted_labels,
              sec_probabilities,
          ))
          write_partial(savename, output_stacked, i, row_step)
