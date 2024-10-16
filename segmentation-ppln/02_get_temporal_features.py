#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Supported by BiCubes project (HFRI grant 3943)

"""
This script processes temporal features from satellite imagery data.
It reads input feature tables, computes temporal statistics, and saves the results.
The script is designed to handle Sentinel-2 data and requires specific input formats.

Dependencies:
- numpy
- osgeo (gdal)
- os
- sys
- optparse
"""

import numpy as np
import os
from osgeo import gdal
from osgeo.gdal_array import DatasetReadAsArray
import sys
import optparse

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

def temporal_table(input_table, input_features_per_date, temporal_feature_count):
  """
  Compute temporal statistics for each feature in the input table.

  Parameters:
  - input_table: numpy array, the input feature table.
  - input_features_per_date: int, number of features per date.
  - temporal_feature_count: int, number of temporal features to compute.

  Returns:
  - temporal_table: numpy array, the computed temporal features.
  - extra_features: int, number of extra features not part of the temporal computation.
  """
  # Use a more convenient variable name
  f = input_features_per_date

  # Image dimensions
  rows, cols = input_table.shape
  
  # If applicable, remove extra features (DEM, CLC, etc.)
  extra_features = cols % input_features_per_date
  if extra_features > 0:
      input_table = input_table[:, :-extra_features]
  
  # Initialize temporal image
  temporal_table = np.zeros((rows, f * temporal_feature_count), dtype=np.float32)

  k = 0
  i = 0
  while i < f:
      # Calculate min per band
      temporal_table[:, k] = np.nanmin(input_table[:, i::f], axis=1)
      k += 1
      # Calculate percentiles per band
      temporal_table[:, k:k+5] = np.transpose(np.percentile(input_table[:, i::f], q=[10, 25, 50, 75, 90], axis=1))
      k += 5
      # Calculate max per band
      temporal_table[:, k] = np.nanmax(input_table[:, i::f], axis=1)
      k += 1
      # Calculate mean per band
      temporal_table[:, k] = np.nanmean(input_table[:, i::f], axis=1)
      k += 1
      # Calculate std per band
      temporal_table[:, k] = np.nanstd(input_table[:, i::f], axis=1)
      k += 1
      i += 1
      
  return temporal_table, extra_features

def read(filename):
  """
  Read a geospatial image file and return its data and metadata.

  Parameters:
  - filename: str, the path to the image file.

  Returns:
  - imdata: numpy array, the image data.
  - bandDescriptions: list of str, descriptions of each band.
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

  bandDescriptions = []
  # Get Band Description
  for i in range(imdata.shape[2]):
      band_index = i + 1
      band_object = dataset.GetRasterBand(band_index)
      bandDescriptions.append(band_object.GetDescription())

  # Clear variable
  dataset = None

  return imdata, bandDescriptions, geoTransform, proj, drv_name

def write(filename, imdata, feature_descriptions, geoTransform, proj, drv_name):
  """
  Write image data to a geospatial file with metadata.

  Parameters:
  - filename: str, the path to the output file.
  - imdata: numpy array, the image data to write.
  - feature_descriptions: list of str, descriptions for each feature.
  - geoTransform: tuple, geotransformation parameters.
  - proj: str, projection information.
  - drv_name: str, driver name (file type).
  """
  # Get the image Driver by its short name
  driver = gdal.GetDriverByName(drv_name)

  # Get image dimensions from array
  image_shape = np.shape(imdata)
  if len(image_shape) == 3:  # multiband
      [rows, cols, bands] = image_shape
  elif len(image_shape) == 2:  # singleband
      [rows, cols] = image_shape
      bands = 1

  # Image datatype
  dt = imdata.dtype
  datatype = gdal.GetDataTypeByName(dt.name)

  # Prepare the output dataset
  if datatype == 0:
      # Unknown datatype, try to use uint8 code
      datatype = 1

  # Create Output file
  outDataset = driver.Create(filename, cols, rows, bands, datatype)

  # Set the Georeference first
  outDataset.SetGeoTransform(geoTransform)
  outDataset.SetProjection(proj)

  # Write image data
  if bands == 1:
      outBand = outDataset.GetRasterBand(1)
      outBand.WriteArray(imdata)
  else:
      for band_index in range(0, bands):
          band_number = band_index + 1
          outBand = outDataset.GetRasterBand(band_number)
          outBand.SetDescription(feature_descriptions[band_index])
          outBand.WriteArray(imdata[:, :, band_index])

  # Clear variables and close the file
  outBand = None
  outDataset = None

def create_descriptions(band_descriptions, stats_descriptions):
  """
  Create descriptions for temporal features based on band and statistics descriptions.

  Parameters:
  - band_descriptions: list of str, descriptions of the bands.
  - stats_descriptions: list of str, descriptions of the statistics.

  Returns:
  - features_per_date: int, number of features per date.
  - temporal_descriptions: list of str, descriptions for temporal features.
  """
  # Get feature info from Descriptions
  features = [desc.split(' ')[0] for desc in band_descriptions]

  # Convert to set to get unique values
  # Count unique values with len to get featuresPerDate
  features_per_date = len(set(features))

  # Get unique band descriptions
  unique_descriptions = features[0:features_per_date]

  # Create temporal band Descriptions
  temporal_descriptions = []
  for feature_desc in unique_descriptions:
      for stat_desc in stats_descriptions:
          temporal_descriptions.append(stat_desc + ' ' + feature_desc)

  return features_per_date, temporal_descriptions

def tile_callback(option, opt, value, parser):
  """
  Callback function to handle comma-separated list options.
  """
  setattr(parser.values, option.dest, value.split(','))

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
  parser.add_option("-o", dest="output_path", action="store", type="string", help="Output path for this experiment's results. Will be created if it does not exist. Features will be saved in 'features' dir.", default=None)
  parser.add_option("-d", dest="timeseries_features_description", action="callback", type="string", help="Optional. Band descriptions subseting data in input path matching input pattern. If multiple separate by comma. Default value: B2,B3,B4,B5,B8,B11,B12,NDVI,NDWI,NDBI ", default=['B2','B3','B4','B5','B8','B11','B12','NDVI','NDWI','NDBI'], callback=tile_callback)
  parser.add_option("-s", dest="stats_descriptions", action="callback", type="string", help="Optional. Band descriptions subseting data in input path matching input pattern. If multiple separate by comma. Default value: min,10%,25%,median,75%,90%,max,mean,std", default=['min', '10%', '25%', 'median', '75%', '90%', 'max', 'mean', 'std'], callback=tile_callback)
  parser.add_option("-i", dest="input_timeseries_feature_name", action="store", type="string", help="Optional. Part of the input .npy name, relevant to timeseries features. Default value: ts ", default='ts')
  parser.add_option("-f", dest="temporal_feature_name", action="store", type="string", help="Optional. Part of the output .npy name, relevant to temporal features. Default value: tf ", default='tf')
  
  (options, args) = parser.parse_args()
  # Checking required arguments for the script
  parser.check_required("-t")
  parser.check_required("-o")

tile_name = options.tile_name
output_path = options.output_path
stats_descriptions = options.stats_descriptions
timeseries_features_description = options.timeseries_features_description
input_timeseries_feature_name = options.input_timeseries_feature_name
temporal_feature_name = options.temporal_feature_name
temporal_feature_count = len(stats_descriptions)
input_features_per_date, temporal_descriptions = create_descriptions(timeseries_features_description, stats_descriptions)

print('Calculating the following temporal features:')
print(temporal_descriptions)

# Load Feature Tables
features_path = os.path.join(output_path, 'features')
os.chdir(features_path)
try:
  train_FT = np.load(tile_name + '_' + input_timeseries_feature_name + '_train_FT.npy')
  test_FT = np.load(tile_name + '_' + input_timeseries_feature_name + '_test_FT.npy')
except Exception as e:
  print(e)
  print('Could not load timeseries features. Please make sure they are located in:')
  print(features_path)
  print('They should follow the naming conventions:')
  print('tile_name+\'_\'+input_timeseries_feature_name+\'_train_FT.npy\' ')
  print('tile_name+\'_\'+input_timeseries_feature_name+\'_test_FT.npy\' ')
  sys.exit(-1)

# Calculate temporal Feature Tables
temporal_train_FT, extra_features = temporal_table(train_FT, input_features_per_date, temporal_feature_count)
temporal_test_FT, _ = temporal_table(test_FT, input_features_per_date, temporal_feature_count)
print(extra_features, 'extra feature in this dataset')
if extra_features > 0:  # if extra features exist, stack them at end
  temporal_train_FT = np.hstack((temporal_train_FT, train_FT[:, -extra_features:].reshape(-1, extra_features)))
  temporal_test_FT = np.hstack((temporal_test_FT, test_FT[:, -extra_features:].reshape(-1, extra_features)))
  
# Save temporal Feature Tables
np.save(tile_name + '_' + temporal_feature_name + '_train_FT.npy', temporal_train_FT)
np.save(tile_name + '_' + temporal_feature_name + '_test_FT.npy', temporal_test_FT)
print('DONE')
