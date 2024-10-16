# supported by BiCubes project (HFRI grant 3943)

"""
This script is designed to create virtual date images as part of the Analysis Ready Data (ARD)
production pipeline, specifically for Step 1 of 4. It processes satellite imagery to generate
synthetic images for specified dates using interpolation and handles duplicate date data.

Usage:
  python script.py --input INPUT_PATH --output OUTPUT_PATH --ir INPUT_DATE_RANGE --or OUTPUT_DATE_RANGE --int OUTPUT_INTERVAL --swm SWM_DIR_PATH --reg REG_BASE_IMAGE_FULLPATH

Arguments:
  --input: Path to input images.
  --output: Path to write outputs.
  --ir: Start date and end date for input data. Format: YYYYMMDD YYYYMMDD.
  --or: Start date and end date for output data. Format: YYYYMMDD YYYYMMDD.
  --int: Optional. Interval between consecutive output dates, in days. Default is 10.
  --swm: Optional. Path to seawater masks. Default is '/mnt/mapping-greece/experiments/SeaWaterMasks/SeaWater_Masks'.
  --reg: Optional. Path to xlsx containing registration image dates. Default is '/mnt/mapping-greece/data/Registration_Base_Images.xlsx'.
"""

import fnmatch
import multiprocessing
import argparse
import os
import shutil
import sys
import datetime

from ast import literal_eval
from glob import glob
from pathlib import Path
from distutils.dir_util import copy_tree
from functools import partial

from tqdm import tqdm
import cv2
import natsort
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import pandas as pd
import openpyxl

from geoimread import geoimread
from geoimwrite import geoimwrite

def get_band_nodata(tif_fullpaths, band_name_contents, nodata_value=-10000):
  """
  Load a file in tif_fullpaths matching band_name_contents and count instances of nodata.

  Parameters:
  tif_fullpaths (list): List of file paths to search.
  band_name_contents (str): Band name to search for in file paths.
  nodata_value (int): Value representing nodata. Default is -10000.

  Returns:
  tuple: Indices of nodata values and the area of nodata values.
  """
  for fullpath in tif_fullpaths:
      if band_name_contents in fullpath:
          band, gtr, proj, drv = geoimread(fullpath)
  nodata_indices = band == nodata_value
  nodata_area = nodata_indices.sum()

  return nodata_indices, nodata_area

def get_tif_files(directory, tif_name_contents):
  """
  Find all .tif full paths in directories matching tif_name_contents.

  Parameters:
  directory (str): Directory to search.
  tif_name_contents (list): List of contents to match in filenames.

  Returns:
  list: Sorted list of .tif file paths.
  """
  tif_fullpaths = []
  pattern = '*.tif'
  for path, dirs, files in os.walk(directory):
      for filename in fnmatch.filter(files, pattern):
          for contents in tif_name_contents:
              if contents in filename:
                  fullpath = os.path.join(path, filename)
                  tif_fullpaths.append(fullpath)
  tif_fullpaths = natsort.natsorted(tif_fullpaths)

  return tif_fullpaths

def merge_s2_duplicate_dates(dates_paths, nodata_value=-10000):
  """
  Merge duplicate date data for Sentinel-2 images.

  Parameters:
  dates_paths (list): List of tuples containing dates and paths.
  nodata_value (int): Value representing nodata. Default is -10000.
  """
  l2a_paths = [tup[1] for tup in dates_paths]
  print('\nReplace values and remove S2 duplicate dates\n--------------------------------------------\n')
  
  start_date_time = []
  for pathname in l2a_paths:
      dirname = os.path.split(pathname)[-1]
      part3 = dirname.split('_')[1].split('-')[0]
      start_date_time.append(part3)

  seen = {}
  duplicates = []
  for item in start_date_time:
      if item not in seen:
          seen[item] = 1
      else:
          if seen[item] == 1:
              duplicates.append(item)
          seen[item] += 1
  print(duplicates)

  for duplicate_date in duplicates:
      print('Merging data for ', duplicate_date)
      l2a_duplicate_paths = [pathname for pathname in l2a_paths if duplicate_date in pathname]

      for source in l2a_duplicate_paths:
          dirname = os.path.split(source)[-1]
          basedir = Path(source).parent
          mergeddir = os.path.join(basedir, 'merged_duplicate_date_components')
          if not os.path.exists(mergeddir):
              os.mkdir(mergeddir)
          target = os.path.join(mergeddir, dirname)
          copy_tree(source, target)
      print('Copied L2A', duplicate_date, ' directories to', mergeddir)

      tif_name_contents_R1 = ['B2', 'B3', 'B4', 'B8.tif', 'R1']
      tif_fullpaths_R1_0 = get_tif_files(l2a_duplicate_paths[0], tif_name_contents_R1)
      tif_fullpaths_R1_1 = get_tif_files(l2a_duplicate_paths[1], tif_name_contents_R1)
      tif_name_contents_R2 = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12', 'R2']
      tif_fullpaths_R2_0 = get_tif_files(l2a_duplicate_paths[0], tif_name_contents_R2)
      tif_fullpaths_R2_1 = get_tif_files(l2a_duplicate_paths[1], tif_name_contents_R2)

      tif_count_R1 = len(tif_fullpaths_R1_0)
      if tif_count_R1 != len(tif_fullpaths_R1_1):
          print('Different number of 10m tif files between:', l2a_duplicate_paths)
          break
      tif_count_R2 = len(tif_fullpaths_R2_0)
      if tif_count_R2 != len(tif_fullpaths_R2_1):
          print('Different number of 20m tif files between:', l2a_duplicate_paths)
          break

      nodata_indices_R1_0, nodata_area_R1_0 = get_band_nodata(tif_fullpaths_R1_0, 'FRE_B4', nodata_value)
      nodata_indices_R1_1, nodata_area_R1_1 = get_band_nodata(tif_fullpaths_R1_1, 'FRE_B4', nodata_value)
      nodata_indices_R2_0, nodata_area_R2_0 = get_band_nodata(tif_fullpaths_R2_0, 'FRE_B8A', nodata_value)
      nodata_indices_R2_1, nodata_area_R2_1 = get_band_nodata(tif_fullpaths_R2_1, 'FRE_B8A', nodata_value)

      print('\nReplacing all .tif files...\n')
      if nodata_area_R1_0 <= nodata_area_R1_1:
          for f in range(tif_count_R1):
              img0, gtr0, proj0, drv0 = geoimread(tif_fullpaths_R1_0[f])
              img1, gtr1, proj1, drv1 = geoimread(tif_fullpaths_R1_1[f])
              img0[nodata_indices_R1_0] = img1[nodata_indices_R1_0]
              geoimwrite(tif_fullpaths_R1_0[f], img0, gtr0, proj0, drv0)
          for f in range(tif_count_R2):
              img0, gtr0, proj0, drv0 = geoimread(tif_fullpaths_R2_0[f])
              img1, gtr1, proj1, drv1 = geoimread(tif_fullpaths_R2_1[f])
              img0[nodata_indices_R2_0] = img1[nodata_indices_R2_0]
              geoimwrite(tif_fullpaths_R2_0[f], img0, gtr0, proj0, drv0)
          shutil.rmtree(l2a_duplicate_paths[1])
          print('\nRemoved directory:')
          print(l2a_duplicate_paths[1])
          print('')
      else:
          for f in range(tif_count_R1):
              img0, gtr0, proj0, drv0 = geoimread(tif_fullpaths_R1_0[f])
              img1, gtr1, proj1, drv1 = geoimread(tif_fullpaths_R1_1[f])
              img1[nodata_indices_R1_1] = img0[nodata_indices_R1_1]
              geoimwrite(tif_fullpaths_R1_1[f], img1, gtr1, proj1, drv1)
          for f in range(tif_count_R2):
              img0, gtr0, proj0, drv0 = geoimread(tif_fullpaths_R2_0[f])
              img1, gtr1, proj1, drv1 = geoimread(tif_fullpaths_R2_1[f])
              img1[nodata_indices_R2_1] = img0[nodata_indices_R2_1]
              geoimwrite(tif_fullpaths_R2_1[f], img1, gtr1, proj1, drv1)
          shutil.rmtree(l2a_duplicate_paths[0])
          print('\nRemoved directory:')
          print(l2a_duplicate_paths[0])
          print('')

def subset_band_paths(band_paths, band_name_contents):
  """
  Subset band paths based on band name contents.

  Parameters:
  band_paths (list): List of band paths.
  band_name_contents (list): List of band name contents to match.

  Returns:
  list: Subset of band paths.
  """
  band_paths_subset = []
  for band_path in band_paths:
      band_filename = os.path.basename(band_path)
      for current_band_name in band_name_contents:
          if current_band_name in band_filename:
              band_paths_subset.append(band_path)

  return band_paths_subset

def check_dates(dates, paths):
  """
  Check for missing dates in paths.

  Parameters:
  dates (list): List of dates.
  paths (list): List of paths.

  Returns:
  list: Missing dates.
  """
  dates_in_paths = []
  for path in paths:
      filename = os.path.basename(path)
      date = filename.split('_')[1].split('-')[0]
      dates_in_paths.append(date)
  missing_dates = np.setdiff1d(dates, dates_in_paths)

  return missing_dates.tolist()

def get_band_and_cloudmask_paths(dates, l2a_paths):
  """
  Get band and cloud mask paths.

  Parameters:
  dates (list): List of dates.
  l2a_paths (list): List of L2A paths.

  Returns:
  tuple: Band paths, R1 cloud mask paths, R2 cloud mask paths, and unique dates missing data.
  """
  band_paths = []
  clm_R1_paths = []
  clm_R2_paths = []
  unique_dates_missing_data = []
  for l2a_path in l2a_paths:
      if os.path.exists(l2a_path):
          l2a_path_contents = os.listdir(l2a_path)
          l2a_path_contents = natsort.natsorted(l2a_path_contents)
          for cont in l2a_path_contents:
              if 'FRE' in cont:
                  band_path = os.path.join(l2a_path, cont)
                  band_paths.append(band_path)
              if 'MASKS' in cont:
                  mask_dir_path = os.path.join(l2a_path, cont)
                  mask_dir_contents = natsort.natsorted(os.listdir(mask_dir_path))
                  for mask_name in mask_dir_contents:
                      if 'CLM_R1.tif' in mask_name:
                          mask_path = os.path.join(mask_dir_path, mask_name)
                          clm_R1_paths.append(mask_path)
                      if 'CLM_R2.tif' in mask_name:
                          mask_path = os.path.join(mask_dir_path, mask_name)
                          clm_R2_paths.append(mask_path)

  missing_R1 = check_dates(dates, clm_R1_paths)
  missing_R2 = check_dates(dates, clm_R2_paths)
  missing_bands = []
  mia_check = []
  bnames = [['B2'], ['B3'], ['B4'], ['B5'], ['B6'], ['B7'], ['B8.tif'], ['B8A'], ['B11'], ['B12']]
  for bname in bnames:
      bpaths = subset_band_paths(band_paths, bname)
      mia = check_dates(dates, bpaths)
      mia_check.extend(mia)
      missing_bands.append((bname[0], mia))
  if len(missing_R1) + len(missing_R2) + len(mia_check) > 0:
      print('Missing R1 cloudmasks for dates:', missing_R1)
      print('Missing R2 cloudmasks for dates:', missing_R2)
      print('Missing bands for dates:')
      print(np.matrix(missing_bands, dtype=object))
      unique_dates_missing_data = set(missing_R1 + missing_R2 + mia_check)
      print('Data from dates', unique_dates_missing_data, 'will not be included in the pipeline.')
      for date_missing_data in unique_dates_missing_data:
          band_paths = [b for b in band_paths if date_missing_data not in b]
          clm_R1_paths = [c for c in clm_R1_paths if date_missing_data not in c]
          clm_R2_paths = [d for d in clm_R2_paths if date_missing_data not in d]

  band_paths_clear = [band_path for band_path in band_paths if band_path.split('.')[-1] == 'tif']
  clm_R1_paths_clear = [clm_R1_path for clm_R1_path in clm_R1_paths if clm_R1_path.split('.')[-1] == 'tif']
  clm_R2_paths_clear = [clm_R2_path for clm_R2_path in clm_R2_paths if clm_R2_path.split('.')[-1] == 'tif']

  return band_paths_clear, clm_R1_paths_clear, clm_R2_paths_clear, list(unique_dates_missing_data)

def pixel_polyfit10(data, t, st, num_bands):
  """
  Perform polynomial fitting for 10m resolution data.

  Parameters:
  data (tuple): Tuple containing y and w values for a pixel.
  t (ndarray): Timespace array.
  st (ndarray): Synthetic timespace array.
  num_bands (int): Number of bands.

  Returns:
  ndarray: Synthetic pixel values.
  """
  y0 = data[0]
  w0 = data[1]
  synth_pixel = np.full((st.size, num_bands), 0, dtype=np.uint16)
  if np.sum(w0[:, 2]) > 2:
      valid_observations = w0[:, 2] == 1
      vo = np.repeat(valid_observations[:, np.newaxis], num_bands, 1)
      y = y0[vo].reshape(-1, num_bands)
      x = t[vo[:, 0]].reshape(-1,)
      for b in range(num_bands):
          synth_pixel[:, b] = np.interp(x=st, xp=x, fp=y[:, b])

  return synth_pixel

def pixel_polyfit20(data, t, st, num_bands):
  """
  Perform polynomial fitting for 20m resolution data.

  Parameters:
  data (tuple): Tuple containing y and w values for a pixel.
  t (ndarray): Timespace array.
  st (ndarray): Synthetic timespace array.
  num_bands (int): Number of bands.

  Returns:
  ndarray: Synthetic pixel values.
  """
  yp = data[0]
  wp = data[1]
  synth_pixel = np.full((st.size, num_bands), 0, dtype=np.uint16)
  if np.sum(wp[:, 0]) > 2 * num_bands:
      valid_observations = wp[:, 0] == 1
      vo = np.repeat(valid_observations[:, np.newaxis], num_bands, 1)
      y = yp[vo].reshape(-1, num_bands)
      x = t[vo[:, 0]].reshape(-1,)
      for b in range(num_bands):
          synth_pixel[:, b] = np.interp(x=st, xp=x, fp=y[:, b])

  return synth_pixel

def calc_chunksize(n_workers, len_iterable, factor=4):
  """
  Calculate chunksize argument for Pool-methods.

  Parameters:
  n_workers (int): Number of workers.
  len_iterable (int): Length of the iterable.
  factor (int): Factor for calculation. Default is 4.

  Returns:
  int: Calculated chunksize.
  """
  chunk_size, extra = divmod(len_iterable, n_workers * factor)
  if extra:
      chunk_size += 1
  return chunk_size

def interp_clear_pixels(band_paths, clm_R1_paths, clm_R2_paths, swm_dir_path, timespace, synth_timespace, write_path, input_date_start_obj, NODATA_VALUE=-10000):
  """
  Interpolate clear pixels to create synthetic images.

  Parameters:
  band_paths (list): List of band paths.
  clm_R1_paths (list): List of R1 cloud mask paths.
  clm_R2_paths (list): List of R2 cloud mask paths.
  swm_dir_path (str): Path to seawater masks.
  timespace (list): List of timespace values.
  synth_timespace (list): List of synthetic timespace values.
  write_path (str): Path to write outputs.
  input_date_start_obj (datetime): Start date object for input data.
  NODATA_VALUE (int): Value representing nodata. Default is -10000.
  """
  cpu_count_minus = 1
  agents = multiprocessing.cpu_count() - cpu_count_minus

  print("Number of agents is: ", agents)
  with multiprocessing.Pool(processes=agents) as pool:

      num_dates = len(timespace)
      t_arr = np.asarray(timespace)
      st_arr = np.asarray(synth_timespace)
      parts = 18
      
      band_name_contents_R1 = ['B2', 'B3', 'B4', 'B8.tif']
      band_paths_subset_R1 = subset_band_paths(band_paths, band_name_contents_R1)
      band_name_contents_R2 = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
      band_paths_subset_R2 = subset_band_paths(band_paths, band_name_contents_R2)
      
      dataset_R1 = gdal.Open(band_paths_subset_R1[0], GA_ReadOnly)
      cols_R1 = dataset_R1.RasterXSize
      rows_R1 = dataset_R1.RasterYSize
      num_bands_per_date_R1 = len(band_name_contents_R1)        
      dataset_R2 = gdal.Open(band_paths_subset_R2[0], GA_ReadOnly)
      cols_R2 = dataset_R2.RasterXSize
      rows_R2 = dataset_R2.RasterYSize
      num_bands_per_date_R2 = len(band_name_contents_R2)
      
      imagepart_R1_dir_path = os.path.join(write_path, 'image_parts_R1')
      if not os.path.exists(imagepart_R1_dir_path):
          os.mkdir(imagepart_R1_dir_path)
      imagepart_R2_dir_path = os.path.join(write_path, 'image_parts_R2')
      if not os.path.exists(imagepart_R2_dir_path):
          os.mkdir(imagepart_R2_dir_path)             

      example_band_name = os.path.basename(os.path.normpath(band_paths_subset_R1[0]))
      tile_name = example_band_name.split('_')[3]
      tile_name = tile_name[1:]
      swm_names = os.listdir(swm_dir_path)
      for swm_name in swm_names:
          if tile_name in swm_name:
              tile_swm_name = swm_name
      try:
          tile_swm_path = os.path.join(swm_dir_path, tile_swm_name)
          sea_array, gtr, proj, drv = geoimread(tile_swm_path)
          sea_array = cv2.resize(sea_array, (rows_R1, cols_R1), interpolation=cv2.INTER_NEAREST)
          sea_row_R1 = (sea_array == 1).reshape(rows_R1 * cols_R1)
          sea_array = cv2.resize(sea_array, (rows_R2, cols_R2), interpolation=cv2.INTER_NEAREST)
          sea_row_R2 = (sea_array == 1).reshape(rows_R2 * cols_R2)
      except Exception as e:
          print(e)
          print('No swm file for tile', tile_name, 'in', swm_dir_path)
          tiles_without_sea = ['34TEK', '34TEL', '34TDL', '34TGM', '35TMG']
          if tile_name in tiles_without_sea:
              print('Tile', tile_name, 'has no sea, using an array of zeros as mask.')
              sea_row_R1 = np.zeros((rows_R1 * cols_R1,), dtype=bool)
              sea_row_R2 = np.zeros((rows_R2 * cols_R2,), dtype=bool)
          else:
              sys.exit(1)
      
      _, gtr_R1, proj, drv = geoimread(clm_R1_paths[0])
      _, gtr_R2, proj, drv = geoimread(clm_R2_paths[0])
      
      row_step_R1 = rows_R1 // parts
      row_residual_R1 = rows_R1 - (row_step_R1 * parts)     
      row_step_R2 = rows_R2 // parts
      row_residual_R2 = rows_R2 - (row_step_R2 * parts)

      print('Creating timeseries')
      
      for i in tqdm(range(0, parts)):

          print('This is process: ', i)
          sys.stdout.flush()
          p1_R1 = cols_R1 * row_step_R1 * i
          p1_R2 = cols_R2 * row_step_R2 * i
          if i != parts - 1:
              p2_R1 = cols_R1 * row_step_R1 * (i + 1)
              p2_R2 = cols_R2 * row_step_R2 * (i + 1)
          else:
              p2_R1 = cols_R1 * rows_R1
              p2_R2 = cols_R2 * rows_R2
          pixel_count_R1 = p2_R1 - p1_R1
          pixel_count_R2 = p2_R2 - p1_R2

          if i != parts - 1:
              rows_part_R1 = row_step_R1
              rows_part_R2 = row_step_R2
          else:
              rows_part_R1 = row_step_R1 + row_residual_R1
              rows_part_R2 = row_step_R2 + row_residual_R2
          
          timeseries_R1 = np.zeros((num_dates, pixel_count_R1, num_bands_per_date_R1), dtype=np.uint16)
          synth_timeseries_R1 = np.zeros((pixel_count_R1, st_arr.size, num_bands_per_date_R1), dtype=np.uint16)
          weights_R1 = np.ones((num_dates, pixel_count_R1, num_bands_per_date_R1), dtype=bool)
          
          for d in range(num_dates):
              clm_object = gdal.Open(clm_R1_paths[d], gdal.GA_ReadOnly)
              if i != parts - 1:
                  clm_part = clm_object.ReadAsArray(0, row_step_R1 * i, cols_R1, row_step_R1)
              else:
                  clm_part = clm_object.ReadAsArray(0, row_step_R1 * i, cols_R1, row_step_R1 + row_residual_R1)
              clm_row = (clm_part > 0).reshape(pixel_count_R1)
              d1_R1 = num_bands_per_date_R1 * d
              d2_R1 = num_bands_per_date_R1 * (d + 1)
              b = 0
              for b, band_path in enumerate(band_paths_subset_R1[d1_R1:d2_R1]):
                  dataset_object = gdal.Open(band_path, gdal.GA_ReadOnly)
                  band_object = dataset_object.GetRasterBand(1)
                  if i != parts - 1:
                      band_part = band_object.ReadAsArray(0, row_step_R1 * i, cols_R1, row_step_R1)
                  else:
                      band_part = band_object.ReadAsArray(0, row_step_R1 * i, cols_R1, row_step_R1 + row_residual_R1)
                  dataset_object = None
                  band_row = band_part.reshape(pixel_count_R1)
                  del band_part
                  if b == 0:
                      weights_R1[d, :, b] = np.where(clm_row == True, False, weights_R1[d, :, b])
                      weights_R1[d, :, b] = np.where(band_row == NODATA_VALUE, False, weights_R1[d, :, b])
                      sea_mask_R1 = np.where(sea_row_R1[p1_R1:p2_R1] == True, True, False)
                  else:
                      weights_R1[d, :, b] = weights_R1[d, :, 0]
                  timeseries_R1[d, :, b] = band_row
                  del band_row
              del clm_row
          sea_list_R1 = [sea_mask_R1[p] for p in range(pixel_count_R1)]
          sea_count_R1 = 0
          t_data = []
          w_data = []
          for p, values in enumerate(sea_list_R1):
              if sea_list_R1[p] == False:
                  t_data.append(timeseries_R1[:, p, :])
                  w_data.append(weights_R1[:, p, :])
              else:
                  sea_count_R1 += 1

          if sea_count_R1 > 0:
              print(sea_count_R1, 'Sea pixels in this part of the image')
              sys.stdout.flush()

          del timeseries_R1, weights_R1

          chunk_size = calc_chunksize(n_workers=agents, len_iterable=pixel_count_R1 - sea_count_R1, factor=4)
          fn = partial(pixel_polyfit10, t=t_arr, st=st_arr, num_bands=num_bands_per_date_R1)

          if sea_count_R1 == 0:
              data = zip(t_data, w_data)
              output = list(pool.imap(fn, data, chunksize=chunk_size))
              print("All Land in process ", i)
              synth_timeseries_R1 = np.array(output, dtype=np.uint16)
              del data, output
          elif sea_count_R1 == pixel_count_R1:
              print("All Sea in process ", i)
              synth_timeseries_R1 = np.full((pixel_count_R1, st_arr.size, num_bands_per_date_R1), 0, dtype=np.uint16)
          else:
              print("Mixed Sea and Land in process ", i)
              sys.stdout.flush()
              mixed_output = []
              data = zip(t_data, w_data)
              output = list(pool.imap(fn, data, chunksize=chunk_size))
              c = 0
              sea_pixel = np.full((st_arr.size, num_bands_per_date_R1), 0, dtype=np.uint16)
              for p, value in enumerate(sea_list_R1):
                  if value == False:
                      mixed_output.append(output[c])
                      c += 1
                  else:
                      mixed_output.append(sea_pixel)
              synth_timeseries_R1 = np.array(mixed_output, dtype=np.uint16)
              del data, output, mixed_output
          sys.stdout.flush()
          
          for b, current_band_name in enumerate(band_name_contents_R1):
              if '.' in current_band_name:
                  current_band_name = current_band_name.split('.')[0]
              for d, timepoint in enumerate(synth_timespace):
                  date = input_date_start_obj + datetime.timedelta(days=timepoint - 1)
                  date_str = date.strftime('%Y%m%d')
                  filename = '_'.join([TILE_NAME, 'VD' + date_str, current_band_name, 'part', str(i) + '.tif'])
                  band_path = os.path.join(imagepart_R1_dir_path, filename)
                  driver = gdal.GetDriverByName(drv)
                  outDataset = driver.Create(band_path, cols_R1, row_step_R1, 1, gdal.GDT_UInt16)
                  gtr_part = list(gtr_R1)
                  gtr_part[3] = gtr_part[3] + gtr_part[5] * i * row_step_R1
                  gtr_part = tuple(gtr_part)
                  outDataset.SetGeoTransform(gtr_part)
                  outDataset.SetProjection(proj)
                  outBand = outDataset.GetRasterBand(1)
                  outBand.WriteArray(synth_timeseries_R1[:, d, b].reshape(-1, cols_R1))
                  outDataset = None
          del synth_timeseries_R1
          
          timeseries_R2 = np.zeros((num_dates, pixel_count_R2, num_bands_per_date_R2), dtype=np.uint16)
          synth_timeseries_R2 = np.zeros((pixel_count_R2, st_arr.size, num_bands_per_date_R2), dtype=np.uint16)
          weights_R2 = np.ones((num_dates, pixel_count_R2, num_bands_per_date_R2), dtype=bool)
          sea_mask_R2 = np.zeros((pixel_count_R2), dtype=bool)
          for d in range(num_dates):
              clm_object = gdal.Open(clm_R2_paths[d], gdal.GA_ReadOnly)
              if i != parts - 1:
                  clm_part = clm_object.ReadAsArray(0, row_step_R2 * i, cols_R2, row_step_R2)
              else:
                  clm_part = clm_object.ReadAsArray(0, row_step_R2 * i, cols_R2, row_step_R2 + row_residual_R2)
              clm_row = (clm_part > 0).reshape(pixel_count_R2)
              d1_R2 = num_bands_per_date_R2 * d
              d2_R2 = num_bands_per_date_R2 * (d + 1)
              b = 0
              for b, band_path in enumerate(band_paths_subset_R2[d1_R2:d2_R2]):
                  dataset_object = gdal.Open(band_path, gdal.GA_ReadOnly)
                  band_object = dataset_object.GetRasterBand(1)
                  if i != parts - 1:
                      band_part = band_object.ReadAsArray(0, row_step_R2 * i, cols_R2, row_step_R2)
                  else:
                      band_part = band_object.ReadAsArray(0, row_step_R2 * i, cols_R2, row_step_R2 + row_residual_R2)
                  dataset_object = None
                  band_row = band_part.reshape(pixel_count_R2)
                  del band_part
                  if b == 0:
                      weights_R2[d, :, b] = np.where(clm_row == True, False, weights_R2[d, :, b])
                      weights_R2[d, :, b] = np.where(band_row == NODATA_VALUE, False, weights_R2[d, :, b])
                      sea_mask_R2 = np.where(sea_row_R2[p1_R2:p2_R2] == True, True, False)
                  else:
                      weights_R2[d, :, b] = weights_R2[d, :, 0]
                  timeseries_R2[d, :, b] = band_row
                  del band_row
              del clm_row
          sea_list_R2 = [sea_mask_R2[p] for p in range(pixel_count_R2)]
          sea_count_R2 = 0
          t_data = []
          w_data = []
          for p, values in enumerate(sea_list_R2):
              if sea_list_R2[p] == False:
                  t_data.append(timeseries_R2[:, p, :])
                  w_data.append(weights_R2[:, p, :])
              else:
                  sea_count_R2 += 1

          if sea_count_R2 > 0:
              print(sea_count_R2, 'Sea pixels in this part of the image')
          del timeseries_R2, weights_R2

          chunk_size = calc_chunksize(n_workers=agents, len_iterable=pixel_count_R2 - sea_count_R2, factor=4)
          fn = partial(pixel_polyfit20, t=t_arr, st=st_arr, num_bands=num_bands_per_date_R2)

          if sea_count_R2 == 0:
              data = zip(t_data, w_data)
              output = list(pool.imap(fn, data, chunksize=chunk_size))
              synth_timeseries_R2 = np.array(output, dtype=np.uint16)
              del data, output
          elif sea_count_R2 == pixel_count_R2:
              synth_timeseries_R2 = np.full((pixel_count_R2, st_arr.size, num_bands_per_date_R2), 0, dtype=np.uint16)
          else:
              mixed_output = []
              data = zip(t_data, w_data)
              output = list(pool.imap(fn, data, chunksize=chunk_size))
              c = 0
              sea_pixel = np.full((st_arr.size, num_bands_per_date_R2), 0, dtype=np.uint16)
              for p, values in enumerate(sea_list_R2):
                  if sea_list_R2[p] == False:
                      mixed_output.append(output[c])
                      c += 1
                  else:
                      mixed_output.append(sea_pixel)
              synth_timeseries_R2 = np.array(mixed_output, dtype=np.uint16)
              del data, output, mixed_output

          for b, current_band_name in enumerate(band_name_contents_R2):
              for d, timepoint in enumerate(synth_timespace):
                  date = input_date_start_obj + datetime.timedelta(days=timepoint - 1)
                  date_str = date.strftime('%Y%m%d')
                  filename = '_'.join([TILE_NAME, 'VD' + date_str, current_band_name, 'part', str(i) + '.tif'])
                  band_path = os.path.join(imagepart_R2_dir_path, filename)
                  driver = gdal.GetDriverByName(drv)
                  outDataset = driver.Create(band_path, cols_R2, row_step_R2, 1, gdal.GDT_UInt16)
                  gtr_part = list(gtr_R2)
                  gtr_part[3] = gtr_part[3] + gtr_part[5] * i * row_step_R2
                  gtr_part = tuple(gtr_part)
                  outDataset.SetGeoTransform(gtr_part)
                  outDataset.SetProjection(proj)
                  outBand = outDataset.GetRasterBand(1)
                  outBand.WriteArray(synth_timeseries_R2[:, d, b].reshape(-1, cols_R2))
                  outDataset = None
          del synth_timeseries_R2
          
      for b, current_band_name in enumerate(band_name_contents_R1):
          if '.' in current_band_name:
              current_band_name = current_band_name.split('.')[0]
          for d, timepoint in enumerate(synth_timespace):
              date = input_date_start_obj + datetime.timedelta(days=timepoint - 1)
              date_str = date.strftime('%Y%m%d')
              filename = '_'.join([TILE_NAME, 'VD' + date_str, current_band_name + '.tif'])
              band_path = os.path.join(write_path, filename)
              driver = gdal.GetDriverByName(drv)
              outDataset = driver.Create(band_path, cols_R1, rows_R1, 1, gdal.GDT_UInt16)
              outDataset.SetGeoTransform(gtr_R1)
              outDataset.SetProjection(proj)
              outBand = outDataset.GetRasterBand(1)
              os.chdir(imagepart_R1_dir_path)
              part_files = glob('*' + date_str + '_' + current_band_name + '_*')
              part_files = natsort.natsorted(part_files)  
              for p, part_filename in enumerate(part_files):
                  part_dataset = gdal.Open(part_filename)
                  image_part = part_dataset.ReadAsArray()
                  outBand.WriteArray(image_part, 0, row_step_R1 * p)                 
                  part_dataset = None
              outBand = None
              outDataset = None
      shutil.rmtree(imagepart_R1_dir_path)
          
      for b, current_band_name in enumerate(band_name_contents_R2):
          for d, timepoint in enumerate(synth_timespace):
              date = input_date_start_obj + datetime.timedelta(days=timepoint - 1)
              date_str = date.strftime('%Y%m%d')
              filename = '_'.join([TILE_NAME, 'VD' + date_str, current_band_name + '.tif'])
              band_path = os.path.join(write_path, filename)
              driver = gdal.GetDriverByName(drv)
              outDataset = driver.Create(band_path, cols_R1, rows_R1, 1, gdal.GDT_UInt16)
              outDataset.SetGeoTransform(gtr_R1)
              outDataset.SetProjection(proj)
              outBand = outDataset.GetRasterBand(1)
              os.chdir(imagepart_R2_dir_path)
              part_files = glob('*' + date_str + '_' + current_band_name + '_*')
              part_files = natsort.natsorted(part_files)  
              for p, part_filename in enumerate(part_files):
                  part_dataset = gdal.Open(part_filename)
                  image_part = part_dataset.ReadAsArray()
                  if p != len(part_files) - 1:
                      image_part = cv2.resize(image_part, (cols_R1, row_step_R1), interpolation=cv2.INTER_NEAREST)
                  else:
                      image_part = cv2.resize(image_part, (cols_R1, row_step_R1 + row_residual_R1), interpolation=cv2.INTER_NEAREST)
                  outBand.WriteArray(image_part, 0, row_step_R1 * p)              
                  part_dataset = None
              outBand = None
              outDataset = None   
      shutil.rmtree(imagepart_R2_dir_path)

def virtual_dates(dates_paths, input_date_range, output_date_range, output_interval, base_date_obj, swm_dir_path, write_dir_name='intermediate_output', NODATA_VALUE=-10000):
  """
  Create virtual date images.

  Parameters:
  dates_paths (list): List of tuples containing dates and paths.
  input_date_range (list): List containing start and end dates for input data.
  output_date_range (list): List containing start and end dates for output data.
  output_interval (int): Interval between consecutive output dates, in days.
  base_date_obj (datetime): Base date object.
  swm_dir_path (str): Path to seawater masks.
  write_dir_name (str): Directory name for writing outputs. Default is 'intermediate_output'.
  NODATA_VALUE (int): Value representing nodata. Default is -10000.
  """
  write_path_name = os.path.join(TILE_PATH, write_dir_name)
  if not os.path.exists(write_path_name):
      os.mkdir(write_path_name)

  dates = [x for x, _ in dates_paths]
  input_date_start, input_date_end = input_date_range
  output_date_start, output_date_end = output_date_range

  valid_dates = [date for date in dates if (date >= input_date_start) and (date <= input_date_end)]

  l2a_paths = [p for d, p in dates_paths if d in valid_dates]

  band_paths, clm_R1_paths, clm_R2_paths, unique_dates_missing_data = get_band_and_cloudmask_paths(valid_dates, l2a_paths)

  for udmd in unique_dates_missing_data:
      valid_dates = [d for d in valid_dates if udmd not in d]

  input_timespace = []
  input_date_start_obj = datetime.datetime.strptime(input_date_start, '%Y%m%d')
  for valid_date in valid_dates:
      date_obj = datetime.datetime.strptime(valid_date, '%Y%m%d')
      dt = (date_obj - input_date_start_obj).days + 1
      input_timespace.append(dt)

  output_date_start_obj = datetime.datetime.strptime(output_date_start, '%Y%m%d')
  output_date_end_obj = datetime.datetime.strptime(output_date_end, '%Y%m%d')
  start_diff = (output_date_start_obj - input_date_start_obj).days + 1
  output_range_in_days = (output_date_end_obj - output_date_start_obj).days
  interval_timespace = []
  i = start_diff
  while i <= output_range_in_days + start_diff:
      interval_timespace.append(i)
      i += int(output_interval)

  output_timespace = [e for e in interval_timespace]
  base_date_dt = (base_date_obj - input_date_start_obj).days + 1
  if base_date_dt < 0:
      src_file = os.path.join(REG_BASE_IMAGE_FULLPATH, BASE_IMAGE_NAME)
      shutil.copy2(src_file, write_path_name, follow_symlinks=True)
  elif base_date_dt not in output_timespace:
      output_timespace.append(base_date_dt)
      output_timespace = natsort.natsorted(output_timespace)

  if len(input_timespace) == len(valid_dates) == len(clm_R1_paths) == len(clm_R2_paths) == len(band_paths) / 10:
      print('Day 1 corresponds to the first element in input_date_range')
      print('Input timespace count is', len(input_timespace), 'and the corresponding values are:')
      print(input_timespace)
      print('Output timespace count is', len(output_timespace), 'and the corresponding values are:')
      print(output_timespace)
  else:
      print('Input timespace count is', len(input_timespace), 'and the corresponding values are:')
      print(input_timespace)
      print('Number of dates with data:', len(valid_dates))
      print('Number of R1 cloud masks:', len(clm_R1_paths))
      print('Number of R2 cloud masks:', len(clm_R2_paths))
      print('Number of bands:', len(band_paths))
      print('Please check data. Valid dates should have the same count as input timespace, R1 cloud masks, R2 cloud masks. Bands should be a multiple of the valid date count and the number of R1 & R2 bands per date.')
      sys.exit(1)

  interp_clear_pixels(band_paths, clm_R1_paths, clm_R2_paths, swm_dir_path, input_timespace, output_timespace, write_path_name, input_date_start_obj)

  base_date_str = base_date_obj.strftime('%Y%m%d')
  os.chdir(write_path_name)
  reg_bands = glob('*' + base_date_str + '*')
  for f in reg_bands:
      current_name = os.path.join(write_path_name, f)
      fn, ext = os.path.splitext(f)
      if fn.split('_')[-1] not in ['base', 'AROP']:
          if base_date_dt not in interval_timespace:
              rename_incl = '_base'
          else:
              rename_incl = '_AROP'
          new_name = os.path.join(write_path_name, fn + rename_incl + ext)
          os.rename(current_name, new_name)

def get_dates_paths(input_path):
  """
  Get dates and paths from input directory.

  Parameters:
  input_path (str): Path to input directory.

  Returns:
  tuple: Dates and paths, and tile name.
  """
  dir_names = os.listdir(input_path)
  l2a_dirs_all = [dir_name for dir_name in dir_names if '_L2A_' in dir_name]
  l2a_paths_all = [os.path.join(input_path, l2a_dir) for l2a_dir in l2a_dirs_all]
  l2a_paths_c = []
  l2a_dirs = []
  for l2a_path in l2a_paths_all:
      for File in os.listdir(l2a_path):
          if File.endswith(".tif"):
              l2a_paths_c.append(l2a_path)
              l2a_dirs.append(os.path.basename(os.path.normpath(l2a_path)))
              break

  dates = [l2a_dir.split('_')[1].split('-')[0] for l2a_dir in l2a_dirs]
  dates_paths = zip(dates, l2a_paths_c)
  dates_paths = natsort.natsorted(dates_paths)

  tile_name = l2a_dirs[0].split('_')[3]
  tile_name = tile_name[1:]

  return dates_paths, tile_name

code_folder = os.getcwd()

parser = argparse.ArgumentParser(description='Create virtual date images, Analysis Ready Data Step 1 of 4')

parser.add_argument("--input", dest="input_path", help="Path to input images.", default=None)
parser.add_argument("--output", dest="output_path", help="Path to write outputs.", default=None)
parser.add_argument("--ir", dest="input_date_range", nargs='+', help="Start date and end date for input data. Separate by space. format YYYYMMDD, e.g. from September 1st, 2018 to November 1st 2019: 20180901 20191101", default=None)
parser.add_argument("--or", dest="output_date_range", nargs='+', help="Start date and end date for output data. Separate by space. format YYYYMMDD, e.g. from October 1st, 2018 to October 1st 2019: 20181001 20191001", default=None)
parser.add_argument("--int", dest="output_interval", help="Optional. Interval between consecutive output dates, in days. Default value: 10", default=10)
parser.add_argument("--swm", dest="swm_dir_path", help="Optional. Path to seawater masks. Default value: /mnt/mapping-greece/experiments/SeaWaterMasks/SeaWater_Masks", default='/mnt/mapping-greece/experiments/SeaWaterMasks/SeaWater_Masks')
parser.add_argument("--reg", dest="reg_base_image_fullpath", help="Optional. Path to xlsx containing registration image dates. Default value: /mnt/mapping-greece/data/Registration_Base_Images.xlsx", default='/mnt/mapping-greece/data/Registration_Base_Images.xlsx')
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
swm_dir_path = args.swm_dir_path
input_date_range = args.input_date_range
output_date_range = args.output_date_range
output_interval = args.output_interval
REG_BASE_IMAGE_FULLPATH = args.reg_base_image_fullpath

dates_paths, TILE_NAME = get_dates_paths(input_path)
TILE_PATH = os.path.join(output_path, TILE_NAME)

if not os.path.exists(output_path):
  os.mkdir(output_path)
if not os.path.exists(TILE_PATH):
  os.mkdir(TILE_PATH)

DATE_FORMAT = "%Y%m%d"
os.chdir(REG_BASE_IMAGE_FULLPATH)
BASE_IMAGE_NAME = glob('*' + TILE_NAME + '*')[0]
base_date = BASE_IMAGE_NAME.split('_')[1][2:]
base_date_obj = datetime.datetime.strptime(base_date, DATE_FORMAT)

merge_s2_duplicate_dates(dates_paths)
dates_paths, _ = get_dates_paths(input_path)

virtual_dates(dates_paths, input_date_range, output_date_range, output_interval, base_date_obj, swm_dir_path, write_dir_name='intermediate_output')
