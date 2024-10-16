# supported by BiCubes project (HFRI grant 3943)

"""
This script is designed to create sharpened images as part of the Analysis Ready Data (ARD)
production pipeline, specifically for Step 3 of 4. It processes satellite imagery to perform
pansharpening using high-pass filter fusion and generates output images with enhanced spatial
resolution.

Usage:
  python script.py --output OUTPUT_PATH --tile TILE_NAME --swm SWM_DIR_PATH --wm WM_DIR_PATH

Arguments:
  --output: Path to write outputs.
  --tile: Tile name, e.g., 34SEJ.
  --swm: Optional. Path to seawater masks. Default is '/mnt/mapping-greece/experiments/SeaWaterMasks/SeaWater_Masks'.
  --wm: Optional. Path to water masks. Default is '/mnt/mapping-greece/experiments/WaterMasks/Water_Masks'.
"""

import argparse
import os
import sys

from glob import glob
from tqdm import tqdm

import cv2
import natsort
import numpy as np

from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from scipy import signal

def geoimread(filename):
  """
  Reads a georeferenced image file.

  Parameters:
  filename (str): The path to the image file.

  Returns:
  tuple: Image data, geotransform, projection, and driver name.
  """
  dataset = gdal.Open(filename, GA_ReadOnly)
  imdata = dataset.ReadAsArray()
  drv_name = dataset.GetDriver().ShortName
  geoTransform = dataset.GetGeoTransform()
  proj = dataset.GetProjection()
  dataset = None

  if imdata.ndim > 2:
      imdata = np.moveaxis(imdata, 0, 2)

  return imdata, geoTransform, proj, drv_name

def geoimwrite_descriptions(filename, imdata, descriptions, geoTransform, proj, drv_name):
  """
  Writes a georeferenced image file with band descriptions.

  Parameters:
  filename (str): The path to the output image file.
  imdata (ndarray): Image data to write.
  descriptions (list): Descriptions for each band.
  geoTransform (tuple): Geotransform information.
  proj (str): Projection information.
  drv_name (str): Driver name for the output file.
  """
  driver = gdal.GetDriverByName(drv_name)
  cols = np.size(imdata, 1)
  rows = np.size(imdata, 0)
  dims = np.shape(imdata.shape)[0]
  if dims == 2:
      bands = 1
  elif dims == 3:
      bands = np.size(imdata, 2)
  else:
      raise Exception('Error x01: Invalid image dimensions.')
  dt = imdata.dtype
  datatype = gdal.GetDataTypeByName(dt.name)
  if datatype == 0:
      datatype = 1
  outDataset = driver.Create(filename, cols, rows, bands, datatype)
  outDataset.SetGeoTransform(geoTransform)
  outDataset.SetProjection(proj)
  if bands == 1:
      outBand = outDataset.GetRasterBand(1)
      outBand.WriteArray(imdata, 0, 0)
  else:
      for b in range(bands):
          band_index = b + 1
          outBand = outDataset.GetRasterBand(band_index)
          outBand.SetDescription(descriptions[b])
          outBand.WriteArray(imdata[:, :, b], 0, 0)
  outBand = None
  outDataset = None

def hpfc_fusion(L, H, ratio, water_array):
  """
  Performs high-pass filter fusion for pansharpening.

  Parameters:
  L (ndarray): Low-resolution multispectral image.
  H (ndarray): High-resolution panchromatic image.
  ratio (float): Resolution ratio between L and H.
  water_array (ndarray): Water mask array.

  Returns:
  ndarray: Fused image.
  """
  Hrows, Hcols = H.shape
  Lrows, Lcols, bands = L.shape
  R = ratio
  
  Hstrict = np.where(water_array == True, np.nan, H)
  
  # HPF algorithm
  if (1 < R) and (R < 2.5):
      hpk = 5
      dv = 24
      M = 0.25
  elif (2.5 <= R) and (R < 3.5):
      hpk = 7
      dv = 48
      M = 0.5
  elif (3.5 <= R) and (R < 5.5):
      hpk = 9
      dv = 80
      M = 0.5
  elif (5.5 <= R) and (R < 7.5):
      hpk = 11
      dv = 120
      M = 0.65
  elif (7.5 <= R) and (R < 9.5):
      hpk = 13
      dv = 168
      M = 1
  elif (R >= 9.5):
      hpk = 15
      dv = 336
      M = 1.35

  HPKW = (np.ones((hpk, hpk))) * (-1)
  rhpk = int(round(hpk / 2))
  HPKW[rhpk, rhpk] = HPKW[rhpk, rhpk] * (-1) * dv
  HPFI = signal.convolve2d(H, HPKW, mode='same', boundary='symm')
  HPFIstrict = signal.convolve2d(Hstrict, HPKW, mode='same', boundary='symm')

  W = np.zeros((bands, 1))
  Lstd = np.zeros((bands, 1))
  Lstrict = L.copy()
  for band in range(bands):
      Lstrict[:, :, band] = np.where(water_array == True, np.nan, L[:, :, band])
      Lstd[band] = np.nanstd(Lstrict[:, :, band], ddof=1)
      W[band] = ((Lstd[band]) / np.nanstd(HPFIstrict, ddof=1)) * M

  F = np.zeros((Hrows, Hcols, bands), dtype=np.single)
  L = np.single(L)

  for band in range(bands):
      F[:, :, band] = L[:, :, band] + (HPFI * W[band])
      Fstrict = np.where(water_array == True, np.nan, F[:, :, band])
      F[:, :, band] = F[:, :, band] - np.nanmean(Fstrict) \
                      * (Lstd[band] / np.nanstd(Fstrict)) \
                      + np.nanmean(Lstrict[:, :, band])

  return F

def pansharp_hpfc(pan_data, multi_paths, sea_array, water_array, ratio=2):
  """
  Performs pansharpening using high-pass filter fusion.

  Parameters:
  pan_data (ndarray): High-resolution panchromatic data.
  multi_paths (list): Paths to multispectral images.
  sea_array (ndarray): Sea mask array.
  water_array (ndarray): Water mask array.
  ratio (int): Resolution ratio. Default is 2.

  Returns:
  tuple: Two pansharpened images.
  """
  multi1, gtr, proj, drv = geoimread(multi_paths[0])
  multi2, gtr, proj, drv = geoimread(multi_paths[1])
  
  multi1 = multi1.astype(np.float32)
  multi2 = multi2.astype(np.float32)
  
  multi1 = np.where(sea_array == True, np.nan, multi1)
  multi2 = np.where(sea_array == True, np.nan, multi2)
  
  multi = np.dstack((multi1, multi2))
  multi = np.dstack((multi,) * 2)
  multi_sharp = hpfc_fusion(multi, pan_data, ratio, water_array)
  
  return multi_sharp[:, :, 0].astype(np.int16), multi_sharp[:, :, 1].astype(np.int16)

def pansharpening(tile_name, swm_dir_path, wm_dir_path, input_dir_name='intermediate_output', output_dir_name='final_output'):
  """
  Performs pansharpening on images for a given tile.

  Parameters:
  tile_name (str): Name of the tile.
  swm_dir_path (str): Path to seawater masks.
  wm_dir_path (str): Path to water masks.
  input_dir_name (str): Directory name for input images. Default is 'intermediate_output'.
  output_dir_name (str): Directory name for output images. Default is 'final_output'.
  """
  print('Pansharpening')
  input_path_name = os.path.join(TILE_PATH, input_dir_name)
  output_path_name = os.path.join(TILE_PATH, output_dir_name)
  if not os.path.exists(output_path_name):
      os.mkdir(output_path_name)
  print(input_path_name)
  os.chdir(input_path_name)
  if TILE_NAME != '35SLD':
      filenames = glob('*AROP.tif')
  else:
      filenames = glob('*.tif')
  unique_dates = set()
  unique_bands = set()
  for f in filenames:
      print(input_path_name)
      print(f)
      if 'base' not in f:
          parts = f.split('_')
          unique_dates.add(parts[1])
          if TILE_NAME != '35SLD':
              unique_bands.add(parts[2])
          else:
              unique_bands.add(parts[2].split('.')[0])
          if 'B2' in f:
              sample_B2 = f
  dates = natsort.natsorted(list(unique_dates))
  bands = natsort.natsorted(list(unique_bands))
  print('dates', dates)
  print('bands', bands)
  dataset = gdal.Open(os.path.join(input_path_name, sample_B2), GA_ReadOnly)
  cols = dataset.RasterXSize
  rows = dataset.RasterYSize
  num_bands = len(bands)

  for date in tqdm(dates):
      swm_names = os.listdir(swm_dir_path)
      for swm_name in swm_names:
          if tile_name in swm_name:
              tile_swm_name = swm_name
      try:
          tile_swm_path = os.path.join(swm_dir_path, tile_swm_name)
          sea_array, gtr, proj, drv = geoimread(tile_swm_path)
          sea_array = cv2.resize(sea_array, (rows, cols), interpolation=cv2.INTER_NEAREST)
      except Exception as e:
          sea_array = np.zeros((rows, cols), dtype=bool)
          print(e)
          print('No swm file for tile', tile_name, 'in', swm_dir_path)
          tiles_without_sea = ['34TEK', '34TEL', '34TDL', '34TGM', '35TMG']
          if tile_name in tiles_without_sea:
              print('Tile', tile_name, 'has no sea, using an array of zeros as mask.')
          else:
              sys.exit(1)

      wm_names = os.listdir(wm_dir_path)
      for wm_name in wm_names:
          if tile_name in wm_name:
              tile_wm_name = wm_name
      try:
          tile_wm_path = os.path.join(wm_dir_path, tile_wm_name)
          water_array, gtr, proj, drv = geoimread(tile_wm_path)
          water_array = cv2.resize(water_array, (rows, cols), interpolation=cv2.INTER_NEAREST)
      except Exception as e:
          water_array = np.zeros((rows, cols), dtype=bool)
          print(e)
          print('No wm file for tile', tile_name, 'in', wm_dir_path)
          tiles_without_water = ['35TMG']
          if tile_name in tiles_without_water:
              print('Tile', tile_name, 'has no water, using an array of zeros as mask.')
          else:
              sys.exit(1)

      if TILE_NAME != '35SLD':
          b_paths = [os.path.join(input_path_name, '_'.join([tile_name, date, b, 'AROP.tif'])) for b in bands]
      else:
          b_paths = [os.path.join(input_path_name, '_'.join([tile_name, date, b]) + '.tif') for b in bands]

      ps = np.zeros((rows, cols, num_bands), dtype=np.int16)
      ps[:, :, 0], gtr, proj, drv = geoimread(b_paths[0])
      ps[:, :, 1], gtr, proj, drv = geoimread(b_paths[1])
      ps[:, :, 2], gtr, proj, drv = geoimread(b_paths[2])
      ps[:, :, 6], gtr, proj, drv = geoimread(b_paths[6])
      ps[:, :, 8], gtr, proj, drv = geoimread(b_paths[8])
      ps[:, :, 9], gtr, proj, drv = geoimread(b_paths[9])

      ps[:, :, 3], ps[:, :, 4] = pansharp_hpfc(ps[:, :, 2], (b_paths[3], b_paths[4]), sea_array, water_array)
      ps[:, :, 5], ps[:, :, 7] = pansharp_hpfc(ps[:, :, 6], (b_paths[5], b_paths[7]), sea_array, water_array)

      ps[ps < 0] = 0

      try:
          sea_cube = np.repeat(sea_array[:, :, np.newaxis], num_bands, axis=2)
          ps = np.where(sea_cube == True, -10000, ps)
      except Exception as e:
          print(e)
          print('No swm file for tile', tile_name, 'in', swm_dir_path)
          tiles_without_sea = ['34TEK', '34TEL', '34TDL', '34TGM', '35TMG']
          if tile_name in tiles_without_sea:
              print('Tile', tile_name, 'has no sea, using an array of zeros as mask.')
          else:
              sys.exit(1)

      ps_path = os.path.join(output_path_name, tile_name + '_' + date + '.tif')
      geoimwrite_descriptions(ps_path, ps, bands, gtr, proj, 'GTiff')

code_folder = os.getcwd()

parser = argparse.ArgumentParser(description='Create sharpened images, Analysis Ready Data Step 3 of 4')

# Building parameters
parser.add_argument("--output", dest="output_path", help="Path to write outputs.", default=None)
parser.add_argument("--tile", dest="TILE_NAME", help="Tile name., e.g., 34SEJ ", default=None)
parser.add_argument("--swm", dest="swm_dir_path", help="Optional. Path to seawater masks. Default value: /mnt/mapping-greece/experiments/SeaWaterMasks/SeaWater_Masks", default='/mnt/mapping-greece/experiments/SeaWaterMasks/SeaWater_Masks')
parser.add_argument("--wm", dest="wm_dir_path", help="Optional. Path to water masks. Default value: /mnt/mapping-greece/experiments/WaterMasks/Water_Masks", default='/mnt/mapping-greece/experiments/WaterMasks/Water_Masks')
args = parser.parse_args()

output_path = args.output_path
swm_dir_path = args.swm_dir_path
wm_dir_path = args.wm_dir_path
TILE_NAME = args.TILE_NAME

if not os.path.exists(output_path):
  os.mkdir(output_path)
TILE_PATH = os.path.join(output_path, TILE_NAME)
if not os.path.exists(TILE_PATH):
  os.mkdir(TILE_PATH)

pansharpening(TILE_NAME, swm_dir_path, wm_dir_path, input_dir_name='intermediate_output', output_dir_name='final_output')
