# supported by BiCubes project (HFRI grant 3943)

"""
This script is designed to create registered images as part of the Analysis Ready Data (ARD)
production pipeline, specifically for Step 2 of 4. It processes satellite imagery to generate
registration parameter files and executes the AROP (Automated Registration and Ortho-rectification
Processing) using specified base images and parameters.

Usage:
  python script.py --output OUTPUT_PATH --tile TILE_NAME --reg REG_BASE_IMAGE_FULLPATH --ortho ORTHO_FOLDER

Arguments:
  --output: Path to write outputs.
  --tile: Tile name, e.g., 34SEJ.
  --reg: Path to xlsx containing registration image dates. Default is '/mnt/mapping-greece/data/Registration_Base_Images.xlsx'.
  --ortho: Path to folder containing 'ortho' executable and 'lndortho.cps_par.ini'. Default is '/mnt/mapping-greece/experiments/code'.
"""

import os
import re
import sys
import shutil
import argparse
import fileinput
import subprocess
import multiprocessing

from glob import glob

import natsort
import rasterio
import openpyxl

import numpy as np
import pandas as pd
import datetime

def metadata(path, name, verbose=False):
  """
  Reads raster file metadata.

  Parameters:
  path (str): The directory path where the raster file is located.
  name (str): The name of the raster file.
  verbose (bool): If True, prints detailed metadata information.

  Returns:
  tuple: Contains CRS, number of bands, upper left corner, pixel size, rows, columns, data types, data type codes, driver, and UTM zone.
  """
  dtype_fwd = {
      None: 0,  # GDT_Unknown
      'uint8': 1,  # GDT_Byte
      'ubyte': 1,  # GDT_Byte
      'uint16': 2,  # GDT_UInt16
      'int16': 3,  # GDT_Int16
      'uint32': 4,  # GDT_UInt32
      'int32': 5,  # GDT_Int32
      'float32': 6,  # GDT_Float32
      'float64': 7,  # GDT_Float64
      'complex_': 8,  # GDT_CInt16
      'complex_': 9,  # GDT_CInt32
      'complex64': 10,  # GDT_CFloat32
      'complex128': 11  # GDT_CFloat64
  }

  image = rasterio.open(os.path.join(path, name))
  crs = image.crs
  bands = image.count
  up_l_crn = image.transform * (0, 0)
  pixel_size = image.transform[0]
  rows = image.width
  cols = image.height
  dtps = image.dtypes
  dtp_code = [dtype_fwd[dtp] for dtp in dtps]
  
  if verbose:
      print(f'Data Type: {dtps} - {dtp_code}')
      print(f'CRS: {crs}')
      print(f'Bands: {bands}')
      print(f'Upper Left Corner: {up_l_crn}')
      print(f'Pixel size: {pixel_size}')
      print(f'Rows: {rows}')
      print(f'Columns: {cols}')
      print(f'Driver: {image.driver}')
      print(f'UTM Zone: {crs.wkt}')

  return crs, bands, up_l_crn, pixel_size, rows, cols, dtps, dtp_code, image.driver, crs.wkt

def replace_line(file, search, replace, verbose=False):
  """
  Replaces a line in a file.

  Parameters:
  file (str): The file to modify.
  search (str): The string to search for.
  replace (str): The string to replace with.
  verbose (bool): If True, prints the replacement operation.
  """
  if verbose:
      print(f'Replacing "{search}" with: {replace}')
  for line in fileinput.input(file, inplace=True):
      if search in line:
          name = line.split(" = ")[1]  # Split by equal sign
          line = line.replace(str(name), str(replace) + '\n')
      sys.stdout.writelines(line)

def initialize_txt(path, filename):
  """
  Creates an AROP parameter file.

  Parameters:
  path (str): The directory path where the file will be created.
  filename (str): The name of the file to create.
  """
  with open(f"{os.path.join(path, filename)}.txt", "w+") as f:
      f.write("PARAMETER_FILE\n")
      f.write("BASE_LANDSAT = LC08_base_band4.tif\n")
      f.write("BASE_FILE_TYPE = GEOTIFF\n")
      f.write("BASE_DATA_TYPE = 2\n")
      f.write("BASE_NSAMPLE = 10980\n")
      f.write("BASE_NLINE = 10980\n")
      f.write("BASE_PIXEL_SIZE = 30.0\n")
      f.write("BASE_UPPER_LEFT_CORNER = 600000.000, 4900020.000\n")
      f.write("BASE_UTM_ZONE = 31\n")
      f.write("BASE_DATUM = 12\n")
      f.write("WARP_NBANDS = 2\n")
      f.write("WARP_LANDSAT_BAND = LC08_warp_band1.tif LC08_warp_band2.tif\n")
      f.write("WARP_BAND_DATA_TYPE = 2 2\n")
      f.write("# use Landsat1-5, Landsat7, Landsat8, TERRA, CBERS1, CBERS2, AWIFS, HJ1, Sentinel2\n")
      f.write("BASE_SATELLITE = Landsat8\n")
      f.write("WARP_BASE_MATCH_BAND = LC08_warp_band4.tif\n")
      f.write("WARP_FILE_TYPE = GEOTIFF\n")
      f.write("# following four variables (-1) will be read from warp match band if it's in GeoTIFF format\n")
      f.write("WARP_NSAMPLE = 7671\n")
      f.write("WARP_NLINE = 7791\n")
      f.write("WARP_PIXEL_SIZE = 30.0\n")
      f.write("WARP_UPPER_LEFT_CORNER = 536985.000, 4899615.000\n")
      f.write("WARP_UTM_ZONE = 31\n")
      f.write("WARP_DATUM = 12\n")
      f.write("WARP_SATELLITE = Sentinel2\n")
      f.write("# Landsat orthorectied output images\n")
      f.write("OUT_PIXEL_SIZE = 30.0\n")
      f.write("# NN-nearest neighbor; BI-bilinear interpolation; CC-cubic convolution; AGG-aggregation; none-skip resampling\n")
      f.write("RESAMPLE_METHOD = CC\n")
      f.write("OUT_EXTENT = BASE\n")
      f.write("OUT_LANDSAT_BAND = AROP_LC08_warp_band1.tif AROP_LC08_warp_band2.tif\n")
      f.write("OUT_BASE_MATCH_BAND = AROP_LC08_warp_band5.tif\n")
      f.write("OUT_BASE_POLY_ORDER = 1\n")
      f.write("# define log file for control points and polynomial function\n")
      f.write("CP_LOG_FILE = cp_log.txt\n")
      f.write("# ancillary input for orthorectification process\n")
      f.write("CP_PARAMETERS_FILE = lndortho.cps_par.ini\n")
      f.write("END\n")

def create_txt(input_dirpath, base_image_path, band_descriptions, base_band_description='B4', resampling='NN'):
  """
  Creates parameter files for image registration.

  Parameters:
  input_dirpath (str): Directory containing images to be registered.
  base_image_path (str): Path to the base image.
  band_descriptions (list): List of band descriptions.
  base_band_description (str): Description of the base band. Default is 'B4'.
  resampling (str): Resampling method. Default is 'NN'.

  Returns:
  str: The path to the parameter file.
  """
  os.chdir(input_dirpath)  # Change to input directory
  vd_B4_paths = glob('*' + base_band_description + '*')  # Get paths for B4
  base_image_dirpath, base_image_filename = os.path.split(base_image_path)  # Split base image path and filename
  base_date = base_image_filename.split('_')[1][2:]  # Extract base date

  for file in vd_B4_paths:
      if 'AROP' not in file:
          parts = file.split('_')  # Split filename into parts
          date = parts[1][2:]  # Extract date
          if base_date not in date:  # Create txt if current date is not the base date
              name = '_'.join([parts[0], parts[1]])  # Get root of name (TILE and DATE)
              par_fullpath = os.path.join(input_dirpath, f'{name}.txt')  # Full path for the txt file

              # Initialize .txt
              initialize_txt(input_dirpath, name)

              # Change Base Parameters
              searchfile = "BASE_LANDSAT = "
              replace_line(par_fullpath, searchfile, base_image_filename)

              # Reading metadata
              crs, bands, up_l_crn, pixel_size, rows, cols, dtps, dtp_code, driver, utm = metadata(base_image_dirpath, base_image_filename)
              if driver == 'GTiff':
                  driver = 'GEOTIFF'
              replace_line(par_fullpath, "BASE_FILE_TYPE =", driver)
              replace_line(par_fullpath, "BASE_DATA_TYPE =", dtp_code[0])
              replace_line(par_fullpath, "BASE_NLINE =", rows)
              replace_line(par_fullpath, "BASE_NSAMPLE =", cols)
              replace_line(par_fullpath, "BASE_PIXEL_SIZE =", pixel_size)
              replace_line(par_fullpath, "BASE_UPPER_LEFT_CORNER =", f'{up_l_crn[0]}, {up_l_crn[1]}')
              utm_zone = re.sub("[^0-9]", "", utm.split(',')[0].split('zone ')[1].split('"')[0])
              replace_line(par_fullpath, 'BASE_UTM_ZONE = ', utm_zone)

              # Change image parameters
              crs, bands, up_l_crn, pixel_size, rows, cols, dtps, dtp_code, driver, utm = metadata(input_dirpath, file)

              replace_line(par_fullpath, 'WARP_NBANDS = ', len(band_descriptions))
              band_names = ' '.join(['_'.join([name, desc + '.tif']) for desc in band_descriptions])
              replace_line(par_fullpath, 'WARP_LANDSAT_BAND =', band_names)
              replace_line(par_fullpath, 'WARP_BAND_DATA_TYPE = ', ' '.join([str(dtp_code[0])] * len(band_descriptions)))
              replace_line(par_fullpath, 'BASE_SATELLITE = ', 'Sentinel2')
              replace_line(par_fullpath, 'WARP_BASE_MATCH_BAND = ', '_'.join([name, base_band_description + '.tif']))
              replace_line(par_fullpath, "WARP_FILE_TYPE = ", driver)
              replace_line(par_fullpath, "WARP_NLINE = ", rows)
              replace_line(par_fullpath, "WARP_NSAMPLE = ", cols)
              replace_line(par_fullpath, 'WARP_PIXEL_SIZE = ', pixel_size)
              replace_line(par_fullpath, "WARP_UPPER_LEFT_CORNER = ", f'{up_l_crn[0]}, {up_l_crn[1]}')
              utm_zone = re.sub("[^0-9]", "", utm.split(',')[0].split('zone ')[1].split('"')[0])
              replace_line(par_fullpath, 'WARP_UTM_ZONE = ', utm_zone)
              replace_line(par_fullpath, 'OUT_PIXEL_SIZE = ', pixel_size)

              if resampling not in ['NN', 'BI', 'CC', 'AGG', 'none']:
                  raise ValueError("Resampling options are: NN-nearest neighbor; BI-bilinear interpolation; CC-cubic convolution; AGG-aggregation; none-skip resampling")
              replace_line(par_fullpath, 'RESAMPLE_METHOD = ', resampling)

              out_band_names = ' '.join(['_'.join([name, desc, 'AROP.tif']) for desc in band_descriptions])
              replace_line(par_fullpath, 'OUT_LANDSAT_BAND =', out_band_names)
              out_match = '_'.join([name, base_band_description, 'AROP.tif'])
              replace_line(par_fullpath, 'OUT_BASE_MATCH_BAND = ', out_match)

  return par_fullpath

def arop_registration(base_date_obj, ortho_folder, dir_name='intermediate_output', band_descriptions=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'], base_band_description='B4', resampling='NN'):
  """
  Performs AROP registration for images.

  Parameters:
  base_date_obj (datetime): The base date for registration.
  ortho_folder (str): Path to the folder containing 'ortho' executable and 'lndortho.cps_par.ini'.
  dir_name (str): Directory name for intermediate output. Default is 'intermediate_output'.
  band_descriptions (list): List of band descriptions. Default includes common Sentinel-2 bands.
  base_band_description (str): Description of the base band. Default is 'B4'.
  resampling (str): Resampling method. Default is 'NN'.
  """
  path_name = os.path.join(TILE_PATH, dir_name)

  opath1 = os.path.join(ortho_folder, 'ortho_folder', 'ortho')
  opath1c = os.path.join(path_name, 'ortho')
  opath2 = os.path.join(ortho_folder, 'ortho_folder', 'lndortho.cps_par.ini')
  opath2c = os.path.join(path_name, 'lndortho.cps_par.ini')
  shutil.copy2(opath1, opath1c)
  shutil.copy2(opath2, opath2c)

  os.chmod(opath1c, 0o774)
  os.chmod(opath2c, 0o774)

  # Base image assignment
  base_date_str = base_date_obj.strftime('%Y%m%d')  # Base date string

  # Determine base image filename
  base_image_filename = '_'.join([TILE_NAME, 'VD' + base_date_str, base_band_description + '_AROP.tif'])
  base_image_path = os.path.join(path_name, base_image_filename)
  if not os.path.exists(base_image_path):
      base_image_filename = '_'.join([TILE_NAME, 'VD' + base_date_str, base_band_description + '_base.tif'])
      base_image_path = os.path.join(path_name, base_image_filename)

  create_txt(path_name, base_image_path, band_descriptions, base_band_description='B4', resampling='NN')

  # Registration parameter file
  os.chdir(path_name)
  completed = "completed_arop.txt"
  if not os.path.exists(os.path.join(path_name, completed)):
      with open(os.path.join(path_name, completed), "w") as f:
          f.close()

  with open(os.path.join(path_name, completed), "r+") as f:
      implemented = f.read().split("\n")
      rpfiles = glob(TILE_NAME + '*.txt')
      rpfiles2 = natsort.natsorted(rpfiles)

      for i in range(len(rpfiles2)):
          if rpfiles2[i] in implemented:
              print(f"Already done {rpfiles2[i]}")
          else:
              subprocess.call(opath1c + ' -r ' + rpfiles2[i], shell=True)
              with open(os.path.join(path_name, completed), 'a') as f:
                  f.write(rpfiles2[i] + "\n")

  print('')
  print('Registration finished successfully.')
  print('')

parser = argparse.ArgumentParser(description='Create registered images, Analysis Ready Data Step 2 of 4')

# Building parameters
parser.add_argument("--output", dest="output_path", help="Path to write outputs.", default=None)
parser.add_argument("--tile", dest="TILE_NAME", help="Tile name., e.g., 34SEJ ", default=None)
parser.add_argument("--reg", dest="reg_base_image_fullpath", help="Path to xlsx containing registration image dates. Default value: /mnt/mapping-greece/data/Registration_Base_Images.xlsx", default='/mnt/mapping-greece/data/Registration_Base_Images.xlsx')
parser.add_argument("--ortho", dest="ortho_folder", help="Path to folder containing 'ortho' executable and 'lndortho.cps_par.ini'. Default value: /mnt/mapping-greece/experiments/code", default='/mnt/mapping-greece/experiments/code')
args = parser.parse_args()

output_path = args.output_path
reg_base_image_fullpath = args.reg_base_image_fullpath
ortho_folder = args.ortho_folder
TILE_NAME = args.TILE_NAME

# Ensure output path exists
if not os.path.exists(output_path):
  os.mkdir(output_path)
TILE_PATH = os.path.join(output_path, TILE_NAME)

# Ensure tile path exists
if not os.path.exists(TILE_PATH):
  os.mkdir(TILE_PATH)

if TILE_NAME != '35SLD':
  # Get registration datetime from base image
  DATE_FORMAT = "%Y%m%d"
  os.chdir(reg_base_image_fullpath)
  base_image_name = glob('*' + TILE_NAME + '*')[0]
  base_date = base_image_name.split('_')[1][2:]
  base_date_obj = datetime.datetime.strptime(base_date, DATE_FORMAT)

  arop_registration(base_date_obj, ortho_folder, dir_name='intermediate_output')
else:
  print('Tile 35SLD is exempt from the AROP processing step. Not enough land.')
