# Supported by BiCubes project (HFRI grant 3943)

import numpy as np
from osgeo import gdal

def geoimwrite(filename, imdata, geoTransform, proj, drv_name):
  """
  Writes image data to a georeferenced file.

  Parameters:
  filename (str): The path to the output file.
  imdata (numpy.ndarray): The image data to be written.
  geoTransform (tuple): Georeference transformation parameters.
  proj (str): Projection information.
  drv_name (str): The name of the driver to use for writing the file.

  Raises:
  Exception: If the image dimensions are invalid.
  """
  # Get the image driver by its short name
  driver = gdal.GetDriverByName(drv_name)
  
  # Get image dimensions from the array
  cols = np.size(imdata, 1)
  rows = np.size(imdata, 0)
  dims = np.shape(imdata.shape)[0]
  
  # Determine the number of bands based on dimensions
  if dims == 2:
      bands = 1
  elif dims == 3:
      bands = np.size(imdata, 2)
  else:
      raise Exception('Error x01: Invalid image dimensions.')
  
  # Determine the image data type
  dt = imdata.dtype
  datatype = gdal.GetDataTypeByName(dt.name)
  
  # Prepare the output dataset
  if datatype == 0:
      # Unknown datatype, try to use uint8 code
      datatype = 1
  
  outDataset = driver.Create(filename, cols, rows, bands, datatype)
  
  # Set the georeference information
  outDataset.SetGeoTransform(geoTransform)
  outDataset.SetProjection(proj)
  
  # Write the image data to the file
  if bands == 1:
      outBand = outDataset.GetRasterBand(1)
      outBand.WriteArray(imdata, 0, 0)
  else:
      # Iterate over the bands and write them to the file
      for b_idx in range(1, bands + 1):
          outBand = outDataset.GetRasterBand(b_idx)
          outBand.WriteArray(imdata[:, :, b_idx - 1], 0, 0)
  
  # Clear variables and close the file
  outBand = None
  outDataset = None

def geoimwrite_descriptions(filename, imdata, descriptions, geoTransform, proj, drv_name):
  """
  Writes image data to a georeferenced file with band descriptions.

  Parameters:
  filename (str): The path to the output file.
  imdata (numpy.ndarray): The image data to be written.
  descriptions (list of str): Descriptions for each band.
  geoTransform (tuple): Georeference transformation parameters.
  proj (str): Projection information.
  drv_name (str): The name of the driver to use for writing the file.

  Raises:
  Exception: If the image dimensions are invalid.
  """
  # Get the image driver by its short name
  driver = gdal.GetDriverByName(drv_name)
  
  # Get image dimensions from the array
  cols = np.size(imdata, 1)
  rows = np.size(imdata, 0)
  dims = np.shape(imdata.shape)[0]
  
  # Determine the number of bands based on dimensions
  if dims == 2:
      bands = 1
  elif dims == 3:
      bands = np.size(imdata, 2)
  else:
      raise Exception('Error x01: Invalid image dimensions.')
  
  # Determine the image data type
  dt = imdata.dtype
  datatype = gdal.GetDataTypeByName(dt.name)
  
  # Prepare the output dataset
  if datatype == 0:
      # Unknown datatype, try to use uint8 code
      datatype = 1
  
  outDataset = driver.Create(filename, cols, rows, bands, datatype)
  
  # Set the georeference information
  outDataset.SetGeoTransform(geoTransform)
  outDataset.SetProjection(proj)
  
  # Write the image data to the file with descriptions
  if bands == 1:
      outBand = outDataset.GetRasterBand(1)
      outBand.WriteArray(imdata, 0, 0)
  else:
      # Iterate over the bands and write them to the file with descriptions
      for b in range(bands):
          band_index = b + 1
          outBand = outDataset.GetRasterBand(band_index)
          outBand.SetDescription(descriptions[b])
          outBand.WriteArray(imdata[:, :, b], 0, 0)
  
  # Clear variables and close the file
  outBand = None
  outDataset = None
