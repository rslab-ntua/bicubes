# Supported by BiCubes project (HFRI grant 3943)

from numpy import moveaxis
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly

def geoimread(filename):
  """
  Reads a georeferenced image file and returns its data along with georeference information.

  Parameters:
  filename (str): The path to the georeferenced image file.

  Returns:
  tuple: A tuple containing:
      - imdata (numpy.ndarray): The image data.
      - geoTransform (tuple): Georeference transformation parameters.
      - proj (str): Projection information.
      - drv_name (str): The name of the driver used to read the file.
  """
  # Open the file with ReadOnly access
  dataset = gdal.Open(filename, GA_ReadOnly)
  
  # Read image data as an array
  imdata = dataset.ReadAsArray()
  
  # Get the driver name used to open the file
  drv_name = dataset.GetDriver().ShortName
  
  # Get georeference transformation parameters
  geoTransform = dataset.GetGeoTransform()
  
  # Get projection information
  proj = dataset.GetProjection()
  
  # Clear variables and release the file
  dataset = None

  # If the image data has more than two dimensions, move the axis
  if imdata.ndim > 2:
      imdata = moveaxis(imdata, 0, 2)

  return (imdata, geoTransform, proj, drv_name)
