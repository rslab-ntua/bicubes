#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Supported by BiCubes project (HFRI grant 3943)

"""
Mosaic images.

This script creates a mosaic from given images using rasterio. It reads images from a specified directory,
computes the spatial intersection, and writes the mosaic to an output file.

Requirements:
- numpy
- rasterio
- argparse
"""

import os
import glob
import numpy as np
import rasterio
from rasterio import windows
from rasterio.coords import disjoint_bounds
from rasterio.transform import Affine
from argparse import ArgumentParser

if __name__ == '__main__':
  # Create argument-parser
  arg_parser = ArgumentParser(description="Mosaic given images.")

  # Add arguments to argument-parser
  arg_parser.add_argument("-src", dest="scr_path", required=True, type=str,
                          help="Full path of directory containing images. This will also be the output directory.")

  arg_parser.add_argument("-nm", dest="version", required=True, type=str,
                          help="Suffix of result filename (without format).")

  # Parse arguments
  args = arg_parser.parse_args()

  # Output filename
  dst_fname_mos = os.path.join(args.scr_path, f'mosaic_{args.version}.tif')

  ### MOSAICING
  # Find mosaics
  mos_34 = glob.glob(os.path.join(args.scr_path, f"mosaic_34.tif"))[0]            # NOTE: Hardcoded
  mos_3534 = glob.glob(os.path.join(args.scr_path, f"mosaic_35_rprj.tif"))[0]     # NOTE: Hardcoded
  first_35 = [mos_3534, mos_34]

  with rasterio.open(mos_34, 'r') as first:
      first_profile = first.profile
      first_res = first.res
      nodataval = first.nodatavals[0]
      dt = first.dtypes[0]
      bands = first.count

  # Scan input files
  xs = []
  ys = []
  for dataset in first_35:
      with rasterio.open(dataset, 'r') as src:
          left, bottom, right, top = src.bounds
      xs.extend([left, right])
      ys.extend([bottom, top])
  # dst_w, dst_n = min(xs), max(ys)
  dst_s, dst_e = min(ys), max(xs)
  # Lock to CRS 34 images
  dst_w, dst_n = 300000.0, 4720000.0

  # Resolution/pixel size
  res = (10, 10)

  # Compute output array shape. We guarantee it will cover the output bounds completely
  output_width = int(round((dst_e - dst_w) / res[0]))
  output_height = int(round((dst_n - dst_s) / res[1]))

  output_transform = Affine.translation(dst_w, dst_n) * Affine.scale(res[0], -res[1])

  # Create destination array
  dest = np.zeros((bands, output_height, output_width), dtype=dt)
  nodataval = 0

  for idx, dataset in enumerate(first_35):
      with rasterio.open(dataset, 'r') as src:
          filename = os.path.basename(dataset)
          print(filename)

          if disjoint_bounds((dst_w, dst_s, dst_e, dst_n), src.bounds):
              print(f"Skipping source: {filename}")
              continue

          # 1. Compute spatial intersection of destination and source
          src_w, src_s, src_e, src_n = src.bounds

          int_w = src_w if src_w > dst_w else dst_w
          int_s = src_s if src_s > dst_s else dst_s
          int_e = src_e if src_e < dst_e else dst_e
          int_n = src_n if src_n < dst_n else dst_n

          # 2. Compute the source window
          src_window = windows.from_bounds(
              int_w, int_s, int_e, int_n, src.transform, precision=0)

          # 3. Compute the destination window
          dst_window = windows.from_bounds(
              int_w, int_s, int_e, int_n, output_transform, precision=0)

          # 4. Read data in source window into temp
          # Round src and dst windows
          src_window_rnd_shp = src_window.round_shape(pixel_precision=0)
          dst_window_rnd_shp = dst_window.round_shape(pixel_precision=0)
          # Column and row offsets rounded
          dst_window_rnd_off = dst_window_rnd_shp.round_offsets(pixel_precision=0)

          temp_height, temp_width = (
              dst_window_rnd_off.height,
              dst_window_rnd_off.width,
          )

          temp_shape = (bands, temp_height, temp_width)

          # Read current image
          temp_src = src.read(
              out_shape=temp_shape,
              window=src_window_rnd_shp,
              masked=True,
          )

      # 5. Copy elements of temp into dest
      # Keep from which row/col of dest starts my region (region=current image)
      roff, coff = (
          max(0, dst_window_rnd_off.row_off),
          max(0, dst_window_rnd_off.col_off),
      )
      region = dest[:, roff: roff + temp_height, coff: coff + temp_width]

      # Sea or black margin mask. False where class 0.
      sea_mask = temp_src[0, :, :] != 0

      # True where img is bigger than region.
      img_bigger_than_region_mask = np.greater(temp_src[1, :, :], region[1, :, :])

      # Combine masks. Write values when both masks are True.
      mask = np.logical_and(sea_mask, img_bigger_than_region_mask)
      mask = mask[np.newaxis, :, :]
      mask = np.vstack((mask, mask))

      # Copy classes from image to region using mask
      np.copyto(region, temp_src, where=mask, casting="no")

  first_profile.update(
      transform=output_transform,
      height=output_height,
      width=output_width,
      nodata=nodataval
  )
      
  with rasterio.open(dst_fname_mos, "w", **first_profile) as dst:
      dst.write(dest)
