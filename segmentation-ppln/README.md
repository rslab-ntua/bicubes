# Satellite Imagery Processing and Classification

This repository contains scripts for processing and classifying satellite imagery data. The work is conducted as part of the BiCubes project, supported by HFRI grant 3943.

## Overview

The scripts in this repository are designed to handle various tasks related to satellite imagery, including:

- Image classification using transfer learning with a pretrained ResNet50 model.
- Training and evaluating a Random Forest classifier on satellite imagery data.
- Processing satellite imagery data to classify and segment images using a pre-trained classifier.
- Extracting temporal features from satellite imagery data.
- Creating mosaics from given images.

## Dependencies

The scripts require the following Python packages:

- `numpy`
- `tensorflow`
- `keras`
- `sklearn`
- `osgeo` (gdal)
- `pandas`
- `natsort`
- `joblib`
- `optparse`
- `xlsxwriter`
- `multiprocessing`
- `rasterio`
- `tqdm`
- `argparse`

## Scripts

### Image Classification with ResNet50

- **Description**: This script performs image classification using transfer learning with a pretrained ResNet50 model. It fine-tunes the model on a custom dataset and evaluates its performance.
- **Usage**: Run the script with the required arguments for data path, number of epochs, batch size, image size, number of classes, and learning rate.

### Random Forest Classifier

- **Description**: This script trains and evaluates a Random Forest classifier on satellite imagery data. It handles feature extraction, model training, and validation, and outputs a confusion matrix.
- **Usage**: Run the script with the required options for feature type, nomenclature file path, output path, training and validation tile names.

### Image Segmentation and Classification

- **Description**: This script processes satellite imagery data to classify and segment images using a pre-trained classifier. It reads input images, computes indices, and applies a Random Forest classifier to generate prediction maps.
- **Usage**: Run the script with the required options for tile name, input path, output path, classifier file, and feature type.

### Temporal Feature Processing

- **Description**: This script processes temporal features from satellite imagery data. It reads input feature tables, computes temporal statistics, and saves the results.
- **Usage**: Run the script with the required options for tile name and output path.

### Image Mosaicking

- **Description**: This script creates a mosaic from given images using rasterio. It reads images from a specified directory, computes the spatial intersection, and writes the mosaic to an output file.
- **Usage**: Run the script with the required arguments for the source path and version suffix for the result filename.

## Contact

For any questions or issues, please contact [karank@central.ntua.gr](mailto:karank@central.ntua.gr).

---

This project is supported by the BiCubes project (HFRI grant 3943).
