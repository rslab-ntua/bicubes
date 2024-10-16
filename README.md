# BiCubes Project: Analysis-Ready Geospatial Big Data Cubes and Cloud-based Analytics

Welcome to the BiCubes project, supported by HFRI grant 3943. This project introduces a holistic and scalable approach to processing and analyzing Earth Observation (EO) data, focusing on data harmonization and machine learning analytics to enhance geospatial applications for monitoring land and water.

## Overview

BiCubes aims to revolutionize the use of satellite imagery by creating Analysis Ready Data (ARD) and developing advanced machine learning frameworks. The project is divided into two main components:

1. **Satellite Imagery Processing and Classification**: This component focuses on processing and classifying satellite imagery using machine learning techniques, including transfer learning and Random Forest classifiers.

2. **Analysis Ready Data (ARD) Production Pipeline**: This component generates ARD from satellite imagery, ensuring the data is geometrically and radiometrically corrected, temporally consistent, and spatially enhanced.

## Key Objectives

- **Data Harmonization**: Develop methodologies to harmonize high-resolution, multitemporal EO data into cloud/shadow-free geospatial data cubes.
- **Machine Learning Frameworks**: Design robust machine learning frameworks for semantic information extraction and analytics, utilizing deep learning techniques such as generative adversarial networks and recurrent neural networks.
- **Scalable Solutions**: Exploit big data technologies and cloud environments to develop scalable solutions for monitoring land and water resources.

## Repositories

### 1. Satellite Imagery Processing and Classification

This repository includes scripts for:

- **Image Classification**: Using transfer learning with a pretrained ResNet50 model.
- **Random Forest Classification**: Training and evaluating classifiers for image segmentation.
- **Temporal Feature Processing**: Extracting and processing temporal features.
- **Image Mosaicking**: Creating mosaics from multiple images.

#### Key Dependencies

- Python packages: `numpy`, `tensorflow`, `keras`, `sklearn`, `osgeo`, `pandas`, `natsort`, `joblib`, `optparse`, `xlsxwriter`, `multiprocessing`, `rasterio`, `tqdm`, `argparse`.

### 2. Analysis Ready Data (ARD) Production Pipeline

This repository provides a pipeline for:

- **Virtual Dates Generation**: Creating synthetic images for specified dates.
- **Registration (AROP)**: Aligning images for geometric consistency.
- **Sharpening**: Enhancing spatial resolution using high-pass filter fusion.
- **Cleanup**: Organizing and cleaning up output data.

#### Key Dependencies

- Python packages: `GDAL`, `OpenCV`, `NumPy`, `Pandas`, `Rasterio`, `TQDM`, `Natsort`, `Openpyxl`.

## Installation

To set up the environment for both repositories, ensure you have Python 3.6 or higher and install the required packages using pip:
bash
pip install -r requirements.txt


## Usage

Each repository contains detailed instructions on how to run the scripts and customize the processing for different datasets. Refer to the individual `README.md` files in each sub-repository for specific usage guidelines.

## Contact

For any questions or issues, please contact the project team at [karank@central.ntua.gr](mailto:karank@central.ntua.gr).

## Acknowledgments

This project is supported by the BiCubes project (HFRI grant 3943). We acknowledge the contributions of all team members involved in the development and testing of these pipelines.

## License

This project is licensed under the MIT License.
