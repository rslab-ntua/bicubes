# BiCubes Project: Analysis Ready Data (ARD) Production Pipeline

This repository contains scripts and tools developed as part of the BiCubes project (HFRI grant 3943) for generating Analysis Ready Data (ARD) from satellite imagery. The pipeline is designed to process Sentinel-2 data through a series of steps, including virtual date creation, registration, sharpening, and cleanup.

## Project Overview

The BiCubes project aims to enhance the usability of satellite imagery by creating ARD products that are ready for analysis. This involves several processing steps to ensure the data is geometrically and radiometrically corrected, temporally consistent, and spatially enhanced.

## Pipeline Steps

1. **Virtual Dates Generation**: This step involves creating synthetic images for specified dates using interpolation techniques. It handles duplicate date data and generates virtual date images.

2. **Registration (AROP)**: Automated Registration and Ortho-rectification Processing (AROP) is performed to align images with a base image, ensuring geometric consistency across the dataset.

3. **Sharpening**: This step enhances the spatial resolution of the images using high-pass filter fusion techniques, improving the clarity and detail of the imagery.

4. **Cleanup**: The final step involves removing intermediate files and organizing the output data for easy access and analysis.

## Usage

Each step of the pipeline is executed through a separate script, which can be run with specific command-line arguments to customize the processing for different tiles and dates.

### Virtual Dates Generation
bash
python script.py --input INPUT_PATH --output OUTPUT_PATH --ir INPUT_DATE_RANGE --or OUTPUT_DATE_RANGE --int OUTPUT_INTERVAL --swm SWM_DIR_PATH --reg REG_BASE_IMAGE_FULLPATH


### Registration (AROP)

bash
python script.py --output OUTPUT_PATH --tile TILE_NAME --reg REG_BASE_IMAGE_FULLPATH --ortho ORTHO_FOLDER


### Sharpening

bash
python script.py --output OUTPUT_PATH --tile TILE_NAME --swm SWM_DIR_PATH --wm WM_DIR_PATH


### Cleanup

bash
python script.py --year YYYY --tiles TILE_NAMES --acc ACCOUNT


## Requirements

- Python 3.6 or higher
- GDAL
- OpenCV
- NumPy
- Pandas
- Rasterio
- TQDM
- Natsort
- Openpyxl

## Installation

To set up the environment, install the required Python packages using pip:

bash
pip install -r requirements.txt


## Acknowledgments

This work was conducted as part of the BiCubes project, supported by HFRI grant 3943. We acknowledge the contributions of all team members involved in the development and testing of this pipeline.

## License

This project is licensed under the MIT License.

## Contact

For questions or further information, please contact the project team at [karank@central.ntua.gr].
