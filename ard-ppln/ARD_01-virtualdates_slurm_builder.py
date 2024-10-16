# supported by BiCubes project (HFRI grant 3943)

"""
This script generates .slurm files for the Analysis Ready Data (ARD) production pipeline.
It is designed to handle virtual dates as part of a four-step process. The script takes
several command-line arguments to customize the execution, including the agricultural year,
output date, tile names, CPU usage, and account information.

Usage:
  python script.py --year YYYY --date MMDD --tiles TILE_NAMES --cpus CPUS --acc ACCOUNT

Arguments:
  --year: The agricultural year for which the pipeline will be executed (format: YYYY).
  --date: The first date for which outputs will be generated (format: MMDD).
  --tiles: Sentinel-2 tile names (format: 34XXX or 35XXX). Default is all tiles depicting Greece.
  --cpus: Number of CPUs to use per task (max 20). Default is 20.
  --acc: Account name provided by administrators. Default is "pr015025_thin".
"""

import os
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

# List of all Greek tiles sorted alphabetically
all_greek_tiles = ["34TGL","34TGM","34SEF","34SEG","34SEH",
"34SEJ","34SFF","34SDG","34SDH","34SDJ","34TGK",
"34SFG","34SFH","34SFJ","34SCJ","34TFL","34TFK",
"34TDL","34SGF","34SGG","34SGH","34SGJ","34SGE",
"34TEK","34TEL","34TDK","34TCK",
"35SLU","35SLV","35SMA","35SMB","35SMC","35TLF",
"35SMD","35SMU","35SMV","35SNA","35SNV","35TMF",
"35SKA","35TMG","35SKB","35SKC","35SKD","35SKU",
"35SKV","35SLA","35SLB","35TLE","35SPA",
"35SQA","35SLC","35SLD"]
all_greek_tiles.sort()

# Argument parser setup
parser = argparse.ArgumentParser(description='Generate .slurm files for Analysis Ready Data production, virtual dates step 1 of 4')
parser.add_argument("--year", dest="agricultural_year", help="Agricultural year for which the pipeline will be executed, in format YYYY.", default='2020')
parser.add_argument("--date", dest="first_output_date", help="First date for which outputs will be generated, in format MMDD", default='1031')
parser.add_argument("--tiles", dest="tiles", nargs='+', help="Sentinel-2 tile names, in format 34XXX or 35XXX. If multiple separate by space. Default value: all 53 tiles depicting Greece.", default=all_greek_tiles)
parser.add_argument("--cpus", dest="cpus_per_task", help="Number of cpus to use per task (max 20).", default='20')
parser.add_argument("--acc", dest="account", help="Account name, provided by administrators.", default="pr015025_thin")
args = parser.parse_args()

# Extracting and converting arguments
ay = int(args.agricultural_year)
fod = args.first_output_date
tiles = args.tiles
cpus_per_task = int(args.cpus_per_task)
account = args.account

# SLURM job configuration
ntasks = 1
nodes = 1
ntasks_per_node = 20
mem = 56
partition = "compute"
time_string = "05:00:00"
homedir = "/users/pa17/rslab/"

# Paths for various scripts and directories
python_fullpath = os.path.join(homedir, 'envs/s2ppln/bin/python3')
ardppln_fullpath = os.path.join(homedir, "jobs/ard-ppln")
log_fullpath = os.path.join(homedir, "jobs/log")
virtual_dates_script_fullpath = os.path.join(homedir, 'tools/ard-ppln/ard-ppln_virtual_dates.py')
experiment_name = "AY" + str(ay)
dirname = "virtual_dates_" + experiment_name
fullpath = os.path.join(ardppln_fullpath, dirname)

# Create directory if it doesn't exist
if not os.path.exists(fullpath):
  os.mkdir(fullpath)
os.chdir(fullpath)

# Options for input and output paths
L2A_fullpath = os.path.join(homedir, 'work/maja-run/L2A')
output_fullpath = os.path.join(homedir, 'work/ard-ppln/', experiment_name)

# Date calculations for input and output
from_output_date = str(ay - 1) + fod
to_output_date = str(ay) + fod
date_format = "%Y%m%d"
from_output_date_obj = datetime.strptime(from_output_date, date_format)
from_input_date_obj = from_output_date_obj - relativedelta(days=31)
from_input_date = from_input_date_obj.strftime(date_format)
to_output_date_obj = datetime.strptime(to_output_date, date_format)
to_input_date_obj = to_output_date_obj + relativedelta(days=31)
to_input_date = to_input_date_obj.strftime(date_format)
input_date_range = " ".join([from_input_date, to_input_date])
output_date_range = " ".join([from_output_date, to_output_date])

# Interval and paths for additional data
interval = '5'
swm_fullpath = os.path.join(homedir, 'work/data/SeaWater_Masks/')
reg_fullpath = os.path.join(homedir, 'work/data/Registration_Base_Images')

# Loop through each tile to generate SLURM files
for tile in tiles:
  # Calculate number of scenes for the given date range and tile
  scene_count = 0
  tilepath = os.path.join(L2A_fullpath, tile)
  os.chdir(tilepath)

  # Options for the virtual dates script
  virtual_dates_options_list = ['--input', tilepath, '--output', output_fullpath, '--ir', input_date_range, '--or', output_date_range, '--int', interval, '--swm', swm_fullpath, '--reg', reg_fullpath]
  virtual_dates_options = " ".join(virtual_dates_options_list)

  # SLURM file configuration
  slurm_filename = dirname + '_' + tile
  slurm_lines = [
      "#!/bin/bash -l",
      "",
      "#SBATCH --job-name=" + slurm_filename,
      "#SBATCH --output=" + log_fullpath + "/" + slurm_filename + ".%j.out",
      "#SBATCH --error=" + log_fullpath + "/" + slurm_filename + "j.err",
      "#SBATCH --ntasks=" + str(ntasks),
      "#SBATCH --nodes=" + str(nodes),
      "## #SBATCH --ntasks-per-node=" + str(ntasks_per_node),
      "#SBATCH --cpus-per-task=" + str(cpus_per_task),
      "#SBATCH --mem=" + str(mem) + "G",
      "#SBATCH --time=" + time_string,
      "#SBATCH --partition=" + partition,
      "#SBATCH --account=" + account,
      "",
      'if [ x$SLURM_CPUS_PER_TASK == x ]',
      "then export OMP_NUM_THREADS=1",
      "else export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
      "fi",
      "",
      "export PATH=/bin:$PATH",
      "## LOAD MODULES",
      "module purge",
      "module load intel/19      # required by gdal",
      "module load proj4/4.9.3   # required by gdal",
      "module load gnu/6.4.0     # required by gdal",
      "module load python/3.6.5  # required by gdal",
      "module load opencv/3.4.0",
      "module load gdal/2.2.0    # required by gdal",
      "",
      "# run your program: srun <EXECUTABLE> <EXECUTABLE ARGUMENTS> ",
      " ".join(["srun", python_fullpath, virtual_dates_script_fullpath, virtual_dates_options])
  ]

  # Write the SLURM file
  slurm_fullpath = os.path.join(fullpath, slurm_filename + '.slurm')
  with open(slurm_fullpath, 'w') as f:
      print("Writing file:", slurm_fullpath)
      for line in slurm_lines:
          f.write(line)
          f.write('\n')
  
  # Append the SLURM command to a batch file
  commands_fullpath = os.path.join(fullpath, 'sbatch_slurm_commands.txt')
  with open(commands_fullpath, 'a') as f:
      f.write('sbatch ' + slurm_fullpath)
      f.write('\n')
