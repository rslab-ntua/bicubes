# supported by BiCubes project (HFRI grant 3943)

"""
This script generates .slurm files for the Analysis Ready Data (ARD) production pipeline,
specifically for the CLEANUP step, which is the fourth and final step in the process. The
script allows customization through command-line arguments, including the agricultural year,
tile names, and account information.

Usage:
  python script.py --year YYYY --tiles TILE_NAMES --acc ACCOUNT

Arguments:
  --year: The agricultural year for which the pipeline will be executed (format: YYYY).
  --tiles: Sentinel-2 tile names (format: 34XXX or 35XXX). Default is all tiles depicting Greece.
  --acc: Account name provided by administrators. Default is "pr015025_thin".
"""

import os
import argparse

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
parser = argparse.ArgumentParser(description='Generate .slurm files for Analysis Ready Data production, CLEANUP step 4 of 4')
parser.add_argument("--year", dest="agricultural_year", help="Agricultural year for which the pipeline will be executed, in format YYYY.", default='2020')
parser.add_argument("--tiles", dest="tiles", nargs='+', help="Sentinel-2 tile names, in format 34XXX or 35XXX. If multiple separate by space. Default value: all 53 tiles depicting Greece.", default=all_greek_tiles)
parser.add_argument("--acc", dest="account", help="Account name, provided by administrators.", default="pr015025_thin")
args = parser.parse_args()

# Extracting and converting arguments
ay = int(args.agricultural_year)
tiles = args.tiles
cpus_per_task = 1
account = args.account

# SLURM job configuration
ntasks = 1
nodes = 1
ntasks_per_node = 1
mem = 4
partition = "compute"
time_string = "00:05:00"
homedir = "/users/pa17/rslab/"

# Paths for various scripts and directories
python_fullpath = os.path.join(homedir, 'envs/s2ppln/bin/python3')
ardppln_fullpath = os.path.join(homedir, "jobs/ard-ppln")
log_fullpath = os.path.join(homedir, "jobs/log")
experiment_name = "AY" + str(ay)
dirname = "cleanup_" + experiment_name
fullpath = os.path.join(ardppln_fullpath, dirname)

# Create directory if it doesn't exist
if not os.path.exists(fullpath):
  os.mkdir(fullpath)
os.chdir(fullpath)

# Options for input and output paths
L2A_fullpath = os.path.join(homedir, 'work/maja-run/L2A')
output_fullpath = os.path.join(homedir, 'work/s2ppln/', experiment_name)
reg_fullpath = os.path.join(homedir, 'work/data/Registration_Base_Images.xlsx')

# Loop through each tile to generate SLURM files
for tile in tiles:
  # Path for the current tile
  tilepath = os.path.join(L2A_fullpath, tile)
  slurm_filename = dirname + '_' + tile
  
  # SLURM file configuration
  slurm_lines = [
      "#!/bin/bash -l",
      "",
      "#SBATCH --job-name=" + slurm_filename,
      "#SBATCH --output=" + log_fullpath + "/" + slurm_filename + ".%j.out",
      "#SBATCH --error=" + log_fullpath + "/" + slurm_filename + "j.err",
      "#SBATCH --ntasks=" + str(ntasks),
      "#SBATCH --nodes=" + str(nodes),
      "#SBATCH --ntasks-per-node=" + str(ntasks_per_node),
      "### #SBATCH --cpus-per-task=" + str(cpus_per_task),
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
      "# run your program: srun <EXECUTABLE> <EXECUTABLE ARGUMENTS> ",
      " ".join(["srun", "rm", "-rf", os.path.join(output_fullpath, tile, 'intermediate_output')])
  ]

  # Write the SLURM file
  slurm_fullpath = os.path.join(fullpath, slurm_filename + '.slurm')
  with open(slurm_fullpath, 'w') as f:
      for line in slurm_lines:
          f.write(line)
          f.write('\n')
  
  # Append the SLURM command to a batch file
  commands_fullpath = os.path.join(fullpath, 'sbatch_slurm_commands.txt')
  with open(commands_fullpath, 'a') as f:
      f.write('sbatch ' + slurm_fullpath)
      f.write('\n')
