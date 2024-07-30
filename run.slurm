#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=SM-UNet-OSD
#SBATCH --time=0
#SBATCH --mem=0

/home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-tf/bin/python main.py
