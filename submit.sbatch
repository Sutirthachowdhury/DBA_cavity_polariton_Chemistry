#!/bin/bash
#SBATCH --partition=et3
#SBATCH --output bridge_rate.out
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1


time python3 bridge_coup.py
