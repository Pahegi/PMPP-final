#!/bin/bash

# SLURM parameters
#SBATCH -n 1
#SBATCH -t 5
#SBATCH --mem-per-cpu 3800
#SBATCH --gpus-per-task=1

# Special parameters. DO NOT CHANGE THESE!
#SBATCH -A kurs00082
#SBATCH -p kurs00082
#SBATCH --reservation=kurs00082

# Redirect stdout and stderr
#SBATCH -o /work/scratch/kurse/kurs00082/ph37bilu/test.out
#SBATCH -e /work/scratch/kurse/kurs00082/ph37bilu/test.err
#SBATCH -D /work/scratch/kurse/kurs00082/ph37bilu
#SBATCH --time=0:02:00             # Maximale Laufzeit (hh:mm:ss)
#SBATCH --cpus-per-task=1          # Anzahl CPU-Kerne

module purge
module load cuda/12.5 gcc/13.1.0
mkdir -p $SCRATCH/out
# run normally
srun $SCRATCH/build/test_pmpp_final

# run with profiling
# srun nvprof --log-file $SCRATCH/out/nvprof_trace.txt $SCRATCH/build/test_pmpp_final
