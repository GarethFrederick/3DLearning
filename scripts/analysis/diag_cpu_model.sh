#!/bin/bash
#SBATCH --job-name=cpumodel
#SBATCH --time=00:03:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --constraint=milan
# Reads the CPU model string of a Milan node for the thesis compute statement.
echo "node: $(hostname)"
lscpu | grep -E "Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|^CPU\(s\)"
free -g | head -2
module load miniforge3; source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
$PY -c "import gurobipy as g; print('gurobipy', g.gurobi.version(), 'from', g.__file__)"
echo CPU_MODEL_DONE
