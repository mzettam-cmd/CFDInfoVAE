#!/bin/bash
#SBATCH -J Root
#SBATCH --partition=shortq
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

module  load Python/3.9.6-GCCcore-11.2.0
source ~/my_ai_env/bin/activate

echo "Running VAE training..."

python main.py -re 40 -m run -nn easy

echo "Done"