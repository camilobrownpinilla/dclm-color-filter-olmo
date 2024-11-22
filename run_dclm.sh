#!/bin/bash
#SBATCH --job-name=color-filter
#SBATCH --output=logs/%A_%a.log
#SBATCH -p kempner
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4     
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=250GB		
#SBATCH --constraint=a100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=camilobrownpinilla@college.harvard.edu
#SBATCH --array=1-1

# Custom environment
source ~/.bashrc
module load python/3.10.13-fasrc01
conda deactivate
conda activate color-filter

python ../dclm.py --filtered_dir /n/holyscratch01/sham_lab/dclm/color_filter_data/first_run/prior/chunk_scores.jsonl --selected_dir /n/holyscratch01/sham_lab/dclm/color_filter_data/pipeline_test --k 10 --type chunks --evaluation light