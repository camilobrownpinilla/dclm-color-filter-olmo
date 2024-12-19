#!/bin/bash
#SBATCH --job-name=select-2b
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH -p sapphire
#SBATCH --account=pslade_lab
#SBATCH -N 1
#SBATCH -n 96          
#SBATCH --time=10:00:00
#SBATCH --mem=250GB		
#SBATCH --mail-type=ALL
#SBATCH --mail-user=camilobrownpinilla@college.harvard.edu

module load python/3.10.13-fasrc01
conda deactivate
conda activate color-filter
 
cd dclm_color_filter_olmo
python -m scripts.select_topk configs/dclm/select_topk.yaml