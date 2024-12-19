#!/bin/bash
#SBATCH --job-name=1b-fullRun
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH -p kempner_h100
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4     
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=250GB		
#SBATCH --constraint=h100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=camilobrownpinilla@college.harvard.edu
#SBATCH --array=1-1

# Custom environment
source ~/.bashrc
module load python/3.10.13-fasrc01
conda deactivate
conda activate color-filter
cd DCLM/

python ../dclm.py \
    --selected_dir /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/selected/dclm-filtered_core-train-tasks_3-to-5/top_1b \
    --dclm_scale 411m_1x\
    --evaluation heavy\