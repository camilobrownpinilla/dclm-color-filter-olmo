#!/bin/bash
#SBATCH --job-name=color-filter
#SBATCH --output=logs/%A_%a.log
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
    --selected_dir /n/holyscratch01/sham_lab/dclm/color_filter_data/dclm_train_test \
    --dclm_scale 411m_1x\
    --evaluation light