#!/bin/bash
#SBATCH --job-name=dclm-tok1-fullRun
#SBATCH --output=logs/%A_%a.log
#SBATCH -p kempner_h100
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2     
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
    --selected_dir /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/tokshuf/dclm-filtered/dclm-tokenized-1 \
    --dclm_scale 411m_1x\
    --evaluation mmlu_and_lowvar\
    --multiple_data_passes