#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=kanitest
#
#SBATCH --output=/nlpgpu/data/andrz/logs/%j.%x.log
#SBATCH --error=/nlpgpu/data/andrz/logs/%j.%x.log
#SBATCH --time=7-0
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=320G
#SBATCH --gpus=8
#SBATCH --constraint=48GBgpu

python $*
