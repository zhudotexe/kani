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

python sandbox/trains_test.py meta-llama/Llama-3.3-70B-Instruct NaiveJSONToolCallParser
python sandbox/trains_test.py mistralai/Mistral-Large-Instruct-2411 MistralToolCallParser
python sandbox/trains_test.py deepseek-ai/DeepSeek-R1 DeepSeekR1ToolCallParser
