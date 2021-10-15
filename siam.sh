#!/bin/bash

#SBATCH --job-name=siam
#SBATCH --output=siam-%j.out
#SBATCH --error=siam_err.out

#srun /home/m405305/miniconda3/bin/python siam_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 512 --miner pairmargin --loss-fn pair
#srun /home/m405305/miniconda3/bin/python siam_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 512 --miner triplet_all --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 512 --miner triplet_easy --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 512 --miner triplet_semi --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 512 --miner triplet_hard --loss-fn triplet
srun /home/m405305/miniconda3/bin/python siam_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 64 --code-size 512 --miner triplet_all --loss-fn triplet



