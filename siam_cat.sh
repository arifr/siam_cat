#!/bin/bash

#SBATCH --job-name=siam_cat
#SBATCH --output=siam_cat-%j.out
#SBATCH --error=scat_err.out

#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 512 --miner pairmargin --loss-fn pair
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 512 --miner triplet_all --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 512 --miner triplet_easy --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 512 --miner triplet_semi --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 512 --miner triplet_hard --loss-fn triplet

#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 256--miner pairmargin --loss-fn pair
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 64 --code-size 512 --miner triplet_all --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 256 --miner triplet_easy --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 256 --miner triplet_semi --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 50 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 80 --code-size 256 --miner triplet_hard --loss-fn triplet

#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 256 --miner pairmargin --loss-fn pair
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 256 --miner triplet_all --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 256 --miner triplet_easy --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 256 --miner triplet_semi --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 256 --miner triplet_hard --loss-fn triplet

#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 128 --miner pairmargin --loss-fn pair
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 128 --miner triplet_all --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 128 --miner triplet_easy --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 128 --miner triplet_semi --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 128 --miner triplet_hard --loss-fn triplet

#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 64 --miner pairmargin --loss-fn pair
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 64 --miner triplet_all --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 64 --miner triplet_easy --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 64 --miner triplet_semi --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'SOPClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 64 --miner triplet_hard --loss-fn triplet

#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 512 --miner pairmargin --loss-fn pair
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 512 --miner triplet_all --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 512 --miner triplet_easy --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 512 --miner triplet_semi --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 512 --miner triplet_hard --loss-fn triplet

#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 256 --miner pairmargin --loss-fn pair
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 256 --miner triplet_all --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 256 --miner triplet_easy --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 256 --miner triplet_semi --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 256 --miner triplet_hard --loss-fn triplet

#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 128 --miner pairmargin --loss-fn pair
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 128 --miner triplet_all --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 128 --miner triplet_easy --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 128 --miner triplet_semi --loss-fn triplet
#srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 128 --miner triplet_hard --loss-fn triplet

srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 64 --miner pairmargin --loss-fn pair
srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 64 --miner triplet_all --loss-fn triplet
srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 64 --miner triplet_easy --loss-fn triplet
srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 64 --miner triplet_semi --loss-fn triplet
srun /home/m405305/miniconda3/bin/python siam_cat_main.py --max-epoch 100 --dataset 'custShopClass' --model 'mobilenet' --lr-model 0.0001 --batch-size 96 --code-size 64 --miner triplet_hard --loss-fn triplet



