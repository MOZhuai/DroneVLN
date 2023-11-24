#!/bin/bash
cd ..

python3 mains/train/train_supervised.py spa_cot_train_stage1

#python3 mains/train/train_supervised.py corl_pvn_pretrain_stage2

exit 0
