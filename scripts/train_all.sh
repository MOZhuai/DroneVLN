#!/bin/bash
cd ..

python3 mains/train/train_supervised.py corl_pvn_train_stage1

python3 mains/train/train_supervised.py corl_pvn_pretrain_stage2

python3 mains/train/train_dagger.py corl_pvn_finetune_stage2

exit 0
