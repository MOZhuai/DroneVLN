#!/bin/bash

cd ../..
exit_code=4

while [ $exit_code -eq 134 ] || [ $exit_code -eq 4 ] || [ $exit_code -eq 139 ]
do
  sleep 6
  python3 mains/train/train_dagger.py start_aux_map_spa_sbert_finetune_stage2
  exit_code=$?
  echo $exit_code
done

exit 0
