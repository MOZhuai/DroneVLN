#!/bin/bash

cd ../..
exit_code=4

while [ $exit_code -eq 134 ] || [ $exit_code -eq 4 ] || [ $exit_code -eq 139 ]
do
  sleep 6
  python3 -X faulthandler mains/eval/evaluate.py spa_sbert_eval
  exit_code=$?
  echo $exit_code
done

exit 0
