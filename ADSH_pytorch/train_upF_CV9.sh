#!/bin/bash

lambda_1s=(0.01 0.02 0.04 0.1)
lambda_2s=(0.01 0.02 0.04 0.1)
lambda_3s=(0)

echo 'mean no BBT only upF'

for lambda_1 in ${lambda_1s[*]}
do
  for lambda_2 in ${lambda_2s[*]}
  do
    for lambda_3 in ${lambda_3s[*]}
      do
      python DAGH_tr_meanF.py --lambda-1 $lambda_1 --lambda-2 $lambda_2 --lambda-3 $lambda_3 --epochs=2 --max-iter=6 --learning-rate=0.01 --bits=64 --dataname=imageNet50 --gpu 0
      echo '###################################'
      echo $lambda_1, $lambda_2, $lambda_3
      echo '###################################'
      done
  done
done
