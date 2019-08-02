#!/bin/bash

lambda_1s=(0.02 0.02)
lambda_2s=(0.02 0.02)
lambda_3s=(0)

echo 'DAGH_run2_194 *10 poly'

for lambda_1 in ${lambda_1s[*]}
do
  for lambda_2 in ${lambda_2s[*]}
  do
    for lambda_3 in ${lambda_3s[*]}
      do
      python DAGH_run2_194.py --lambda-1 $lambda_1 --lambda-2 $lambda_2 --lambda-3 $lambda_3 --epochs=2 --max-iter=6 --learning-rate=0.01 --bits=16 --dataname=fashion_mnist --gpu 7
      echo '###################################'
      echo $lambda_1, $lambda_2, $lambda_3
      echo '###################################'
      done
  done
done
