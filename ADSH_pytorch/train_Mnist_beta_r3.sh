#!/bin/bash

lambda_1=(0.01)
lambda_2=(0.02)
lambda_3=(0)
beta=(0.00001 0.0001 0.001 0.01 0.1 1 10 100)
bits=(16)

echo 'B_random DAGH_run2 *10'

for b in ${beta[*]}
do
  python DAGH_run2_194.py --lambda-1 $lambda_1 --lambda-2 $lambda_2 --lambda-3 $lambda_3 --beta-1 $b --beta-2 $b --epochs=2 --max-iter=6 --learning-rate=0.01 --bits $bits --dataname=Mnist --gpu 2
  echo '###################################'
  echo $b 
  echo '###################################'
done
