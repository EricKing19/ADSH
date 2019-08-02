#!/bin/bash

lambda_1=(0.01)
lambda_2=(0.02)
lambda_3=(0)
bits=(8 16 32 64)

echo 'B_random DAGH_run2'

for bit in ${bits[*]}
do
  python DAGH_run3.py --lambda-1 $lambda_1 --lambda-2 $lambda_2 --lambda-3 $lambda_3 --epochs=2 --max-iter=6 --learning-rate=0.01 --bits $bit --dataname=NUSWIDE --gpu 1
  echo '###################################'
  echo $bit 
  echo '###################################'
done
