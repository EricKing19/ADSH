#!/bin/bash

lambda_1s=(0.000001 0.00001 0.0001 0.001 0.01)
lambda_3s=(0.000001 0.00001 0.0001 0.001 0.01)

for lambda_1 in ${lambda_1s[*]}
do
  for lambda_3 in ${lambda_3s[*]}
  do
  python DC_L.py --lambda-1 $lambda_1 --lambda-3 $lambda_3 --gpu 7
  echo $lambda_1,$lambda_3
  done
done 