#!/bin/bash

lambda_2s='0.00001 0.0001 0.001 0.1 10 1000 1000'

#for lambda_2 in ${lambda_2s}
#do
#python DC_DAGH.py  --lambda-1 10 --lambda-2 $lambda_2 --lambda-3 1 --gpu 7
#echo $lambda_ba
#done 
for lambda_2 in ${lambda_2s}
do
python DC_L.py --gpu 7
done