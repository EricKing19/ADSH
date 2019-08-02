#!/bin/bash


#python DC_L.py   --lambda-1 10 --lambda-2 0.5 --lambda-3 1 --lambda-ba 1 --gpu 2

lambda_bas='0.001 0.01 0.1 0.5 1 2 5 10 100'

for lambda_ba in ${lambda_bas}
do
python DC_L.py   --lambda-1 10 --lambda-2 0.5 --lambda-3 1 --lambda-ba $lambda_ba --gpu 2
echo $lambda_ba
done 