import re
import numpy as np


def get_number (map_str):
    map_str_ = []
    for temp in map_str:
        map_str_.append (temp[-7:-1])
    map_str_int = np.array (map_str_)

    return map_str_int

filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/fashionCV2_0822.log'

f = open(filename, 'r')
buff = f.read()

lambda_1 = re.compile('lambda_1:'+'.*?'+']')
lambda_2 = re.compile('lambda_2:'+'.*?'+']')
mAP_sy = re.compile(': mAP_sy:'+'.*?'+']')
mAP_asy = re.compile(': mAP_asy:'+'.*?'+']')
topK_mAP_sy = re.compile('topK_mAP_sy:'+'.*?'+']')
topK_mAP_asy = re.compile('topK_mAP_asy:'+'.*?'+']')
Pres_sy = re.compile('Pres_sy:'+'.*?'+']')
Pres_asy = re.compile('Pres_asy:'+'.*?'+']')
topK_ndcg_sy = re.compile('topK_ndcg_sy:'+'.*?'+']')
topK_ndcg_asy = re.compile('topK_ndcg_asy:'+'.*?'+']')

re_lambda_1=lambda_1.findall(buff)
re_lambda_2=lambda_2.findall(buff)
re_mAP_sy=mAP_sy.findall(buff)
re_mAP_asy=mAP_asy.findall(buff)
re_topK_mAP_sy=topK_mAP_sy.findall(buff)
re_topK_mAP_asy=topK_mAP_asy.findall(buff)
re_Pres_sy=Pres_sy.findall(buff)
re_Pres_asy=Pres_asy.findall(buff)
re_topK_ndcg_sy=topK_ndcg_sy.findall(buff)
re_topK_ndcg_asy=topK_ndcg_asy.findall(buff)

mAP_sy_int = get_number(re_mAP_sy)
mAP_asy_int = get_number(re_mAP_asy)
topK_mAP_sy_int = get_number(re_topK_mAP_sy)
topK_mAP_asy_int = get_number(re_topK_mAP_asy)
Pres_sy_int = get_number(re_Pres_sy)
Pres_asy_int = get_number(re_Pres_asy)
topK_ndcg_sy_int = get_number(re_topK_ndcg_sy)
topK_ndcg_asy_int = get_number(re_topK_ndcg_asy)

print(result)