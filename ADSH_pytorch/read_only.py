import pickle
import os
import torch
import numpy as np
import scipy.io

def load_label(filename, DATA_DIR):
    path = os.path.join (DATA_DIR, filename)
    fp = open (path, 'r')
    labels = [x.strip () for x in fp]
    fp.close ()
    return torch.LongTensor (list (map (int, labels)))


###R2
#filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/run2-SUN20-18-11-12-17-23-44/16bits-record.pkl'
#filename  = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/run2-SUN20-18-10-15-20-51-21/16bits-record.pkl'
filename  = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/run2-Mnist-18-11-14-17-12-04/16bits-record.pkl'


#filename =  '/home/dacheng/PycharmProjects/ADSH_pytorch/newlog/run_new2-SUN20-18-11-01-14-48-52//8bits-record.pkl'
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-STL10-0.7052//8bits-record.pkl' ###  no tanh both
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-fashion_mnist-18-06-22-23-25-50/8bits-record.pkl' ###  no tanh both
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-CIFAR100-18-06-25-22-52-49//8bits-record.pkl' ###  no tanh both
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-21-14-55-47/8bits-record.pkl' ###  tanh only F1
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-21-14-55-47/8bits-record.pkl' ### tanh only F1
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-21-11-30-48/8bits-record.pkl' ### no tanh only F1
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-21-10-35-52/8bits-record.pkl' ### no tanh
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-21-09-07-57//8bits-record.pkl' ### 0.05 tanh both
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-20-22-25-43//8bits-record.pkl' ### 0.5 both poor
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-20-15-35-48//8bits-record.pkl' ### only var
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/log-DAGH-FF-cifar10-18-05-07-11-12-49/12bits-record.pkl'
# filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-05-21-03-36//8bits-record.pkl' ## no tanh
# /home/dacheng/PycharmProjects/ADSH_pytorch/log/log-DAGH-FF-cifar10-18-05-07-11-12-49/
inf = pickle.load (open (filename))

scipy.io.savemat('mnist_16_B_data_3.mat',{'qb':inf['qB'], 'b_sy':inf['rB_sy'], 'b_asy':inf['rB_asy'], 'F':inf['F']})
#scipy.io.savemat('Z_mean_B.mat',{'b_asy':inf['rB_asy'], 'b_sy':inf['rB_sy']})

print (inf['S_time'])
print (inf['b_time'])
print (inf['iter_time'])
print (inf['n_time'])

dataname = 'CIFAR100'
rootpath = os.path.join ('/data/dacheng/Datasets/', dataname)

testlabels_ = load_label ('test_label.txt', rootpath).numpy()
databaselabels_ = load_label ('train_label.txt', rootpath).numpy()
code_length  = np.size (inf['rB'], 1)

# qB = np.zeros ((len(testlabels_), code_length), dtype=np.float)
# rB = np.zeros ((len(databaselabels_), code_length), dtype=np.float)
qB =[]
rB=[]

for i in range(100):
    qb_temp = inf['qB'][testlabels_==i]
    rb_temp = inf['rB'][databaselabels_==i]
    qB.append(qb_temp)
    rB.append(rb_temp)

qB =  np.array(qB).reshape((-1,8))
rB =  np.array(rB).reshape((-1,8))

inf
