import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import h5py

filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/newlog/run2-SUN20-18-11-01-14-41-33/8bits-record.pkl'
#filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/newlog/run_new2-SUN20-18-11-01-16-09-05/8bits-record.pkl' 
#filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-22-09-17-23/8bits-record.pkl' ## no L1
#filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-21-22-33-15/8bits-record.pkl'
#filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-21-10-35-52/8bits-record.pkl'
#filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-21-14-55-47/8bits-record.pkl'
#filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/log-DAGH-FF-cifar10-18-05-07-11-12-49/12bits-record.pkl'
#filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-05-21-03-36//8bits-record.pkl'
#/home/dacheng/PycharmProjects/ADSH_pytorch/log/log-DAGH-FF-cifar10-18-05-07-11-12-49/
inf = pickle.load(open(filename))

inf


from matplotlib import pyplot


def drawHist(heights):
    pyplot.hist(heights, 100)
    pyplot.xlabel('Heights')
    pyplot.ylabel('Frequency')
    pyplot.title('Heights Of Male Students')
    pyplot.show()


def load_label(filename, DATA_DIR):
    path = os.path.join(DATA_DIR, filename)
    fp = open(path, 'r')
    labels = [x.strip() for x in fp]
    fp.close()
    return list(map(int, labels))

b =inf['F'].reshape(-1,1)
qB =inf['qB']
rb = inf['rB_sy']
F = inf['F']
map = inf['map_sy']
drawHist(b)

data = scipy.io.loadmat('/home/dacheng/Hash_code/Similarity-Adaptive-Deep-Hashing-master/analysis/cifar10/tt.mat') 
# print(data.keys())   # 查看mat文件中的所有变量
data = h5py.File('/home/dacheng/Hash_code/Similarity-Adaptive-Deep-Hashing-master/analysis/cifar10/tt.mat')

print(data['matrix1'])
print(data['matrix2'])
matrix1 = data['matrix1'] 
matrix2 = data['matrix2']

#databaselabels = load_label('database_label.txt', 'data/CIFAR-10')

label_color_map = {0:'lime', 1:'red', 2:'yellow', 3:'magenta', 4:'blue',
                   5:'cyan', 6:'green', 7:'purple', 8:'deepskyblue', 9:'gold',
                   10:'gray', 11:'orange', 12:'gold', 13:'darkviolet', 14:'pink'}


testlabels = load_label('test_label.txt', '/data/dacheng/Datasets/cifar10/')
databaselabels = load_label('train_label.txt', '/data/dacheng/Datasets/cifar10/')

colors = [label_color_map[temp] for temp in databaselabels]


#tspae_tsne = TSNE(n_components=2).fit_transform(qB) 

tspae_tsne = TSNE(n_components=2).fit_transform(qB) 

#fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 6))

fig = plt.figure() 
ax1 = fig.add_subplot(111) 
ax1.scatter(tspae_tsne[1:59000,0], tspae_tsne[1:59000,1], c=colors, alpha=5, s=10)



grid = np.random.random((10,10))

s = qB.dot(qB.T)
s2 = binary_test.dot(binary_test.T)

fig = plt.figure() 
ax1 = fig.add_subplot(111) 

ax1.imshow(s2, extent=[0,10,0,10])
ax1.set_title('Default')
 

for i,ax in enumerate(axes.flatten()):
    ax.scatter(data[i][:,0], data[i][:,1], c=colors, alpha=0.5, s=1)
    ax.axis('off')
plt.show()