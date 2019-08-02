import os
from skimage import io
import torchvision.datasets.mnist as mnist
import torch
import codecs
import random
import numpy as np
from matplotlib import pyplot as plt



def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.LongTensor(images).view(-1, 28, 28)

def convert_to_img(train=True):
    X_all = torch.cat((train_set[0],test_set[0]),0).numpy()
    Y_all = torch.cat((train_set[1],test_set[1]),0).numpy()

    for i in range(10):
        data_path = save_root + str(i)
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)

        X_perclass = X_all[Y_all==i]
        Y_perclass = Y_all[Y_all==i]
        for j, (img,label) in enumerate(zip(X_perclass,Y_perclass)):
            img_path=data_path + '/' + str(j) + '.jpg'
            io.imsave(img_path,img)


    fp_test = open (save_root + 'test_img.txt', 'w')
    fp_val = open (save_root + 'val_img.txt', 'w')
    fp_train = open (save_root + 'train_img.txt', 'w')

    label_test = open (save_root + 'test_label.txt', 'w')
    label_val = open (save_root + 'val_label.txt', 'w')
    label_train = open (save_root + 'train_label.txt', 'w')

    index_all = np.random.permutation (7000)
    index_test = index_all [0:100]
    index_val = index_all [100:200]
    index_train = index_all [200:]

    for i in range(10):
        for j in range(len(index_test)):
            img_path = str (i) + '/' + str(index_test[j]) + '.jpg'
            fp_test.write (img_path + '\n')
            label_test.write (str(i)+'\n')

    for i in range(10):
        for j in range(len(index_val)):
            img_path = str (i) + '/' + str(index_val[j]) + '.jpg'
            fp_val.write (img_path + '\n')
            label_val.write (str(i)+'\n')

    for i in range(10):
        for j in range(len(index_train)):
            img_path = str (i) + '/' + str(index_train[j]) + '.jpg'
            fp_train.write (img_path + '\n')
            label_train.write (str(i)+'\n')



    # if(train):
    #     f=open(save_root+'train_label.txt','w')
    #     g = open (save_root + 'train_img.txt', 'w')
    #     data_path=save_root+'train/'
    #     if(not os.path.exists(data_path)):
    #         os.makedirs(data_path)
    #     for i, (img,label) in enumerate(zip(train_set[0],train_set[1])):
    #         img_path=data_path+str(i)+'.jpg'
    #         io.imsave(img_path,img.numpy())
    #         f.write(str(label)+'\n')
    #         g.write(img_path + '\n')
    #     f.close()
    #     g.close()
    # else:
    #     f = open(save_root + 'test_label.txt', 'w')
    #     g = open (save_root + 'test_img.txt', 'w')
    #     data_path = save_root + 'test/'
    #     if (not os.path.exists(data_path)):
    #         os.makedirs(data_path)
    #     for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
    #         img_path = data_path+ str(i) + '.jpg'
    #         io.imsave(img_path, img.numpy())
    #         f.write(str(label) + '\n')
    #         g.write (img_path + '\n')
    #     f.close()
    #     g.close()

if __name__=="__main__":
    root = "/data/dacheng/Datasets/fashion_mnist/"
    save_root="/data/dacheng/Datasets/fashion_mnist/"
    train_set = (
        read_image_file (os.path.join (root, 'train-images-idx3-ubyte')),
        mnist.read_label_file (os.path.join (root, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        read_image_file (os.path.join (root, 't10k-images-idx3-ubyte')),
        mnist.read_label_file (os.path.join (root, 't10k-labels-idx1-ubyte'))
    )
    print("training set :", train_set[0].size ())
    print("test set :", test_set[0].size ())

    convert_to_img(True)