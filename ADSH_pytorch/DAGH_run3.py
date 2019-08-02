
import pickle
import os
import argparse
import logging
import torch
import time
import sys
import random

import numpy as np
from numpy.linalg import *
import torch.optim as optim
import torchvision.transforms as transforms

from sklearn.cluster import KMeans
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
from scipy.optimize import fmin_l_bfgs_b, fmin, minimize

import utils.DC_dataset as dataset
import utils.DAGH_no_Ink_tr_meanF_loss as dl
import utils.cnn_model_DAGH as cnn_model
import utils.calc_hr as calc_hr
from utils.DC_load_data import load_dataset

parser = argparse.ArgumentParser (description="DAGH demo")
parser.add_argument ('--bits', default='8', type=str,
                     help='binary code length (default: 8,16,32,64)')
parser.add_argument ('--gpu', default='3', type=str,
                     help='selected gpu (default: 3)')
parser.add_argument ('--dataname', default='SUN20', type=str,
                     help='MirFlickr, NUSWIDE, COCO, CIFAR10, CIFAR100, Mnist, fashion_mnist, STL10')
parser.add_argument ('--arch', default='vgg11', type=str,
                     help='model name (default: resnet50,vgg11)')
parser.add_argument ('--max-iter', default=4, type=int,
                     help='maximum iteration (default: 50)')
parser.add_argument ('--epochs', default=2, type=int,
                     help='number of epochs (default: 1)')
parser.add_argument ('--batch-size', default=48, type=int,
                     help='batch size (default: 64)')
parser.add_argument ('--lambda-1', default='0.02', type=str,
                     help='hyper-parameter: oth-lambda-1 (default: 0.001,0.01,0.1,1,10,100)')
parser.add_argument ('--lambda-2', default='0.02', type=str,
                     help='hyper-parameter: bla-lambda-2 (default: 0.00001)')
parser.add_argument ('--lambda-3', default='0', type=str,
                     help='hyper-parameter: l1-lambda-3 (default: 0.00001)')
parser.add_argument ('--learning-rate', default=0.01, type=float,
                     help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument ('--num-anchor', default=300, type=int,
                     help='number of anchor: (default: 300)')

parser.add_argument ('--gamma', default=0.0001, type=int,
                     help='hyper-parameter: gamma (default: 0.0001)')
parser.add_argument ('--rho1', default=0.01, type=int,
                     help='hyper-parameter: rho1 (default: 0.001)')
parser.add_argument ('--rho2', default=0.01, type=int,
                     help='hyper-parameter: rho1 (default: 0.001)')
parser.add_argument ('--beta-1', default=0.001, type=int,
                     help='hyper-parameter: oth-lambda-1 (default: 0.001,0.01,0.1,1,10,100)')
parser.add_argument ('--beta-2', default=0.001, type=int,
                     help='hyper-parameter: bla-lambda-2 (default: 0.00001)')


def _logging():
    os.mkdir (logdir)
    global logger
    logfile = os.path.join (logdir, 'log.log')
    logger = logging.getLogger ('')
    logger.setLevel (logging.INFO)
    fh = logging.FileHandler (logfile)
    fh.setLevel (logging.INFO)
    ch = logging.StreamHandler ()
    ch.setLevel (logging.INFO)

    _format = logging.Formatter ("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter (_format)
    ch.setFormatter (_format)

    logger.addHandler (fh)
    logger.addHandler (ch)
    return


def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    record['param'] = {}
    return


def _save_record(record, filename):
    with open (filename, 'wb') as fp:
        pickle.dump (record, fp)
    return


def get_dist_graph(all_points, num_anchor=300):
    """
    get the cluster center as anchor by K-means++
    and calculate distance graph (n data points vs m anchors),
    :param all_points: n data points
    :param num_anchor:  m anchors, default = 300
    :return: distance graph n X m
    """
    # kmeans = KMeans (n_clusters=num_anchor, random_state=0, n_jobs=16, max_iter=50).fit_transform(all_points)
    # print ('dist graph done!')
    # return np.asarray(kmeans)
    ## smaple

    num_data = np.size (all_points, 0)
    sample_rate = 3000
    # sample_rate = num_data
    ind = random.sample (range (num_data), sample_rate)
    sample_points = all_points[ind, :]
    kmeans = KMeans (n_clusters=num_anchor, random_state=0, n_jobs=16, max_iter=50).fit (sample_points)
    km = kmeans.transform (all_points)
    print ('dist graph done!')
    return np.asarray (km)


def calc_Z(dist_graph, s=2):
    """
    calculate anchor graph (n data points vs m anchors),
    :param dist_graph: the distance matrix of n data points vs m anchors, n X m
    :param s: the number of nearest anchors, default = 2
    :return: anchor graph, n X m
    """
    # dist_graph = dist_graph * dist_graph
    n, m = dist_graph.shape
    Z = np.zeros ((n, m))
    val = np.zeros ((n, s))
    pos = np.zeros ((n, s), 'int')
    for i in range (0, s):
        val[:, i] = np.min (dist_graph, 1)
        pos[:, i] = np.argmin (dist_graph, 1)
        x = range (0, n)
        y = pos[:, i]
        dist_graph[x, y] = 1e60
    sigma = np.mean (val[:, s - 1] ** 0.5)

    '''
    normalization
    '''
    val = np.exp (-val / (1 / 1 * sigma ** 2))
    val = np.tile (1. / val.sum (1), (s, 1)).T * val

    for i in range (0, s):
        x = range (0, n)
        y = pos[:, i]
        Z[x, y] = val[:, i]

    print ('Z graph done!')
    return Z


def get_batch_gard(B, left, Z_T, batch_ind):
    """
    get the gradient of each corresponding batch : = BL[:,batch_ind]
    :param B: Binary codes of the all data points, k X n
    :param left: B * Z * inv_A, k X m
    :param Z: anchor graph, n X m
    Laplacian matrix L: I - Z * inv_A * Z^t
    :param batch_ind: batch size X k
    :return: the gradient of the batch points, batch size X k
    """
    grad = B[:, batch_ind] - np.dot (left, Z_T[:, batch_ind])
    return grad.transpose ()


def B_step(F, Z, inv_A):
    """
    Update B : F^t * W
    :param F: output of network as the real-valued embeddings n X k
    :param Z: anchor graph, n X m
    Affinity matrix W: Z * inv_A * Z^t
    :return: the updated B, k X n
    """
    Z_T = Z.transpose ()  # m X n
    temp = np.dot (F.transpose (), np.dot (Z, inv_A))  # k X m
    B = np.dot (temp, Z_T)  # k X n
    print ('B step done!')
    return np.sign (B)


def B4_step(B, F, Z, inv_A, lambda_1, lambda_2, rho1, rho2, gamma):
    """
    Update B : F^t * W
    :param F: output of network as the real-valued embeddings n X k
    :param Z: anchor graph, n X m
    Affinity matrix W: Z * inv_A * Z^t
    :return: the updated B, k X n
    """
    bit, num_train = B.shape
    ini_B = B
    Y1 = np.random.rand (bit, num_train)
    Y2 = np.random.rand (bit, num_train)
    loss_old = 0

    Z1 = ini_B
    Z2 = ini_B

    Z_T = Z.transpose ()
    nI_K = num_train * np.eye (bit, bit)

    def func_BB(B, Z, inv_A, G, nI_K, lambda_1, lambda_2, rho1, rho2):
        bit, num_train = G.shape
        B = np.reshape (B, (bit, num_train))
        temp = np.dot (B, np.dot (Z, inv_A))  # k X m
        temp2 = np.dot (temp, Z_T)  # k X n
        BAB = np.dot (temp2, B.transpose ())  # k X k
        loss_BLB = np.trace (np.dot (B, B.transpose ()) - BAB)

        # reg_loss = (B - F.transpose()) ** 2
        oth_loss = lambda_1 * ((np.dot (B, B.transpose ()) - nI_K) ** 2)
        bla_loss = lambda_2 * (B.sum (1) ** 2)

        rho_loss = ((rho1 + rho2) / 2) * (B ** 2).sum () + np.trace (B.dot (G.transpose ()))
        loss = rho_loss + loss_BLB + 0.25 * oth_loss.sum () + 0.5 * bla_loss.sum ()
        return loss

    def grad_BB(B, Z, inv_A, G, nI_K, lambda_1, lambda_2, rho1, rho2):
        bit, num_train = G.shape
        B = np.reshape (B, (bit, num_train))
        rho_grad = (rho1 + rho2) * B
        temp = np.dot (B, np.dot (Z, inv_A))  # k X m
        temp2 = np.dot (temp, Z_T)  # k X n
        L_grad = 2 * (B - temp2)
        oth_grad = lambda_1 * np.dot ((np.dot (B, B.transpose ()) - nI_K), B)

        bla_grad = lambda_2 * np.dot (B, np.ones ((num_train, num_train)))

        grad_all = rho_grad + L_grad + oth_grad + bla_grad + G
        return grad_all.flatten ()

    def func_B(B, F, Z, inv_A, G, nI_K, lambda_1, lambda_2, rho1, rho2):
        num_train, bit = F.shape
        B = np.reshape (B, (bit, num_train))
        temp = np.dot (B, np.dot (Z, inv_A))  # k X m
        temp2 = np.dot (temp, Z_T)  # k X n
        BAF = np.dot (temp2, F)  # k X k
        loss_BLF = np.trace (np.dot (B, F) - BAF)

        reg_loss = (B - F.transpose ()) ** 2
        oth_loss = lambda_1 * ((np.dot (B, B.transpose ()) - nI_K) ** 2)
        bla_loss = lambda_2 * (B.sum (1) ** 2)

        rho_loss = ((rho1 + rho2) / 2) * (B ** 2).sum () + np.trace (B.dot (G.transpose ()))
        loss = rho_loss + loss_BLF + 0.5 * reg_loss.sum () + 0.25 * oth_loss.sum () + 0.5 * bla_loss.sum ()
        return loss

    def grad_B(B, F, Z, inv_A, G, nI_K, lambda_1, lambda_2, rho1, rho2):
        num_train, bit = F.shape
        B = np.reshape (B, (bit, num_train))
        rho_grad = (rho1 + rho2 + 1) * B
        temp = np.dot (F.transpose (), np.dot (Z, inv_A))  # k X m
        L_grad = np.dot (temp, Z_T)  # k X n
        oth_grad = lambda_1 * np.dot ((np.dot (B, B.transpose ()) - nI_K), B)

        bla_grad = lambda_2 * np.dot (B, np.ones ((num_train, num_train)))

        grad_all = rho_grad - L_grad + oth_grad + bla_grad + G
        return grad_all.flatten ()

    for iter_B in range (50):
        print ('B_iteration %3d\n' % iter_B)
        G = Y1 + Y2 - rho1 * Z1 - rho2 * Z2
        res = minimize (func_B, B,
                        args=(F, Z, inv_A, G, nI_K, lambda_1, lambda_2, rho1, rho2),
                        method='L-BFGS-B',
                        jac=grad_B,
                        options={'maxiter': 50, 'maxfun': 100, 'disp': True})

        # res = minimize (func_BB, B,
        #             args=(Z, inv_A, G, nI_K, lambda_1, lambda_2, rho1, rho2),
        #             method='L-BFGS-B',
        #             jac=grad_BB,
        #             options={'maxiter':50,'maxfun':100, 'disp': False})
        # 'ftol':1e-4,
        Bk = np.reshape (res.x, [bit, num_train])
        count = (Bk > 0).sum ()

        if iter_B == 0:
            Bkk = ini_B
        print ('+1, -1: %.2f%%\n' % (float (count) / num_train / bit * 100))
        print ('res(init_B and Bk): %d\n' % ((np.sign (Bk) - Bkk)).sum ())

        Bkk = np.sign (Bk)

        Z1_k = H1_step (Bk, Y1, rho1)
        Z2_k = H2_step (Bk, Y2, rho2)

        Y1_k, Y2_k = Y_step (Y1, Y2, Z1_k, Z2_k, Bk, rho1, rho2, gamma)

        B = Bk
        Z1 = Z1_k
        Z2 = Z2_k
        Y1 = Y1_k
        Y2 = Y2_k

        loss = calc_all_loss (B, F, Z, inv_A, Z1, Z2, Y1, Y2, rho1, rho2, lambda_1, lambda_2)
        res_error = (loss - loss_old) / loss_old
        loss_old = loss
        print ('loss is %.4f, residual error is %.5f\n' % (loss, res_error))

        if (np.abs (res_error) <= 1e-4):
            break

    return np.sign (B)


def B3_step(B, F, Z, inv_A, lambda_1, lambda_2, rho1, rho2, gamma):
    """
    Update B : F^t * W
    :param F: output of network as the real-valued embeddings n X k
    :param Z: anchor graph, n X m
    Affinity matrix W: Z * inv_A * Z^t
    :return: the updated B, k X n
    """
    bit, num_train = B.shape
    ini_B = B
    Y1 = Variable (torch.randn (bit, num_train))
    Y2 = Variable (torch.randn (bit, num_train))
    loss_old = 0

    B = Variable (torch.from_numpy (ini_B).type (torch.FloatTensor), requires_grad=True)
    Z1 = Variable (torch.from_numpy (ini_B).type (torch.FloatTensor))
    Z2 = Variable (torch.from_numpy (ini_B).type (torch.FloatTensor))

    optimizer_B = optim.LBFGS ([B], lr=0.1)
    F = Variable (torch.from_numpy (F).type (torch.FloatTensor))
    inv_A = Variable (torch.from_numpy (inv_A).type (torch.FloatTensor))
    Z = Variable (torch.from_numpy (Z).type (torch.FloatTensor))
    Z_T = Z.t ()  # m X n
    nI_K = Variable (num_train * torch.eye (bit, bit))

    for iter_B in range (20):
        def closure():
            optimizer_B.zero_grad ()

            temp = torch.mm (B, torch.mm (Z, inv_A))  # k X m
            temp2 = torch.mm (temp, Z_T)  # k X n
            BAF = torch.mm (temp2, F)  # k X k
            loss_BLF = torch.trace (torch.mm (B, F) - BAF)

            reg_loss = (B - F.t ()) ** 2
            oth_loss = lambda_1 * ((torch.mm (B, B) - nI_K) ** 2)
            bla_loss = lambda_2 * (B.sum (0) ** 2)

            G = Y1 + Y2 - rho1 * Z1 - rho2 * Z2

            rho_loss = ((rho1 + rho2) / 2) * (B ** 2).sum () + torch.trace (B.mm (G.t ()))
            loss = (rho_loss + loss_BLF + 0.5 * (reg_loss.sum () + oth_loss.sum () + bla_loss.sum ())) / num_train
            loss.backward ()
            return loss

        optimizer_B.step (closure)
        count = (B.data.numpy () > 0).sum ()
        print ('+1, -1: %.2f%%\n' % (float (count) / num_train / bit * 100))
        print ('res(init_B and Bk): %d\n' % ((np.sign (B.data.numpy ()) - ini_B)).sum ())

        Z1_k = H1_step (B.data.numpy (), Y1.data.numpy (), rho1)
        Z2_k = H2_step (B.data.numpy (), Y2.data.numpy (), rho2)

        Y1_k, Y2_k = Y_step (Y1.data.numpy (), Y2.data.numpy (), Z1_k, Z2_k, B.data.numpy (), rho1, rho2, gamma)

        Z1 = Variable (torch.from_numpy (Z1_k).type (torch.FloatTensor))
        Z2 = Variable (torch.from_numpy (Z2_k).type (torch.FloatTensor))
        Y1 = Variable (torch.from_numpy (Y1_k).type (torch.FloatTensor))
        Y2 = Variable (torch.from_numpy (Y2_k).type (torch.FloatTensor))

        loss = calc_all_loss (B.data.numpy (), F.data.numpy (), Z.data.numpy (), inv_A.data.numpy (), Z1.data.numpy (),
                              Z2.data.numpy (), Y1.data.numpy (), Y2.data.numpy (), rho1, rho2, lambda_1, lambda_2)
        res_error = (loss - loss_old) / loss_old
        loss_old = loss
        print ('loss is %.4f, residual error is %.5f\n' % (loss, res_error))
        if (np.abs (res_error) <= 1e-4):
            break

    return np.sign (B.data.numpy ())


def H1_step(B, Y1, rho1):
    theta1 = B + 1 / rho1 * Y1
    theta1[theta1 > 1] = 1
    theta1[theta1 < -1] = -1
    Z1_k = theta1
    return Z1_k


def H2_step(B, Y2, rho2):
    bit, num_train = B.shape
    theta2 = B + 1 / rho2 * Y2
    norm_B = np.linalg.norm (theta2, 'fro')
    theta2 = np.sqrt (num_train * bit) * theta2 / norm_B
    Z2_k = theta2
    return Z2_k


def Y_step(Y1, Y2, Z1_k, Z2_k, Bk, rho1, rho2, gamma):
    Y1_k = Y1 + gamma * rho1 * (Bk - Z1_k)
    Y2_k = Y2 + gamma * rho2 * (Bk - Z2_k)
    return Y1_k, Y2_k


def calc_all_loss(B, F, Z, inv_A, Z1, Z2, Y1, Y2, rho1, rho2, lambda_1, lambda_2):
    """
    Calculate loss: Tr(BLF^t) = Tr(B * Z * inv_A * Z^t * F^t)
    :param F: output of network n X k
    :param B: binary codes k X n
    :param Z: anchor graph, n X m
    :return: loss: trace(BLF)
    """
    bit, num_train = B.shape
    Z_T = Z.transpose ()  # m X n
    temp = np.dot (B, np.dot (Z, inv_A))  # k X m
    temp2 = np.dot (temp, Z_T)  # k X n
    BAF = np.dot (temp2, F)  # k X k
    Tr_BLF = np.trace (np.dot (B, F) - BAF)

    nI_K = num_train * np.eye (bit, bit)
    res1 = B - Z1
    res2 = B - Z2

    reg_loss = (B - F.transpose ()) ** 2
    oth_loss = lambda_1 * ((np.dot (B, B.transpose ()) - nI_K) ** 2) / 4
    bla_loss = lambda_2 * (B.sum (1) ** 2) / 2
    z1_loss = rho1 * ((res1 ** 2).sum ()) / 2
    z2_loss = rho2 * ((res2 ** 2).sum ()) / 2
    y1_loss = np.trace (np.dot (res1, Y1.transpose ()))
    y2_loss = np.trace (np.dot (res2, Y2.transpose ()))

    loss = Tr_BLF + 0.5 * reg_loss.sum () + oth_loss.sum () + bla_loss.sum () + z1_loss + z2_loss + y1_loss + y2_loss

    print ('Tr_BLF:' + str (Tr_BLF + 0.5 * reg_loss.sum ()))
    print ('loss all done!')
    return loss


def encoding_onehot(target, nclasses=10):
    target_onehot = torch.FloatTensor (target.size (0), nclasses)
    target_onehot.zero_ ()
    target_onehot.scatter_ (1, target.view (-1, 1), 1)
    return target_onehot


# def _dataset(dataname):
#     # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     transformations = transforms.Compose([
#         transforms.Scale(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize
#     ])
#
#     rootpath = os.path.join('/data/dacheng/Datasets/', dataname)
#
#     if dataname=='NUSWIDE':
#         dset_database = dataset.NUSWIDE ('train_img.txt', 'train_label.txt', transformations)
#         dset_test = dataset.NUSWIDE ('test_img.txt', 'test_label.txt', transformations)
#     elif dataname=='MirFlickr':
#         dset_database = dataset.MirFlickr ('train_img.txt', 'train_label.txt', transformations)
#         dset_test = dataset.MirFlickr ('test_img.txt', 'test_label.txt', transformations)
#     elif dataname =='COCO':
#         dset_database = dataset.COCO ('train_img.txt', 'train_label.txt', transformations)
#         dset_test = dataset.COCO ('test_img.txt', 'test_label.txt', transformations)
#     elif dataname == 'CIFAR10':
#         dset_database = dataset.CIFAR10 ('train_img.txt', 'train_label.txt', transformations)
#         dset_test = dataset.CIFAR10 ('test_img.txt', 'test_label.txt', transformations)
#     elif dataname == 'MNIST':
#         dset_database = dataset.MNIST (True, transformations)
#         dset_test = dataset.MNIST (False, transformations)
#
#     num_database, num_test = len (dset_database), len (dset_test)
#
#     def load_label(filename, DATA_DIR):
#         path = os.path.join(DATA_DIR, filename)
#         fp = open(path, 'r')
#         labels = [x.strip() for x in fp]
#         fp.close()
#         return torch.LongTensor(list(map(int, labels)))
#
#     def DC_load_label(filename, DATA_DIR):
#         path = os.path.join(DATA_DIR, filename)
#         label = np.loadtxt (path, dtype=np.int64)
#         return torch.LongTensor(label)
#
#     def load_label_CIFAR100(root, train=True):
#         base_folder = 'cifar-100-python'
#         train_list = [
#             ['train', '16019d7e3df5f24257cddd939b257f8d'],
#         ]
#
#         test_list = [
#             ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
#         ]
#
#         root = os.path.expanduser(root)
#         train = train  # training set or test set
#
#         # now load the picked numpy arrays
#         if train:
#             train_data = []
#             train_labels = []
#             for fentry in train_list:
#                 f = fentry[0]
#                 file = os.path.join(root, base_folder, f)
#                 fo = open(file, 'rb')
#                 if sys.version_info[0] == 2:
#                     entry = pickle.load(fo)
#                 else:
#                     entry = pickle.load(fo, encoding='latin1')
#                 train_data.append(entry['data'])
#                 if 'labels' in entry:
#                     train_labels += entry['labels']
#                 else:
#                     train_labels += entry['fine_labels']
#                 fo.close()
#
#         else:
#             f = test_list[0][0]
#             file = os.path.join(root, base_folder, f)
#             fo = open(file, 'rb')
#             if sys.version_info[0] == 2:
#                 entry = pickle.load(fo)
#             else:
#                 entry = pickle.load(fo, encoding='latin1')
#             test_data = entry['data']
#             if 'labels' in entry:
#                 test_labels = entry['labels']
#             else:
#                 test_labels = entry['fine_labels']
#             fo.close()
#
#         if train:
#             target = train_labels
#         else:
#             target = test_labels
#         return torch.LongTensor(list(map(int, target)))
#
#
#     def DC_load_label_MNIST(filename, root):
#         _, labels = torch.load (os.path.join (root, filename))
#         return torch.LongTensor(labels)
#
#     if dataname=='CIFAR10':
#         testlabels_ = load_label('test_label.txt', rootpath)
#         databaselabels_ = load_label('train_label.txt', rootpath)
#         testlabels = encoding_onehot(testlabels_)
#         databaselabels = encoding_onehot(databaselabels_)
#     elif dataname == 'MNIST':
#         databaselabels_ = DC_load_label_MNIST ('training.pt', root='/home/dacheng/PycharmProjects/ADSH_pytorch/data/processed/')
#         testlabels_ = DC_load_label_MNIST ('test.pt', root='/home/dacheng/PycharmProjects/ADSH_pytorch/data/processed/')
#         testlabels = encoding_onehot(testlabels_)
#         databaselabels = encoding_onehot(databaselabels_)
#     elif dataname == 'CIFAR100':
#         testlabels_ = load_label_CIFAR100('/data/dacheng/Datasets/CIFAR100/', train=False)
#         databaselabels_ = load_label_CIFAR100('/data/dacheng/Datasets/CIFAR100/', train=True)
#         testlabels = encoding_onehot(testlabels_, nclasses=100)
#         databaselabels = encoding_onehot(databaselabels_, nclasses=100)
#     else:
#         databaselabels = DC_load_label('train_label.txt', rootpath)
#         testlabels = DC_load_label('test_label.txt', rootpath)
#
#     dsets = (dset_database, dset_test)
#     nums = (num_database, num_test)
#     labels = (databaselabels, testlabels)
#
#     return nums, dsets, labels

def calc_loss(B, F, Z, inv_A, lambda_1, lambda_2, lambda_3, code_length):
    """
    Calculate loss: Tr(BLF^t) = Tr(B * Z * inv_A * Z^t * F^t)
    :param F: output of network n X k
    :param B: binary codes k X n
    :param Z: anchor graph, n X m
    :return: loss: trace(BLF)
    """
    Z_T = Z.transpose ()  # m X n
    temp = np.dot (B, np.dot (Z, inv_A))  # k X m
    temp2 = np.dot (temp, Z_T)  # k X n
    BAF = np.dot (temp2, F)  # k X k
    Tr_BLF = np.trace (np.dot (B, F) - BAF)

    num_train = np.size (B, 1)
    # nI_K =  num_train * np.eye (code_length, code_length)
    nI_K = np.eye (code_length, code_length)

    one_vectors = np.ones ((num_train, code_length))
    reg_loss = (B - F.transpose ()) ** 2
    # oth_loss = (np.dot(F.transpose(), F) - nI_K) ** 2
    oth_loss = (np.dot (F.transpose (), F) / num_train - nI_K) ** 2

    mean_F = F.mean (0).reshape (1, code_length)
    var_loss = (F - mean_F) ** 2

    bla_loss = (F.sum (0)) ** 2
    susb_loss = np.abs (F) - one_vectors
    L1_loss = np.abs (susb_loss)
    loss = (Tr_BLF + 0.5 * (reg_loss.sum () + lambda_2 * bla_loss.sum () + lambda_3 * L1_loss.sum ())) / (code_length * num_train) + 0.25 * lambda_1 * oth_loss.sum () / (code_length)

    print ('Tr_BLF:' + str (Tr_BLF / code_length / num_train))
    print ('reg_loss:' + str (reg_loss.sum () / code_length / num_train))
    print ('oth_loss:' + str (lambda_1 * oth_loss.sum () / code_length))
    print ('bla_loss:' + str (lambda_2 * bla_loss.sum () / code_length / num_train))
    print ('L1_loss:' + str (lambda_3 * L1_loss.sum () / code_length / num_train))
    print ('var_loss:' + str (lambda_2 * var_loss.sum () / code_length / num_train))
    print ('loss done!')
    return loss


def encode(model, data_loader, num_data, bit):
    B = np.zeros ([num_data, bit], dtype=np.float32)
    for iter, data in enumerate (data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable (data_input.cuda (), volatile=True)
        output = model (data_input)
        B[data_ind.numpy (), :] = torch.sign (output[1].cpu ().data).numpy ()
    return B


def get_F(model, data_loader, num_data, bit):
    B = np.zeros ([num_data, bit], dtype=np.float32)
    for iter, data in enumerate (data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable (data_input.cuda (), volatile=True)
        output = model (data_input)
        B[data_ind.numpy (), :] = output[1].cpu ().data.numpy ()
    return B


def get_fearture(model, data_loader, num_data, bit):
    Features = np.zeros ([num_data, 4096], dtype=np.float32)
    for iter, data in enumerate (data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable (data_input.cuda (), volatile=True)
        output = model (data_input)
        Features[data_ind.numpy (), :] = output[0].cpu ().data.numpy ()
    return Features


def adjusting_learning_rate(optimizer, iter):
    if ((iter % 2) == 0) & (iter !=0):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 5
        print ('learning rate is adjusted!')


def cal_BLF(B, Z, inv_A, F):
    code_length, num_train = B.shape
    Z_T = Z.transpose ()  # m X n
    temp = np.dot (B, np.dot (Z, inv_A))  # k X m
    temp2 = np.dot (temp, Z_T)  # k X n
    BAF = np.dot (temp2, F)  # k X k
    Tr_BLF = np.trace (np.dot (B, F) - BAF) / (code_length * num_train)

    return Tr_BLF


def DAGH_algo(code_length, dataname):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed (0)
    torch.cuda.manual_seed (0)
    # code_length=8

    '''
    parameter setting
    '''
    max_iter = opt.max_iter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    weight_decay = 5 * 10 ** -4
    num_anchor = opt.num_anchor
    lambda_1 = float (opt.lambda_1)
    lambda_2 = float (opt.lambda_2)
    lambda_3 = float (opt.lambda_3)
    rho1 = opt.rho1
    rho2 = opt.rho2
    beta1 = opt.beta_1
    beta2 = opt.beta_2
    gamma = opt.gamma

    record['param']['opt'] = opt
    record['param']['description'] = '[Comment: learning rate decay]'
    logger.info (opt)
    logger.info (code_length)
    logger.info (record['param']['description'])

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = load_dataset (dataname)
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    '''
    model construction
    '''
    model = cnn_model.CNNNet (opt.arch, code_length)
    model.cuda ()
    cudnn.benchmark = True
    DAGH_loss = dl.DAGHLoss (lambda_1, lambda_2, lambda_3, code_length)
    L1_criterion = nn.L1Loss ()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.SGD (model.parameters (), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)  ####
    # optimizer = optim.RMSprop (model.parameters (), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.RMSprop (model.parameters (), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    # optimizer = optim.Adadelta (model.parameters (), weight_decay=weight_decay)
    # optimizer = optim.Adam (model.parameters ())

    B = np.sign (np.random.randn (code_length, num_database))

    model.train ()
    for iter in range (max_iter):
        iter_time = time.time ()

        # adjusting_learning_rate (optimizer, learning_rate, False)

        trainloader = DataLoader (dset_database, batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)

        if iter == 0:
            '''
            initialize the feature of all images to build dist graph
            '''
            ini_Features = np.zeros ((num_database, 4096), dtype=np.float)
            ini_F = np.zeros ((num_database, code_length), dtype=np.float)
            for iteration, (train_input, train_label, batch_ind) in enumerate (trainloader):
                train_input = Variable (train_input.cuda ())
                output = model (train_input)
                ini_Features[batch_ind, :] = output[0].cpu ().data.numpy ()
                ini_F[batch_ind, :] = output[1].cpu ().data.numpy ()
            print ('initialization dist graph forward done!')
            dist_graph = get_dist_graph (ini_Features, num_anchor)
            # dist_graph = np.random.rand(num_database,num_anchor)
            # bf = np.sign(ini_F)
            Z = calc_Z (dist_graph)

            # B = np.sign (ini_F.transpose ())

        # elif ((iter % 3) == 0) | (iter == max_iter - 1):
        else:
            dist_graph = get_dist_graph (Features, num_anchor)
            Z = calc_Z (dist_graph)
            print ('reset dist graph!')

        inv_A = inv (np.diag (Z.sum (0)))  # m X m
        Z_T = Z.transpose ()  # m X n
        left = np.dot (B, np.dot (Z, inv_A))  # k X m

        if iter == 0:
            loss_ini = calc_loss (B, ini_F, Z, inv_A, lambda_1, lambda_2, lambda_3, code_length)
            print('ini_loss' + str (loss_ini))

        '''
        learning deep neural network: feature learning
        '''

        # B = B4_step (B, ini_F, Z, inv_A, lambda_1, lambda_2, rho1, rho2, gamma)

        for epoch in range (epochs):
            F = np.zeros ((num_database, code_length), dtype=np.float)
            running_loss = 0.0
            for iteration, (train_input, train_label, batch_ind) in enumerate (trainloader):
                train_input = Variable (train_input.cuda ())

                output = model (train_input)
                F[batch_ind, :] = output[1].cpu ().data.numpy ()

                batch_grad = get_batch_gard (B, left, Z_T, batch_ind) / (code_length * batch_size)
                batch_grad = Variable (torch.from_numpy (batch_grad).type (torch.FloatTensor).cuda ())
                optimizer.zero_grad ()
                output[1].backward (batch_grad, retain_graph=True)
                # output[1].backward (batch_grad)

                B_cuda = Variable (torch.from_numpy (B[:, batch_ind]).type (torch.FloatTensor).cuda ())
                other_loss = DAGH_loss (output[1].t (), B_cuda)
                one_vectors = Variable (torch.ones (output[1].size ()).cuda ())
                L1_loss = L1_criterion (torch.abs (output[1]), one_vectors)
                # L2_loss = L2_criterion (output[1],B_cuda.t())
                All_loss = other_loss + lambda_3 * L1_loss / code_length
                All_loss.backward ()
                optimizer.step ()

                running_loss += All_loss.data[0]
                if (iteration % 50) == 49:
                    # print ('iteration:' + str (iteration))
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, iteration + 1, running_loss / 50))
                    running_loss = 0.0

            Tr_BLF = cal_BLF (B, Z, inv_A, F)
            print('[%d] Tr_BLF loss: %.5f' % (epoch + 1, Tr_BLF))
        adjusting_learning_rate (optimizer, iter)

        trainloader2 = DataLoader (dset_database, batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=4)
        F = get_F (model, trainloader2, num_database, code_length)
        Features = get_fearture (model, trainloader2, num_database, code_length)

        '''
        learning binary codes: discrete coding
        '''
        #B_new = np.sign (F).transpose()
        #B_random = np.sign (np.random.randn (code_length, num_database))

        # F = np.random.randn (num_datasbase, 12)
        loss_before = calc_loss (B, F, Z, inv_A, lambda_1, lambda_2, lambda_3, code_length)

        B = B_step (F, Z, inv_A)
        #B = B4_step (B_random, np.sign (F), Z, inv_A, beta1, beta2, rho1, rho2, gamma)
        iter_time = time.time () - iter_time
        loss_ = calc_loss (B, F, Z, inv_A, lambda_1, lambda_2, lambda_3, code_length)

        logger.info ('[Iteration: %3d/%3d][Train Loss: before:%.4f, after:%.4f]', iter, max_iter, loss_before, loss_)
        record['train loss'].append (loss_)
        record['iter time'].append (iter_time)

    '''
    training procedure finishes, evaluation
    '''
    model.eval ()
    retrievalloader = DataLoader (dset_database, batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4)
    testloader = DataLoader (dset_test, batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)
    qB = encode (model, testloader, num_test, code_length)
    rB_sy = encode (model, retrievalloader, num_database, code_length)
    rB_asy = B.transpose ()

    topKs = np.arange (1, 500, 50)
    top_ndcg = 100
    # map = calc_hr.calc_map (qB, rB, test_labels.numpy (), database_labels.numpy ())
    # top_map = calc_hr.calc_topMap (qB, rB, test_labels.numpy (), database_labels.numpy (), 2000)
    Pres_asy = calc_hr.calc_topk_pres (qB, rB_asy, test_labels.numpy (), database_labels.numpy (), topKs)
    ndcg_asy = calc_hr.cal_ndcg_k (qB, rB_asy, test_labels.numpy (), database_labels.numpy (), top_ndcg)
    Pres_sy = calc_hr.calc_topk_pres (qB, rB_sy, test_labels.numpy (), database_labels.numpy (), topKs)
    ndcg_sy = calc_hr.cal_ndcg_k (qB, rB_sy, test_labels.numpy (), database_labels.numpy (), top_ndcg)

    map_sy = calc_hr.calc_map (qB, rB_sy, test_labels.numpy (), database_labels.numpy ())
    map_asy = calc_hr.calc_map (qB, rB_asy, test_labels.numpy (), database_labels.numpy ())
    top_map_sy = calc_hr.calc_topMap (qB, rB_sy, test_labels.numpy (), database_labels.numpy (), 2000)
    top_map_asy = calc_hr.calc_topMap (qB, rB_asy, test_labels.numpy (), database_labels.numpy (), 2000)

    logger.info ('[lambda_1: %.4f]', lambda_1)
    logger.info ('[lambda_2: %.4f]', lambda_2)
    logger.info ('[lambda_3: %.4f]', lambda_3)
    logger.info ('[Evaluation: mAP_sy: %.4f]', map_sy)
    logger.info ('[Evaluation: mAP_asy: %.4f]', map_asy)
    logger.info ('[Evaluation: topK_mAP_sy: %.4f]', top_map_sy)
    logger.info ('[Evaluation: topK_mAP_asy: %.4f]', top_map_asy)
    logger.info ('[Evaluation: Pres_sy: %.4f]', Pres_sy[0])
    print Pres_sy
    logger.info ('[Evaluation: Pres_asy: %.4f]', Pres_asy[0])
    print Pres_asy
    logger.info ('[Evaluation: topK_ndcg_sy: %.4f]', ndcg_sy)
    logger.info ('[Evaluation: topK_ndcg_asy: %.4f]', ndcg_asy)
    record['rB_sy'] = rB_sy
    record['rB_asy'] = rB_asy
    record['qB'] = qB
    record['map_sy'] = map_sy
    record['map_asy'] = map_asy
    record['topK_map_sy'] = top_map_sy
    record['topK_map_asy'] = top_map_asy
    record['topK_ndcg_sy'] = ndcg_sy
    record['topK_ndcg_asy'] = ndcg_asy
    record['Pres_sy'] = Pres_sy
    record['Pres_asy'] = Pres_asy
    record['F'] = F
    filename = os.path.join (logdir, str (code_length) + 'bits-record.pkl')

    _save_record (record, filename)
    return top_map_sy


if __name__ == "__main__":
    global opt, logdir
    opt = parser.parse_args ()
    logdir = '-'.join (['log/run3', opt.dataname, datetime.now ().strftime ("%y-%m-%d-%H-%M-%S")])
    _logging ()
    _record ()
    bits = [int (bit) for bit in opt.bits.split (',')]
    for bit in bits:
        DAGH_algo (bit, opt.dataname)
    # lambda_1s = [ float(lambda_1) for lambda_1 in opt.lambda_1.split (',')]
    # lambda_2s = [int (lambda_2) for lambda_2 in opt.lambda_2.split (',')]
    # lambda_3s = [float (lambda_3) for lambda_3 in opt.lambda_3.split (',')]
    # topmap_lambda = np.zeros ((len(lambda_1s),len(lambda_3s)))
    # for i, lambda_1 in enumerate(lambda_1s):
    #     for j, lambda_3 in enumerate(lambda_3s):
    #
    #         topmap = DAGH_algo(lambda_1,lambda_3,opt.dataname)
    #
    #         topmap_lambda[i,j] = topmap
    # record['topmap_lambda'] = topmap_lambda
