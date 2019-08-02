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

import utils.DC_dataset as dataset
import utils.DAGH_no_Ink_tr_meanF_loss as dl
import utils.cnn_model_DAGH as cnn_model
import utils.calc_hr as calc_hr
from utils.DC_load_data import load_dataset

parser = argparse.ArgumentParser(description="DAGH demo")
parser.add_argument('--bits', default='8', type=str,
                    help='binary code length (default: 8,16,32,64)')
parser.add_argument('--gpu', default='2', type=str,
                    help='selected gpu (default: 3)')
parser.add_argument('--dataname', default='NUSWIDE', type=str,
                    help='MirFlickr, NUSWIDE, COCO, CIFAR10, CIFAR100, Mnist, fashion_mnist')
parser.add_argument('--arch', default='vgg11', type=str,
                    help='model name (default: resnet50,vgg11)')
parser.add_argument('--max-iter', default=6, type=int,
                    help='maximum iteration (default: 50)')
parser.add_argument('--epochs', default=4, type=int,
                    help='number of epochs (default: 1)')
parser.add_argument('--batch-size', default=48, type=int,
                    help='batch size (default: 64)')
parser.add_argument('--lambda-1', default='0.01', type=str,
                    help='hyper-parameter: oth-lambda-1 (default: 0.001,0.01,0.1,1,10,100)')
parser.add_argument('--lambda-2', default='0.01', type=str,
                    help='hyper-parameter: bla-lambda-2 (default: 0.00001)')
parser.add_argument('--lambda-3', default='0.01', type=str,
                    help='hyper-parameter: l1-lambda-3 (default: 0.00001)')
parser.add_argument('--learning-rate', default=0.01, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--num-anchor', default=300, type=int,
                    help='number of anchor: (default: 300)')

def _logging():
    os.mkdir(logdir)
    global logger
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    record['param'] = {}
    return

def _save_record(record, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(record, fp)
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
    sample_rate = 3000
    num_data = np.size(all_points,0)
    ind = random.sample(range(num_data),sample_rate)
    sample_points = all_points[ind,:]
    kmeans = KMeans (n_clusters=num_anchor, random_state=0, n_jobs=16, max_iter=50).fit(sample_points)
    km = kmeans.transform(all_points)
    print ('dist graph done!')
    return np.asarray(km)

def calc_Z(dist_graph,s=2):
    """
    calculate anchor graph (n data points vs m anchors),
    :param dist_graph: the distance matrix of n data points vs m anchors, n X m
    :param s: the number of nearest anchors, default = 2
    :return: anchor graph, n X m
    """
    #dist_graph = dist_graph * dist_graph
    n,m = dist_graph.shape
    Z = np.zeros((n,m))
    val = np.zeros((n,s))
    pos = np.zeros((n,s), 'int')
    for i in range(0,s):
        val[:,i] = np.min(dist_graph,1)
        pos[:,i] = np.argmin(dist_graph,1)
        x = range(0,n)
        y = pos[:,i]
        dist_graph[x,y] = 1e60
    sigma = np.mean(val[:,s-1] ** 0.5)

    '''
    normalization
    '''
    val = np.exp(-val / (1 / 1 * sigma ** 2))
    val = np.tile(1./val.sum(1),(s,1)).T * val

    for i in range(0, s):
        x = range (0, n)
        y = pos[:, i]
        Z[x,y] = val[:,i]

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
    grad = B[:,batch_ind] - np.dot(left, Z_T[:,batch_ind])
    return grad.transpose()

def B_step(F, Z, inv_A):
    """
    Update B : F^t * W
    :param F: output of network as the real-valued embeddings n X k
    :param Z: anchor graph, n X m
    Affinity matrix W: Z * inv_A * Z^t
    :return: the updated B, k X n
    """
    Z_T = Z.transpose() # m X n
    temp = np.dot(F.transpose(), np.dot(Z, inv_A)) # k X m
    B = np.dot(temp, Z_T) # k X n
    print ('B step done!')
    return np.sign(B)

def calc_loss(B, F, Z, inv_A, lambda_1, lambda_2, lambda_3, code_length):
    """
    Calculate loss: Tr(BLF^t) = Tr(B * Z * inv_A * Z^t * F^t)
    :param F: output of network n X k
    :param B: binary codes k X n
    :param Z: anchor graph, n X m
    :return: loss: trace(BLF)
    """
    Z_T = Z.transpose ()  # m X n
    temp = np.dot (B, np.dot (Z, inv_A)) # k X m
    temp2 = np.dot (temp, Z_T) # k X n
    BAF = np.dot (temp2, F) # k X k
    Tr_BLF = np.trace(np.dot(B, F) - BAF)

    num_train = np.size (B,1)
    # nI_K =  num_train * np.eye (code_length, code_length)
    nI_K = np.eye (code_length, code_length)

    one_vectors = np.ones ((num_train, code_length))
    reg_loss = (B - F.transpose()) ** 2
    # oth_loss = (np.dot(F.transpose(), F) - nI_K) ** 2
    oth_loss = (np.dot (F.transpose (), F)/num_train - nI_K) ** 2

    mean_F = F.mean (0).reshape (1, 8)
    var_loss = (F - mean_F) ** 2

    bla_loss = (F.sum (0)) ** 2
    susb_loss = np.abs (F) - one_vectors
    L1_loss =  np.abs (susb_loss)
    loss = (Tr_BLF + 0.5 * (reg_loss.sum () + lambda_2 * bla_loss.sum () + lambda_3 * L1_loss.sum())) / (code_length * num_train) + 0.5*lambda_1 * oth_loss.sum ()/ (code_length)

    print ('Tr_BLF:'+ str(Tr_BLF / code_length / num_train))
    print ('reg_loss:' + str (reg_loss.sum () / code_length / num_train))
    print ('oth_loss:' + str (lambda_1 * oth_loss.sum ()/ code_length))
    print ('bla_loss:' + str (lambda_2 * bla_loss.sum ()/ code_length / num_train))
    print ('L1_loss:' + str (lambda_3 * L1_loss.sum()/ code_length / num_train))
    print ('var_loss:' + str (lambda_2 * var_loss.sum ()/ code_length / num_train))
    print ('loss done!')
    return loss

def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda(),volatile=True)
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output[1].cpu().data).numpy()
    return B

def get_F(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda(),volatile=True)
        output = model(data_input)
        B[data_ind.numpy(), :] = output[1].cpu().data.numpy()
    return B

def get_fearture(model, data_loader, num_data, bit):
    Features = np.zeros([num_data, 4096], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda(),volatile=True)
        output = model(data_input)
        Features[data_ind.numpy(), :] = output[0].cpu().data.numpy()
    return Features

def adjusting_learning_rate(optimizer, iter):
    #update_list = [10, 20, 30, 40, 50]
    if ((iter % 3) == 0) & (iter !=0):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 5
        print ('learning rate is adjusted!')

def DAGH_algo(code_length, dataname):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
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
    lambda_1 = float(opt.lambda_1)
    lambda_2 = float(opt.lambda_2)
    lambda_3 = float(opt.lambda_3)

    record['param']['opt'] = opt
    record['param']['description'] = '[Comment: learning rate decay]'
    logger.info(opt)
    logger.info(code_length)
    logger.info(record['param']['description'])

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = load_dataset(dataname)
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    '''
    model construction
    '''
    beta = 2
    model = cnn_model.CNNNet(opt.arch, code_length)
    model.cuda()
    cudnn.benchmark = True
    DAGH_loss = dl.DAGHLoss (lambda_1, lambda_2, lambda_3, code_length)
    L1_criterion = nn.L1Loss ()
    L2_criterion = nn.MSELoss ()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.SGD (model.parameters (), lr=learning_rate, weight_decay=weight_decay, momentum=0.9) ####
    # optimizer = optim.RMSprop (model.parameters (), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.RMSprop (model.parameters (), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    # optimizer = optim.Adadelta (model.parameters (), weight_decay=weight_decay)
    # optimizer = optim.Adam (model.parameters ())

    B = np.sign(np.random.randn(code_length, num_database))

    model.train()
    for iter in range(max_iter):
        iter_time = time.time()

        trainloader = DataLoader(dset_database, batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4)
        F = np.zeros ((num_database, code_length), dtype=np.float)

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
        elif (iter % 3) == 0:
            dist_graph = get_dist_graph (Features, num_anchor)
            Z = calc_Z (dist_graph)
            print ('calculate dist graph forward done!')

        inv_A = inv (np.diag (Z.sum (0)))  # m X m
        Z_T = Z.transpose ()  # m X n
        left = np.dot (B, np.dot (Z, inv_A))  # k X m

        if iter == 0:
            loss_ini = calc_loss (B, ini_F, Z, inv_A, lambda_1, lambda_2, lambda_3, code_length)
            # loss_ini2 = calc_all_loss(B,F,Z,inv_A,Z1,Z2,Y1,Y2,rho1,rho2,lambda_1,lambda_2)
            print(loss_ini)
        '''
        learning deep neural network: feature learning
        '''
        Features = np.zeros ((num_database, 4096), dtype=np.float)
        for epoch in range(epochs):
            for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
                train_input = Variable(train_input.cuda())

                output = model(train_input)
                # Features[batch_ind, :] = output[0].cpu ().data.numpy ()
                # F[batch_ind, :] = output[1].cpu ().data.numpy ()

                batch_grad = get_batch_gard(B, left, Z_T, batch_ind)/(code_length*batch_size)
                batch_grad = Variable (torch.from_numpy (batch_grad).type (torch.FloatTensor).cuda ())
                optimizer.zero_grad()
                # output[1].backward(batch_grad, retain_graph=True)
                output[1].backward (batch_grad)

                B_cuda = Variable (torch.from_numpy (B[:,batch_ind]).type (torch.FloatTensor).cuda ())
                # optimizer.zero_grad ()
                other_loss = DAGH_loss(output[1].t(),B_cuda)
                one_vectors = Variable (torch.ones (output[1].size()).cuda ())
                L1_loss = L1_criterion(torch.abs(output[1]),one_vectors)
                # L2_loss = L2_criterion (output[1],B_cuda.t())
                All_loss = other_loss + lambda_3 * L1_loss / code_length
                All_loss.backward()

                optimizer.step()

                if (iteration % 200) == 0 :
                    print ('iteration:' + str (iteration))
                    #print (model.features[0].weight.data[1, 1, :, :])
                    #print (model.features[18].weight.data[1, 1, :, :])
                    #print (model.classifier[6].weight.data[:, 1])
        adjusting_learning_rate(optimizer, iter)

        trainloader2 = DataLoader (dset_database, batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=4)
        F = get_F (model, trainloader2, num_database, code_length)
        Features = get_fearture (model, trainloader2, num_database, code_length)

        '''
        learning binary codes: discrete coding
        '''
        # bf = np.sign (F)

        # F = np.random.randn (num_database, 12)
        loss_before = calc_loss (B, F, Z, inv_A, lambda_1, lambda_2, lambda_3, code_length)

        B = B_step (F, Z, inv_A)
        iter_time = time.time() - iter_time
        loss_ = calc_loss(B, F, Z, inv_A, lambda_1, lambda_2, lambda_3, code_length)

        logger.info('[Iteration: %3d/%3d][Train Loss: before:%.4f, after:%.4f]', iter, max_iter, loss_before, loss_)
        record['train loss'].append(loss_)
        record['iter time'].append(iter_time)

    '''
    training procedure finishes, evaluation
    '''
    model.eval()
    testloader = DataLoader(dset_test, batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)
    qB = encode(model, testloader, num_test, code_length)
    rB = B.transpose()

    topKs = np.arange (1, 500, 50)
    top_ndcg = 100
    map = calc_hr.calc_map (qB, rB, test_labels.numpy (), database_labels.numpy ())
    top_map = calc_hr.calc_topMap (qB, rB, test_labels.numpy (), database_labels.numpy (), 2000)
    Pres = calc_hr.calc_topk_pres (qB, rB, test_labels.numpy (), database_labels.numpy (), topKs)
    ndcg = calc_hr.cal_ndcg_k (qB, rB, test_labels.numpy (), database_labels.numpy (), top_ndcg)

    logger.info ('[lambda_1: %.4f]', lambda_1)
    logger.info ('[lambda_2: %.4f]', lambda_2)
    logger.info ('[lambda_3: %.4f]', lambda_3)
    logger.info('[Evaluation: mAP: %.4f]', map)
    logger.info ('[Evaluation: topK_mAP: %.4f]', top_map)
    logger.info('[Evaluation: Pres: %.4f]', Pres[0])
    logger.info ('[Evaluation: topK_ndcg: %.4f]', ndcg)
    record['rB'] = rB
    record['qB'] = qB
    record['map'] = map
    record['topK_map'] = top_map
    record['topK_ndcg'] = ndcg
    record['Pres'] = Pres
    record['F'] = F
    filename = os.path.join(logdir, str(code_length) + 'bits-record.pkl')

    _save_record(record, filename)
    return top_map


if __name__=="__main__":
    global opt, logdir
    opt = parser.parse_args()
    logdir = '-'.join(['log/datatest',opt.dataname, datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    _logging()
    _record()
    bits = [int(bit) for bit in opt.bits.split(',')]
    for bit in bits:
        DAGH_algo (bit,opt.dataname)
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
