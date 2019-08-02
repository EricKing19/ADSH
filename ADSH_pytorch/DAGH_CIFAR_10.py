import pickle
import os
import argparse
import logging
import torch
import time

import numpy as np
from numpy.linalg import *
import torch.optim as optim
import torchvision.transforms as transforms

from sklearn.cluster import KMeans
from datetime import datetime
from scipy.optimize import fmin_l_bfgs_b, fmin, minimize
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn

import utils.data_processing as dp
import utils.DAGH_loss as dl
import utils.cnn_model_DAGH as cnn_model
import utils.calc_hr as calc_hr

parser = argparse.ArgumentParser(description="DAGH demo")
parser.add_argument('--bits', default='12', type=str,
                    help='binary code length (default: 12,24,32,48)')
parser.add_argument('--gpu', default='1', type=str,
                    help='selected gpu (default: 3)')
parser.add_argument('--arch', default='vgg11', type=str,
                    help='model name (default: resnet50)')
parser.add_argument('--max-iter', default=9, type=int,
                    help='maximum iteration (default: 50)')
parser.add_argument('--epochs', default=2, type=int,
                    help='number of epochs (default: 1)')
parser.add_argument('--batch-size', default=32, type=int,
                    help='batch size (default: 64)')
parser.add_argument('--gamma', default=0.0001, type=int,
                    help='hyper-parameter: gamma (default: 0.0001)')
parser.add_argument('--rho1', default=0.001, type=int,
                    help='hyper-parameter: rho1 (default: 0.001)')
parser.add_argument('--rho2', default=0.001, type=int,
                    help='hyper-parameter: rho1 (default: 0.001)')
parser.add_argument('--lambda-1', default=0, type=int,
                    help='hyper-parameter: lambda-1 (default: 0.00001)')
parser.add_argument('--lambda-2', default=0.5, type=int,
                    help='hyper-parameter: lambda-2 (default: 0.00001)')
parser.add_argument('--learning-rate', default=0.0001, type=float,
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
    kmeans = KMeans (n_clusters=num_anchor, random_state=0, n_jobs=16, max_iter=50).fit_transform(all_points)
    print ('dist graph done!')
    return np.asarray(kmeans)

def calc_Z(dist_graph,s=2):
    """
    calculate anchor graph (n data points vs m anchors),
    :param dist_graph: the distance matrix of n data points vs m anchors, n X m
    :param s: the number of nearest anchors, default = 2
    :return: anchor graph, n X m
    """
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
    Update B : F * L
    :param F: output of network as the real-valued embeddings n X k
    :param Z: anchor graph, n X m
    Laplacian matrix L: I - Z * inv_A * Z^t
    :return: the updated B, k X n
    """
    Z_T = Z.transpose() # m X n
    temp = np.dot(F.transpose(), np.dot(Z, inv_A)) # k X m
    B = F.transpose() - np.dot(temp, Z_T) # k X n
    print ('B step done!')
    return -np.sign(B)

def B2_step(F, Z, inv_A):
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
    Y1 = np.random.randn(bit, num_train)
    Y2 = np.random.randn(bit, num_train)
    loss_old = 0

    Z1 = ini_B
    Z2 = ini_B

    Z_T = Z.transpose()
    nI_K = num_train * np.eye (bit, bit)


    def func_B(B, F, Z, inv_A, G, nI_K, lambda_1, lambda_2):
        num_train,bit = F.shape
        B = np.reshape (B, (bit, num_train))
        temp = np.dot(B, np.dot(Z, inv_A)) # k X m
        temp2 = np.dot(temp, Z_T) # k X n
        BAF = np.dot(temp2, F) # k X k
        loss_BLF = np.trace(np.dot(B, F) - BAF)

        reg_loss = (B - F.transpose()) ** 2
        oth_loss = lambda_1 * ((np.dot(B, F) - nI_K) ** 2)
        bla_loss = lambda_2 * (B.sum (0) ** 2)

        rho_loss = ((rho1+rho2)/2) * (B**2).sum() + np.trace(B.dot(G.transpose()))
        loss = rho_loss + loss_BLF + 0.5 * (reg_loss.sum() +  oth_loss.sum() + bla_loss.sum())
        return loss

    def grad_B(B, F, Z, inv_A, G, nI_K, lambda_1, lambda_2):
        num_train, bit = F.shape
        B = np.reshape (B, (bit, num_train))
        rho_grad = (rho1+rho2+1) * B
        temp = np.dot (F.transpose(), np.dot (Z, inv_A))  # k X m
        L_grad = np.dot (temp, Z_T)  # k X n
        othr_grad = lambda_1* np.dot((np.dot(B, F) - nI_K),F.transpose())

        bla_grad =  lambda_2* np.dot(B,np.ones((num_train,num_train)))

        grad_all = rho_grad - L_grad + othr_grad + bla_grad + G
        return grad_all.flatten()

    for iter_B in range(20):
        print ('B_iteration %3d\n'% iter_B)
        G = Y1 + Y2 - rho1 * Z1 - rho2 * Z2
        res = minimize (func_B, B,
                    args=(F, Z, inv_A, G, nI_K, lambda_1, lambda_2),
                    method='L-BFGS-B',
                    jac=grad_B,
                    options={ 'ftol':1e-4, 'maxiter':10,'maxfun':20, 'disp': True})

        Bk = np.reshape (res.x, [bit, num_train])
        count = (Bk > 0).sum ()

        print ('+1, -1: %.2f%%\n' % (float(count) / num_train / bit * 100))
        print ('res(init_B and Bk): %d\n' % ((np.sign(B)-ini_B)).sum())

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
        print ('loss is %.4f, residual error is %.5f\n'% (loss, res_error))

        if (np.abs(res_error) <= 1e-4):
            break

    return np.sign(B)

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
    Y1 = Variable(torch.randn(bit, num_train))
    Y2 = Variable(torch.randn(bit, num_train))
    loss_old = 0


    B = Variable(torch.from_numpy(ini_B).type(torch.FloatTensor), requires_grad = True)
    Z1 = Variable(torch.from_numpy(ini_B).type(torch.FloatTensor))
    Z2 = Variable(torch.from_numpy(ini_B).type(torch.FloatTensor))

    optimizer_B = optim.LBFGS ([B], lr=0.1)
    F = Variable (torch.from_numpy(F).type(torch.FloatTensor))
    inv_A = Variable (torch.from_numpy(inv_A).type(torch.FloatTensor))
    Z = Variable (torch.from_numpy(Z).type(torch.FloatTensor))
    Z_T = Z.t ()  # m X n
    nI_K = Variable(num_train * torch.eye (bit, bit))

    for iter_B in range (20):
        def closure():
            optimizer_B.zero_grad ()

            temp = torch.mm(B, torch.mm(Z, inv_A)) # k X m
            temp2 = torch.mm(temp, Z_T) # k X n
            BAF = torch.mm(temp2, F) # k X k
            loss_BLF = torch.trace(torch.mm(B, F) - BAF)

            reg_loss = (B - F.t()) ** 2
            oth_loss = lambda_1 * ((torch.mm(B, F) - nI_K) ** 2)
            bla_loss = lambda_2 * (B.sum (0) ** 2)

            G = Y1 + Y2 - rho1 * Z1 - rho2 * Z2

            rho_loss = ((rho1+rho2)/2) * (B**2).sum() + torch.trace(B.mm(G.t()))
            loss = (rho_loss + loss_BLF + 0.5 * (reg_loss.sum() +  oth_loss.sum() + bla_loss.sum())) / num_train
            loss.backward()
            return loss
        optimizer_B.step(closure)
        count = (B.data.numpy() > 0).sum()
        print ('+1, -1: %.2f%%\n' % (float(count) / num_train / bit * 100))
        print ('res(init_B and Bk): %d\n' % ((np.sign(B.data.numpy())-ini_B)).sum())

        Z1_k = H1_step (B.data.numpy(), Y1.data.numpy(), rho1)
        Z2_k = H2_step (B.data.numpy(), Y2.data.numpy(), rho2)

        Y1_k, Y2_k = Y_step (Y1.data.numpy(), Y2.data.numpy(), Z1_k, Z2_k, B.data.numpy(), rho1, rho2, gamma)

        Z1 = Variable(torch.from_numpy(Z1_k).type(torch.FloatTensor))
        Z2 = Variable(torch.from_numpy(Z2_k).type(torch.FloatTensor))
        Y1 = Variable(torch.from_numpy(Y1_k).type(torch.FloatTensor))
        Y2 = Variable(torch.from_numpy(Y2_k).type(torch.FloatTensor))

        loss = calc_all_loss (B.data.numpy(), F.data.numpy(), Z.data.numpy(), inv_A.data.numpy(), Z1.data.numpy(), Z2.data.numpy(), Y1.data.numpy(), Y2.data.numpy(), rho1, rho2, lambda_1, lambda_2)
        res_error = (loss - loss_old) / loss_old
        loss_old = loss
        print ('loss is %.4f, residual error is %.5f\n'% (loss, res_error))
        if (np.abs(res_error) <= 1e-4):
            break

    return np.sign(B.data.numpy())

def H1_step(B, Y1, rho1):
    theta1 = B + 1/rho1 * Y1
    theta1[theta1 > 1] = 1
    theta1[theta1 < -1] = -1
    Z1_k = theta1
    return Z1_k

def H2_step(B, Y2, rho2):
    bit, num_train =  B.shape
    theta2 = B + 1/rho2 * Y2
    norm_B = np.linalg.norm(theta2,'fro')
    theta2 = np.sqrt(num_train*bit) * theta2 / norm_B
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
    temp = np.dot (B, np.dot (Z, inv_A)) # k X m
    temp2 = np.dot (temp, Z_T) # k X n
    BAF = np.dot (temp2, F) # k X k
    Tr_BLF = np.trace(np.dot(B, F) - BAF)

    nI_K =  num_train * np.eye (bit, bit)
    res1 = B-Z1
    res2 = B-Z2

    reg_loss = (B - F.transpose()) ** 2
    oth_loss = lambda_1 *((np.dot(B, F) - nI_K) ** 2)
    bla_loss = lambda_2 * (F.sum (0) ** 2 + B.sum(1) **2)
    z1_loss = rho1 * ((res1 ** 2).sum())
    z2_loss = rho2 * ((res2 ** 2).sum ())
    y1_loss = np.trace (np.dot(res1, Y1.transpose()))
    y2_loss = np.trace (np.dot (res2, Y2.transpose()))

    loss = (Tr_BLF + 0.5 * (reg_loss.sum () +  oth_loss.sum () + bla_loss.sum () + z1_loss + z2_loss) + y1_loss + y2_loss) /num_train

    print ('loss all done!')
    return loss


def encoding_onehot(target, nclasses=10):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def _dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dset_database = dp.DatasetProcessingCIFAR_10(
        'data/CIFAR-10', 'database_img.txt', 'database_label.txt', transformations)
    dset_test = dp.DatasetProcessingCIFAR_10(
        'data/CIFAR-10', 'test_img.txt', 'test_label.txt', transformations)
    num_database, num_test = len(dset_database), len(dset_test)

    def load_label(filename, DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        fp = open(path, 'r')
        labels = [x.strip() for x in fp]
        fp.close()
        return torch.LongTensor(list(map(int, labels)))
    testlabels = load_label('test_label.txt', 'data/CIFAR-10')
    databaselabels = load_label('database_label.txt', 'data/CIFAR-10')

    testlabels = encoding_onehot(testlabels)
    databaselabels = encoding_onehot(databaselabels)

    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)
    return nums, dsets, labels

def calc_loss(B, F, Z, inv_A, lambda_1, lambda_2, code_length):
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
    nI_K =  num_train * np.eye (code_length, code_length)
    reg_loss = (B - F.transpose()) ** 2
    oth_loss = (np.dot(B, F) - nI_K) ** 2
    bla_loss = (F.sum (0)) ** 2 + (B.sum(1)) **2
    loss = (Tr_BLF + 0.5 * (reg_loss.sum () + lambda_1 * oth_loss.sum () + lambda_2 * bla_loss.sum ())) / num_train

    print ('loss done!')
    return loss

def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output[1].cpu().data).numpy()
    return B

def adjusting_learning_rate(optimizer, iter):
    #update_list = [10, 20, 30, 40, 50]
    if ((iter % 3) == 0) & (iter !=0):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 5
        print ('learning rate is adjusted!')

def DAGH_algo(code_length):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    '''
    parameter setting
    '''
    max_iter = opt.max_iter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    weight_decay = 5 * 10 ** -4
    num_anchor = opt.num_anchor
    lambda_1 = opt.lambda_1
    lambda_2 = opt.lambda_2
    rho1 = opt.rho1
    rho2 = opt.rho2
    gamma = opt.gamma

    record['param']['opt'] = opt
    record['param']['description'] = '[Comment: learning rate decay]'
    logger.info(opt)
    logger.info(code_length)
    logger.info(record['param']['description'])

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = _dataset()
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    '''
    model construction
    '''
    model = cnn_model.CNNNet(opt.arch, code_length)
    model.cuda()
    DAGH_loss = dl.DAGHLoss (lambda_1, lambda_2, code_length)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
            ini_F = np.zeros ((num_database, 12), dtype=np.float)
            for iteration, (train_input, train_label, batch_ind) in enumerate (trainloader):
                train_input = Variable (train_input.cuda ())
                output = model (train_input)
                ini_Features[batch_ind, :] = output[0].cpu ().data.numpy ()
                ini_F[batch_ind, :] = output[1].cpu ().data.numpy ()
            print ('initialization dist graph forward done!')
            dist_graph = get_dist_graph (ini_Features, num_anchor)
            # dist_graph = np.random.rand(num_database,num_anchor)
            bf = np.sign(ini_F)
            Z = calc_Z (dist_graph)
        elif (iter % 3) == 0:
            dist_graph = get_dist_graph (Features, num_anchor)
            Z = calc_Z (dist_graph)
            print ('calculate dist graph forward done!')

        inv_A = inv (np.diag (Z.sum (0)))  # m X m
        Z_T = Z.transpose ()  # m X n
        left = np.dot (B, np.dot (Z, inv_A))  # k X m

        if iter == 0:
            loss_ini = calc_loss (B, ini_F, Z, inv_A, lambda_1, lambda_2, code_length)
        #     #loss_ini2 = calc_all_loss(B,F,Z,inv_A,Z1,Z2,Y1,Y2,rho1,rho2,lambda_1,lambda_2)
            print(loss_ini)
        '''
        learning deep neural network: feature learning
        '''
        Features = np.zeros ((num_database, 4096), dtype=np.float)
        for epoch in range(epochs):
            for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
                train_input = Variable(train_input.cuda())

                output = model(train_input)
                Features[batch_ind, :] = output[0].cpu ().data.numpy ()
                F[batch_ind, :] = output[1].cpu ().data.numpy ()

                batch_grad = get_batch_gard(B, left, Z_T, batch_ind)/(0.5*batch_size)
                batch_grad = Variable (torch.from_numpy (batch_grad).type (torch.FloatTensor).cuda ())
                optimizer.zero_grad()
                output[1].backward(batch_grad, retain_graph=True)

                B_cuda = Variable (torch.from_numpy (B[:,batch_ind]).type (torch.FloatTensor).cuda ())
                # optimizer.zero_grad ()
                loss2 = DAGH_loss(output[1].t(),B_cuda)
                #loss = criterion (output[1],B_cuda.t())
                loss2.backward()

                optimizer.step()

                if (iteration % 200) == 0 :
                    print ('iteration:' + str (iteration))
                    #print (model.features[0].weight.data[1, 1, :, :])
                    #print (model.features[18].weight.data[1, 1, :, :])
                    #print (model.classifier[6].weight.data[:, 1])
        adjusting_learning_rate(optimizer, iter)
        '''
        learning binary codes: discrete coding
        '''
        # bf = np.sign (F)

        # F = np.random.rand (num_database, 12)
        loss_before = calc_loss (B, F, Z, inv_A, lambda_1, lambda_2, code_length)
        # B = B3_step (B, F, Z, inv_A, lambda_1, lambda_2, rho1, rho2, gamma)

        B = B2_step (F, Z, inv_A)
        iter_time = time.time() - iter_time
        loss_ = calc_loss(B, F, Z, inv_A, lambda_1, lambda_2, code_length)

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
    map = calc_hr.calc_map(qB, rB, test_labels.numpy(), database_labels.numpy())
    top_map = calc_hr.calc_topMap (qB, rB, test_labels.numpy (), database_labels.numpy (), 2000)
    logger.info('[Evaluation: mAP: %.4f]', map)
    logger.info ('[Evaluation: topK_mAP: %.4f]', top_map)
    record['rB'] = rB
    record['qB'] = qB
    record['map'] = map
    record['topK_map'] = top_map
    filename = os.path.join(logdir, str(code_length) + 'bits-record.pkl')

    _save_record(record, filename)


if __name__=="__main__":
    global opt, logdir
    opt = parser.parse_args()
    logdir = '-'.join(['log/log-DAGH-cifar10', datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    _logging()
    _record()
    bits = [int(bit) for bit in opt.bits.split(',')]
    for bit in bits:
        DAGH_algo(bit)
