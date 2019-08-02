import pickle
import os
import argparse
import logging
import torch
import time
import math

import numpy as np
from numpy.linalg import *
import torch.optim as optim
import torchvision.transforms as transforms

from sklearn.cluster import KMeans
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn

import utils.data_processing as dp
import utils.DAGH_FF_loss as dl
import utils.cnn_model_DAGH as cnn_model
import utils.calc_hr as calc_hr

parser = argparse.ArgumentParser(description="DAGH demo")
parser.add_argument('--bits', default='12', type=str,
                    help='binary code length (default: 12,24,32,48)')
parser.add_argument('--gpu', default='0', type=str,
                    help='selected gpu (default: 3)')
parser.add_argument('--arch', default='vgg11', type=str,
                    help='model name (default: resnet50)')
parser.add_argument('--max-iter', default=6, type=int,
                    help='maximum iteration (default: 50)')
parser.add_argument('--epochs', default=2, type=int,
                    help='number of epochs (default: 1)')
parser.add_argument('--batch-size', default=32, type=int,
                    help='batch size (default: 64)')
parser.add_argument('--lambda-1', default=0, type=int,
                    help='hyper-parameter: lambda-1 (default: 0.00001)')
parser.add_argument('--lambda-2', default=0.5, type=int,
                    help='hyper-parameter: lambda-2 (default: 0.00001)')
parser.add_argument('--learning-rate', default=0.001, type=float,
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

def calc_W(XTrain):
    '''

    :param XTrain: n X d
    :return: n X (d+1)
    '''
    n,d = XTrain.shape
    FnormX = (XTrain **2).sum(1)
    rho = 2 * FnormX.mean()
    FnormX = np.exp (-FnormX / rho)
    alpha = np.sqrt((np.exp(1) - np.exp (-1)) / rho)
    FnormX_tile = np.tile(FnormX,(d,1))
    part = alpha * XTrain * FnormX_tile.transpose()
    PX = np.vstack((part.transpose(), np.sqrt ((np.exp (1) + np.exp (-1)) / 2) * FnormX))
    QX = np.vstack((part.transpose(), np.sqrt ((np.exp (1) + np.exp (-1)) / 2) * FnormX))

    S = np.dot(PX.transpose(),QX)
    D = 1./np.sqrt(S.sum(0))
    DD =  np.diag (D)
    PXD = np.dot(DD,PX.transpose())
    QXD = np.dot(DD,QX.transpose())
    return PXD,QXD

def get_batch_gard(B, P, Q, batch_ind):
    """
    get the gradient of each corresponding batch : = BL[:,batch_ind]
    :param B: Binary codes of the all data points, k X n
    :param batch_ind: batch size X k
    :return: the gradient of the batch points, batch size X k
    """
    grad = B[:,batch_ind] - np.dot(np.dot(B,P), (Q[batch_ind,:]).transpose())
    return grad.transpose()

def B_step(F, P, Q):
    """
    Update B : F^t * W
    :param F: output of network as the real-valued embeddings n X k
    :param P,Q: n X (d+1)
    Affinity matrix W: P*Q'
    :return: the updated B, k X n
    """
    B = np.dot(np.dot(F.transpose(),P), Q.transpose()) # k X m
    print ('B step done!')
    return np.sign(B)

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

def calc_loss(B, F, P, Q, lambda_1, lambda_2, code_length):
    """
    Calculate loss: Tr(BLF^t) = Tr(B * Z * inv_A * Z^t * F^t)
    :param F: output of network n X k
    :param B: binary codes k X n
    :param Z: anchor graph, n X m
    :return: loss: trace(BLF)
    """
    temp2 = np.dot (np.dot (B, P), Q.transpose ())  # k X m

    BAF = np.dot (temp2, F) # k X k
    Tr_BLF = np.trace(np.dot(B, F) - BAF)

    num_train = np.size (B,1)
    nI_K =  num_train * np.eye (code_length, code_length)
    reg_loss = (B - F.transpose()) ** 2
    oth_loss = (np.dot(F.transpose(), F) - nI_K) ** 2
    bla_loss = (F.sum (0)) ** 2
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
    # DAGH_loss = dl.DAGHLoss (lambda_1, lambda_2, code_length)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    B = np.sign(np.random.randn(code_length, num_database))

    model.train()
    for iter in range(max_iter):
        iter_time = time.time()

        trainloader = DataLoader(dset_database, batch_size=batch_size,
                                 shuffle=True,
                                 )
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
            P,Q = calc_W(ini_Features)
            bf = np.sign(ini_F)
        elif (iter % 3) == 0:
            P,Q = calc_W (Features)
            print ('calculate dist graph forward done!')

        if iter == 0:
            loss_ini = calc_loss (B, ini_F, P, Q, lambda_1, lambda_2, code_length)
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

                batch_grad = get_batch_gard(B, P, Q, batch_ind)/(0.5*batch_size)
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
        loss_before = calc_loss (B, F, P, Q, lambda_1, lambda_2, code_length)

        B = B_step (F, P, Q)
        iter_time = time.time() - iter_time
        loss_ = calc_loss(B, F, P, Q, lambda_1, lambda_2, code_length)

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
    logdir = '-'.join(['log/log-DAGH-FF-cifar10', datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    _logging()
    _record()
    bits = [int(bit) for bit in opt.bits.split(',')]
    for bit in bits:
        DAGH_algo(bit)
