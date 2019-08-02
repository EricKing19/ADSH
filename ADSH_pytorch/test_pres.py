import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import utils.DC_dataset as dataset
import utils.DAGH_no_Ink_loss as dl
import utils.cnn_model_DAGH as cnn_model
import utils.calc_hr as calc_hr
import torch
import torchvision.transforms as transforms

def _dataset(dataname):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    rootpath = os.path.join('/data/dacheng/Datasets/', dataname)

    if dataname=='NUSWIDE':
        dset_database = dataset.NUSWIDE ('train_img.txt', 'train_label.txt', transformations)
        dset_test = dataset.NUSWIDE ('test_img.txt', 'test_label.txt', transformations)
    elif dataname=='MirFlickr':
        dset_database = dataset.MirFlickr ('train_img.txt', 'train_label.txt', transformations)
        dset_test = dataset.MirFlickr ('test_img.txt', 'test_label.txt', transformations)
    elif dataname =='COCO':
        dset_database = dataset.COCO ('train_img.txt', 'train_label.txt', transformations)
        dset_test = dataset.COCO ('test_img.txt', 'test_label.txt', transformations)

    num_database, num_test = len (dset_database), len (dset_test)

    def load_label(filename, DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        fp = open(path, 'r')
        labels = [x.strip() for x in fp]
        fp.close()
        return torch.LongTensor(list(map(int, labels)))

    def DC_load_label(filename, DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        label = np.loadtxt (path, dtype=np.int64)
        return torch.LongTensor(label)

    def load_label2(root, train=True):
        base_folder = 'cifar-10-batches-py'
        train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]

        test_list = [
            ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ]

        root = os.path.expanduser(root)
        train = train  # training set or test set

        # now load the picked numpy arrays
        if train:
            train_data = []
            train_labels = []
            for fentry in train_list:
                f = fentry[0]
                file = os.path.join(root, base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                train_data.append(entry['data'])
                if 'labels' in entry:
                    train_labels += entry['labels']
                else:
                    train_labels += entry['fine_labels']
                fo.close()

        else:
            f = test_list[0][0]
            file = os.path.join(root, base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            test_data = entry['data']
            if 'labels' in entry:
                test_labels = entry['labels']
            else:
                test_labels = entry['fine_labels']
            fo.close()

        if train:
            target = train_labels
        else:
            target = test_labels
        return torch.LongTensor(list(map(int, target)))

    databaselabels = DC_load_label('train_label.txt', rootpath)
    testlabels = DC_load_label('test_label.txt', rootpath)

    # testlabels2 = load_label2('/home/dacheng/PycharmProjects/ADSH_pytorch/data', train=False)
    # databaselabels2 = load_label2('/home/dacheng/PycharmProjects/ADSH_pytorch/data', train=True)

    # testlabels = encoding_onehot(testlabels2)
    # databaselabels = encoding_onehot(databaselabels2)

    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)

    return nums, dsets, labels

filename = '/home/dacheng/PycharmProjects/ADSH_pytorch/log/noInk-MirFlickr-18-06-05-21-03-36//8bits-record.pkl'
#/home/dacheng/PycharmProjects/ADSH_pytorch/log/log-DAGH-FF-cifar10-18-05-07-11-12-49/
inf = pickle.load(open(filename))

qB = inf['qB']
rB = inf['rB']
dataname = 'MirFlickr'
topKs = np.arange (1, 500, 50)
top_ndcg = 100
nums, dsets, labels = _dataset (dataname)
database_labels, test_labels = labels


ndcg = calc_hr.cal_ndcg_k (qB, rB, test_labels.numpy (), database_labels.numpy (), top_ndcg)
map = calc_hr.calc_map (qB, rB, test_labels.numpy (), database_labels.numpy ())
top_map = calc_hr.calc_topMap (qB, rB, test_labels.numpy (), database_labels.numpy (), 2000)
Pres = calc_hr.calc_topk_pres (qB, rB, test_labels.numpy (), database_labels.numpy (), topKs)

print(map)
print(top_map)
