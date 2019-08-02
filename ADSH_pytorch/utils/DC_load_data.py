import pickle
import os
import torch
import sys

import numpy as np
import torchvision.transforms as transforms
import DC_dataset as dataset



def encoding_onehot(target, nclasses=10):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def load_dataset(dataname, unlabeled=False):
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    rootpath = os.path.join('/data/dacheng/Datasets/', dataname)

    if (dataname=='NUSWIDE') | (dataname=='MirFlickr') | (dataname =='COCO'):
        dset_database = dataset.multi_dataset (dataname, 'train_img.txt', 'train_label.txt', transformations)
        dset_test = dataset.multi_dataset (dataname, 'test_img.txt', 'test_label.txt', transformations)
    else:
        if unlabeled:
            dset_database = dataset.single_dataset (dataname, 'un_img.txt', 'un_label.txt', transformations)
        else:
            dset_database = dataset.single_dataset (dataname,'train_img.txt', 'train_label.txt', transformations)
        dset_test = dataset.single_dataset (dataname, 'test_img.txt', 'test_label.txt', transformations)

    # if dataname=='NUSWIDE':
    #     dset_database = dataset.NUSWIDE ('train_img.txt', 'train_label.txt', transformations)
    #     dset_test = dataset.NUSWIDE ('test_img.txt', 'test_label.txt', transformations)
    # elif dataname=='MirFlickr':
    #     dset_database = dataset.MirFlickr ('train_img.txt', 'train_label.txt', transformations)
    #     dset_test = dataset.MirFlickr ('test_img.txt', 'test_label.txt', transformations)
    # elif dataname =='COCO':
    #     dset_database = dataset.COCO ('train_img.txt', 'train_label.txt', transformations)
    #     dset_test = dataset.COCO ('test_img.txt', 'test_label.txt', transformations)
    # elif dataname == 'CIFAR10':
    #     dset_database = dataset.CIFAR10 ('train_img.txt', 'train_label.txt', transformations)
    #     dset_test = dataset.CIFAR10 ('test_img.txt', 'test_label.txt', transformations)
    # elif dataname == 'CIFAR100':
    #     dset_database = dataset.CIFAR100 ('train_img.txt', 'train_label.txt', transformations)
    #     dset_test = dataset.CIFAR100 ('test_img.txt', 'test_label.txt', transformations)
    # elif dataname == 'CIFAR100_py':
    #     dset_database = dataset.CIFAR100 (root = '/data/dacheng/Datasets/CIFAR100_py/', train=True, transform=transformations)
    #     dset_test = dataset.CIFAR100 (root = '/data/dacheng/Datasets/CIFAR100_py/', train=False, transform=transformations)
    # elif dataname == 'Mnist':
    #     dset_database = dataset.Mnist ('train_img.txt', 'train_label.txt', transformations)
    #     dset_test = dataset.Mnist ('test_img.txt', 'test_label.txt', transformations)
    # elif dataname == 'fashion_mnist':
    #     dset_database = dataset.fashion_mnist ('train_img.txt', 'train_label.txt', transformations)
    #     dset_test = dataset.fashion_mnist ('test_img.txt', 'test_label.txt', transformations)

    num_database, num_test = len (dset_database), len (dset_test)


    def load_single_label(filename, DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        fp = open(path, 'r')
        labels = [x.strip() for x in fp]
        fp.close()
        return torch.LongTensor(list(map(int, labels)))

    def load_multi_label(filename, DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        label = np.loadtxt (path, dtype=np.int64)
        return torch.LongTensor(label)

    def load_label_CIFAR100_py(root, train=True):
        base_folder = 'cifar-100-python'
        train_list = [
            ['train', '16019d7e3df5f24257cddd939b257f8d'],
        ]

        test_list = [
            ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
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

    # def DC_load_label_MNIST(filename, root):
    #     _, labels = torch.load (os.path.join (root, filename))
    #     return torch.LongTensor(labels)

    if (dataname=='CIFAR10') | (dataname == 'Mnist') | (dataname == 'fashion_mnist') | (dataname == 'STL10'):
        testlabels_ = load_single_label('test_label.txt', rootpath)
        databaselabels_ = load_single_label('train_label.txt', rootpath)
        testlabels = encoding_onehot(testlabels_)
        databaselabels = encoding_onehot(databaselabels_)

    # elif (dataname == 'Mnist') | (dataname == 'fashion_mnist'):
    #     # databaselabels_ = DC_load_label_MNIST ('training.pt', root='/home/dacheng/PycharmProjects/ADSH_pytorch/data/processed/')
    #     # testlabels_ = DC_load_label_MNIST ('test.pt', root='/home/dacheng/PycharmProjects/ADSH_pytorch/data/processed/')
    #     testlabels_ = load_label('test_label.txt', rootpath)
    #     databaselabels_ = load_label('train_label.txt', rootpath)
    #     testlabels = encoding_onehot(testlabels_)
    #     databaselabels = encoding_onehot(databaselabels_)

    elif dataname == 'CIFAR100_py':
        testlabels_ = load_label_CIFAR100_py('/data/dacheng/Datasets/CIFAR100/', train=False)
        databaselabels_ = load_label_CIFAR100_py('/data/dacheng/Datasets/CIFAR100/', train=True)
        testlabels = encoding_onehot(testlabels_, nclasses=100)
        databaselabels = encoding_onehot(databaselabels_, nclasses=100)

    elif dataname == 'CIFAR100':
        testlabels_ = load_single_label('test_label.txt', rootpath)
        databaselabels_ = load_single_label('train_label.txt', rootpath)
        testlabels = encoding_onehot (testlabels_, nclasses=100)
        databaselabels = encoding_onehot (databaselabels_, nclasses=100)

    elif dataname == 'SUN20':
        testlabels_ = load_single_label('test_label.txt', rootpath)
        databaselabels_ = load_single_label('train_label.txt', rootpath)
        testlabels = encoding_onehot (testlabels_, nclasses=20)
        databaselabels = encoding_onehot (databaselabels_, nclasses=20)

    elif dataname == 'imageNet50':
        testlabels_ = load_single_label('test_label.txt', rootpath)
        databaselabels_ = load_single_label('train_label.txt', rootpath)
        testlabels = encoding_onehot (testlabels_, nclasses=50)
        databaselabels = encoding_onehot (databaselabels_, nclasses=50)

    else:
        databaselabels = load_multi_label('train_label.txt', rootpath)
        testlabels = load_multi_label('test_label.txt', rootpath)


    # elif dataname == 'fashion_mnist':
    #     databaselabels_ = DC_load_label_MNIST ('training.pt', root='/home/dacheng/PycharmProjects/ADSH_pytorch/data/processed/')
    #     testlabels_ = DC_load_label_MNIST ('test.pt', root='/home/dacheng/PycharmProjects/ADSH_pytorch/data/processed/')
    #     testlabels = encoding_onehot(testlabels_)
    #     databaselabels = encoding_onehot(databaselabels_)


    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)

    return nums, dsets, labels
