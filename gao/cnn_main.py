# from __future__ import division
import torch.nn as nn
import time
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
# import torch.utils.data
import numpy as np
# from eer import cal_eer_matlab
# import matplotlib.pyplot as plt
from gao import network, lr_scheduler
# from tqdm import tqdm
import torch.utils.data as tud
import gao.Utils as ut
import os
import torch as tc
import torch.nn.functional as functional
import re
import pickle as pk


# parameters
class P():
    TEST_AVERAGE_SCORE = True

    DATA_TRAIN = "/home/jiangyiheng/matlabProjects/xiangmu/mfcc/train/"
    DATA_TEST = "/home/jiangyiheng/matlabProjects/xiangmu/mfcc/test/"
    SAVE_DIR = "/home/jiangyiheng/pycharmProjects/master/gao/Data/model"
    PKL_TRAIN = "/home/jiangyiheng/pycharmProjects/master/gao/Data/train"
    PKL_TEST = "/home/jiangyiheng/pycharmProjects/master/gao/Data/test"
    ROI_TRAIN = "/home/jiangyiheng/pycharmProjects/master/gao/Data/roi_train.pkl"
    ROI_TEST = "/home/jiangyiheng/pycharmProjects/master/gao/Data/roi_test.pkl"

    LOAD_MODEL = SAVE_DIR  # set it to false or none if do not load,or set it to the direction which you want to load.
    SAVE_MODEL = [True,
                  0.4]  # True represent that need to save model ,0.5 indicate that save model should be when accuracy larger than 0.5
    LR = 0.1
    WEIGHT_DECAY = 1e-4
    EPOCH = 8
    TEST_FREQUENCY = 1

    STRIDE = 50
    FRAMELENS = np.linspace(50, 450, 9).astype(int)

    net_conv_kernel_sizes = [5, 3, 3, 3,
                             1]  # this parameter cannot more than actual model need,cause that len(    net_conv_kernel_sizes = [5, 3, 3, 3, 1]#this parameter length cannot more than actual model need,cause that len(    net_conv_kernel_sizes = [5, 3, 3, 3, 1]#this parameter cannot more than actual model need,cause that len(net_conv_kernel_sizes) will be used in the network.
    net_pooling_kernel_sizes = [[2, 3], [2, 3], [2, 2], [2, 2], [2, 2]]
    net_channels = [64, 128, 256, 512, 512]
    net_num_classes = 16
    net_in_channel = 1
    net_batch_size = 64
    net_mile_stone = [1, 3, 7]  # lr descent as epoch increase
    net_stone_time = 0.1  # lr descent time,i.e.lr=0.1*lr when lr descent
    net_gap = (
    2, 1)  # it can also be None in any dim,that means the dim will not be change in output of this gap layer.


class GenRoi():
    def __init__(self, sizes, stride, frameLens):
        # structure:
        ii = len(frameLens)
        jj = len(sizes)
        self.padding = np.empty(ii, list)
        self.time = np.empty(ii, list)
        # store more data:
        self.stride = stride
        self.frameLens = frameLens
        self.count = np.zeros((ii, jj))

        for i in range(ii):
            fixFrames = frameLens[i]
            tmp = []
            padd = []
            for j in range(jj):  # j is index of origin data,we can get label or data by index 'j'
                dataFrames = sizes[j][1]
                start = 0
                end = dataFrames
                if dataFrames < fixFrames:  # use self data to padding.
                    padd.append(True)
                    self.count[i, j] = 1
                    tmp.append((j, start, end))
                else:  # slide window.
                    # note:include start frame but except end frame,i.e. [start,end)
                    end = fixFrames
                    tmp.append((j, start, end))
                    padd.append(False)
                    co = (dataFrames - fixFrames) // stride
                    for k in range(co):
                        start += stride
                        end += stride
                        tmp.append((j, start, end))
                        padd.append(False)
                    self.count[i, j] = co + 1
            self.time[i] = tmp
            self.padding[i] = padd


class DataGet_Train(tud.Dataset):
    def __init__(self, data, gen, i):  # the second arg is index of frameLens
        self.count = len(gen.padding[i])
        self.origin_data_count = len(data.label)
        self.i = i
        self.gen = gen
        self.data = data

    def __getitem__(self, item):
        dt = self.gen.time[self.i][item]
        data = self.data.data[dt[0]][:, dt[1]:dt[2]]
        if self.gen.padding[self.i][item]:  # need to padding
            diffR = self.gen.frameLens[self.i] - np.size(
                data, axis=1)
            while diffR:
                # if diffR larger than length of data ,data[:diffR] will return data and donnot throw exception
                data = np.concatenate([data, data[:, :diffR]], axis=1)
                diffR = self.gen.frameLens[self.i] - np.size(data, axis=1)
        data = data[np.newaxis, :, :]
        return tc.from_numpy(data).type(tc.FloatTensor), self.data.label[dt[0]], dt[0]

    def __len__(self):
        return self.count


class DataGet_Test(tud.Dataset):
    def __init__(self, data):
        self.data = data
        self.count = len(self.data.label)

    def __getitem__(self, item):
        data = self.data.data[item]
        return tc.from_numpy(data[np.newaxis, :, :]).type(tc.FloatTensor), self.data.label[item], 0

    def __len__(self):
        return self.count


class Data():
    def __init__(self, pklDir, dataDir):
        files=[]
        try:
            files = os.listdir(dataDir)
            files.sort()
            files = [dataDir + i for i in files]
        except:
            print("origin data file doesn't exist")

        data = loadData(pklDir, files)
        # data = ut.multiReadProc(files)
        self.data = data[0]
        self.dataSize = data[1]
        self.label = data[2]


def main():
    setNum = len(P.FRAMELENS)
    # get train data
    train_loader = trainDataLoader(setNum)
    totals,test_loader=testDataLoader(setNum)

    model = modelConstruct()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = tc.optim.Adam(model.parameters(), P.LR, weight_decay=P.WEIGHT_DECAY)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=P.net_mile_stone, gamma=P.net_stone_time)

    for epoch in range(P.EPOCH):
        learning_rate_step(scheduler, epoch)
        # train for one epoch
        for i in range(setNum):  # cross train by different data set in every epoch
            accuracy = test(test_loader, model, epoch, totals)
            train(train_loader[i], model, criterion, optimizer, epoch, i)
            if np.mod(epoch * setNum + i, P.TEST_FREQUENCY) == 0 and epoch * setNum + i != 0:
                # evaluate on test set
                accuracy = test(test_loader, model, epoch, totals)
                modelSave(accuracy, model.state_dict(), epoch, i)


def trainDataLoader(setNum):
    trainD = Data(P.PKL_TRAIN, P.DATA_TRAIN)
    # generate Roi
    gen = genRoi(trainD.dataSize, P.STRIDE, P.FRAMELENS, P.ROI_TRAIN)
    # construct train data set loader
    train_loader = []
    kwargs = {'num_workers': 16, 'pin_memory': True}
    for i in range(setNum):
        data_set = DataGet_Train(trainD, gen, i)
        train_loader.append(tud.DataLoader(data_set, batch_size=P.net_batch_size, shuffle=True, **kwargs))
    return train_loader


def modelConstruct():
    model = network.net(P.net_conv_kernel_sizes, P.net_channels, P.net_num_classes, P.net_in_channel,
                        P.net_pooling_kernel_sizes, P.net_gap)
    model.cuda()
    if P.LOAD_MODEL:
        model.load_state_dict(tc.load(P.LOAD_MODEL))
    return model


@ut.timing("genRoi")
def genRoi(dataSize, stride, frameLens, roi):
    if os.path.exists(roi):
        # gen = np.load(roi + ".npy")
        with open(roi, 'rb') as f:
            gen = pk.load(f)
    else:
        gen = GenRoi(dataSize, stride, frameLens)
        # np.save(roi, gen)
        with open(roi, 'wb') as f:
            pk.dump(gen, f)

    return gen


@ut.timing("loadData")
def loadData(pklDir, files):  # numpy is byte data ,so must indicate that 'b' mode
    if os.path.exists(pklDir + ".npy"):
        data = np.load(pklDir + ".npy")
        # with open(pklDir,'rb') as f:
        #     data = pk.load(f)
    else:
        data = ut.multiReadProc(files)
        np.save(pklDir, data)
        # with open(pklDir,'wb') as f:
        #     pk.dump(data,f)
    return data


def modelSave(accuracy, save, epoch, set):
    if P.SAVE_MODEL[0] and accuracy > P.SAVE_MODEL[1]:
        dir = P.SAVE_DIR + "_" + str(epoch) + "_" + str(set)
        if os.path.exists(dir):
            os.system("rm " + dir)
        tc.save(save, dir)


@ut.timing("Train")
def train(loader, model, criterion, optimizer, epoch, set):
    # type: (object, object, object, object, object) -> object
    """Train for one epoch on the training set"""
    # losses = AverageMeter()
    # top1 = AverageMeter()

    # switch to train mode
    model.train()
    #
    for i, (input, target, _) in enumerate(loader):

        target = target.cuda(async=True)
        input = input.cuda()
        input_var = tc.autograd.Variable(input)
        target_var = tc.autograd.Variable(target)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            pred_y = tc.max(output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == target) / float(target.size(0))
            string = "[train] epoch:%d | set:%d | Loss:%f | accuracy:%f\n" % (epoch, set, loss, accuracy)
            with ut.Log(string):
                pass
    # batch_time.update(time.time() - end)
    string = "[=========train finished=========] epoch :%d | Loss:%f | accuracy:%f\n" % (epoch, loss, accuracy)
    with ut.Log(string):
        pass


@ut.timing("Test")
def test(loader, model, epoch, totals):
    """Test for one epoch on the training set"""
    model.eval()
    accuracy = 0
    if P.TEST_AVERAGE_SCORE:
        for i in range(len(totals)):
            output, target, ind = test_core(loader[i], model)
            _output = []
            _target = []
            for j in range(totals[i]):
                sum_ind = np.where(ind == j)
                _output.append(np.average(output[sum_ind], 0)[np.newaxis, :])
                _target.append(target[sum_ind[0][0]])
            output = np.concatenate(_output, 0)
            target = np.array(_target)
            total = totals[i]
            accuracy = testPerform(output, target, total)
            strings = "[==================test(score average,frame length:%d)==================] | epoch:%d | set:%d | accuracy:%f\n"\
                      % (P.FRAMELENS[i], epoch, i, accuracy)
            with ut.Log(strings):
                pass

    else:
        output, target, _ = test_core(loader, model)
        total = len(target)
        # measure accuracy and record loss
        accuracy = testPerform(output, target, total)
        strings = "[==============================test(usual)==============================] | epoch:%d | accuracy:%f\n" % (epoch, accuracy)
        with ut.Log(strings):
            pass
    return accuracy


def testPerform(output, target, total):
    pred_y = np.argmax(output, 1)
    right = sum(pred_y == target)
    accuracy = right / total
    return accuracy


def test_core(loader, model):
    vector = []
    targets = []
    index = []
    for i, (input, target, _i) in enumerate(loader):
        input = input.cuda()
        input = tc.autograd.Variable(input)

        # compute output
        output = model(input)
        output = functional.log_softmax(output, 1)
        output = output.cpu().data
        vector.append(output)
        targets.append(target)
        index.append(_i)
    return np.concatenate(vector, axis=0), np.concatenate(targets), np.concatenate(
        index)  # for torch,np method will transform it to np by default.


def learning_rate_step(scheduler, epoch):
    scheduler.step(epoch)
    lr = scheduler.get_lr()
    string = 'Learning rate:{0}\n'.format(lr)
    # print('Epoch:{0}, ' 'Learning rate:{1} |'.format(epoch, lr))
    with ut.Log(string):
        pass


def testDataLoader(setNum):
    testD = Data(P.PKL_TEST, P.DATA_TEST)
    kwargs = {'num_workers': 16, 'pin_memory': True}
    totals = []
    test_loader = []
    if P.TEST_AVERAGE_SCORE:
        # generate Roi
        gen_test = genRoi(testD.dataSize, P.STRIDE, P.FRAMELENS, P.ROI_TEST)

        for i in range(setNum):
            data_set = DataGet_Train(testD, gen_test, i)  # use 'train get' way to generate test data
            test_loader.append(tud.DataLoader(data_set, batch_size=P.net_batch_size, shuffle=False, **kwargs))
            totals.append(data_set.origin_data_count)
    else:
        data_set = DataGet_Test(testD)
        test_loader = tud.DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)
    return totals,test_loader


def testModel():
    setNum=len(P.FRAMELENS)
    totals, test_loader = testDataLoader(setNum)
    model = modelConstruct()
    test(test_loader, model, 0, totals)


if __name__ == '__main__':
    #testModel()
    with ut.Log("================================================================================\n"):
        main()
