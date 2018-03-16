from __future__ import division
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from gao.getbatch_old import getBatchTrn, getBatchVal
import shutil
import time
from time import strftime, localtime
import numpy as np
# from eer import cal_eer_matlab
import matplotlib.pyplot as plt
import argparse
from gao import network, lr_scheduler
from tqdm import tqdm
import torch.utils.data as tud
import gao.Utils as ut
import os
import torch as tc
import torch.nn.functional as functional
import re

# parser = argparse.ArgumentParser(description='PyTorch LID-net training')
# parser.add_argument('--imdb-path', default='./nist04_nist05_nist06_nist08_swbd1_swbd2_swbdCellP2_70e_50r_imdb_mfcc.pkl', type=str,
#                     help='imdb of index of data')
# parser.add_argument('--data-root', default='/home/gaozf/SRE/data/mfcc23_vad_mat/train', type=str,
#                     help='root dir of data')
# # parser.add_argument('--num-classes', default=10, type=int,
# #                     help='num_classes')
# # parser.add_argument('--time', default='3s', type=str,
# #                     help='how long time of data')
# parser.add_argument('--min-duration', default=200, type=int,
#                     help='min_duration: 3000, 1000, 300')
# parser.add_argument('--max-duration', default=400, type=int,
#                     help='min_duration: 3000, 1000, 300')
# parser.add_argument('--egs-root', default='/home/gaozf/SRE/data/nist04_nist05_nist06_nist08_swbd1_swbd2_swbdCellP2_70e_50r_egs', type=str,
#                     help='egs_root: 50')
# parser.add_argument('--egs-num', default=70, type=int,
#                     help='egs_num: 50-150')
# parser.add_argument('--num-repeat', default=50, type=int,
#                     help='num-repeat: 50')
# parser.add_argument('--gender', default='f_m', type=str,
#                     help='gender: male(m,M), female(f,F), f_m')
# parser.add_argument('--sre-set', default='nist04_nist05_nist06_nist08_swbd1_swbd2_swbdCellP2', type=str,
#                     help='sre sets: nist04_nist05_nist06_nist08_nist10_swbd1_swbd2_swbdCellP2')
# parser.add_argument('--select_num', default=8, type=int,
#                     help='select_num: select the spk whos utt num >= select_num')
# parser.add_argument('--is-discard', default=True, type=bool,
#                     help='whether to discard utterance which is shorter than discard_gate')
# parser.add_argument('--discard-gate', default=1000, type=int,
#                     help='whether to discard utterance which is shorter than discard_gate')
# parser.add_argument('--filter-sizes', default='5_3_3_1_1_1_1', type=str,
#                     help='the width of filter size: 3, 5,7 ..')
# parser.add_argument('--channels', default='512_512_512_512_1636_512_300', type=str,
#                     help='the numbers of filter out channels: 64, 128, 256 ..')
# parser.add_argument('--dilation', default='1_2_4_1_1_1_1', type=str,
#                     help='the dilation of filter: 1, 1, 2, 4, ..')
# parser.add_argument('--layers', default=100, type=int,
#                     help='total number of layers (default: 100)')
# parser.add_argument('--growth', default=12, type=int,
#                     help='number of new channels per layer (default: 12)')
# parser.add_argument('--droprate', default=0, type=float,
#                     help='dropout probability (default: 0.0)')
# parser.add_argument('--reduce', default=0.5, type=float,
#                     help='compression rate in transition stage (default: 0.5)')
# parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
#                     help='To not use bottleneck block')
# parser.add_argument('--epochs', default=6, type=int,
#                     help='number of total epochs to run')
# parser.add_argument('--test-freq', default=1, type=int,
#                     help='how many train epoch test once:1')
# parser.add_argument('--start-epoch', default=0, type=int,
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('--start-ep', default=0, type=int,
#                     help='manual ep number (useful on restarts)')
# parser.add_argument('--batch-size', default=64, type=int,
#                     help='mini-batch size (default: 128)')
# parser.add_argument('--lr', default=0.1, type=float,
#                     help='initial learning rate')
# parser.add_argument('--lr-step', default=2, type=int,
#                     help='step decay learning rate')
# parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
# parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum')
# parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
#                     help='weight decay (default: 5e-4)')
# parser.add_argument('--print-freq', '-p', default=100, type=int,
#                     help='print frequency (default: 10)')
# parser.add_argument('--num-workers', default=12, type=int,
#                     help='num-workers:1')
# parser.add_argument('--gpu', default='0', type=str,
#                     help='GPU id: 0')
# parser.add_argument('--resume', default='', type=str,
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('--save', default='results', type=str,
#                     help='save parameters and logs in this folder')
# best_prec_eva = 0.
# best_prec_train = 0.
# best_eer2_eva = 1.

#parameters
class P():
    TRAIN_DIR="/home/jiangyiheng/matlabProjects/xiangmu/mfcc/train/"
    TEST_DIR ="/home/jiangyiheng/matlabProjects/xiangmu/mfcc/test/"
    VALIDATION_DIR= "/home/jiangyiheng/matlabProjects/xiangmu/mfcc/verification/"
    ENROLL_DIR="/home/jiangyiheng/matlabProjects/xiangmu/mfcc/enroll/"
    SAVE="/home/jiangyiheng/pycharmProjects/master/gao/"
    LOAD_MODEL=False
    SAVE_MODEL=True
    EXTRACT_FEATURE=False

    LR=0.1
    WEIGHT_DECAY= 5e-4
    EPOCH=15
    TEST_FREQUENCY=2
    net_kernel_sizes=[5, 3, 3, 3, 1, 1]
    net_channels=[64, 128, 256, 512, 512, 128, 128]
    net_num_classes=16
    net_in_channel=1


# p = {
#     "TRAIN_DIR": "/home/jiangyiheng/matlabProjects/xiangmu/mfcc/train/",
#     "TEST_DIR": "/home/jiangyiheng/matlabProjects/xiangmu/mfcc/test/",
#     "VERIFICATION_DIR": "/home/jiangyiheng/matlabProjects/xiangmu/mfcc/verification/",
#     "ENROLL_DIR": "/home/jiangyiheng/matlabProjects/xiangmu/mfcc/enroll/",
#     "SAVE": "/home/jiangyiheng/pycharmProjects/master/gao/model.pkl",
#     "loadModel":False,
#     "LR": 0.1,
#     "WEIGHT_DECAY": 5e-4,
#     "EPOCH":10,
#     "testFrequency":2,
#     "net_kernel_sizes": [5, 3, 3, 3, 1, 1],
#     "net_channels": [64, 128, 256, 512, 512, 128, 128],
#     "net_num_classes": 16,
#     "net_in_channel": 1,
#
# }


class DataGet(tud.Dataset):
    @ut.timing(None)
    def __init__(self, root, stdLen=250):

        files = os.listdir(root)
        files.sort()
        files = [root + i for i in files]
        self.train_data, dataSize, self.train_labels = ut.multiReadProc(files)
        self.count = len(self.train_data)
        self.maxR = max([i[1] for i in dataSize])
        self.stdLen = stdLen

    def __getitem__(self, index):
        data = self.train_data[index]
        diffR = self.stdLen - np.size(data, axis=1)
        if diffR > 0:
            while diffR:
                # if diffR larger than length of data ,data[:diffR] will return data and donnot throw exception
                data = np.concatenate([data, data[:, :diffR]], axis=1)
                diffR = self.stdLen - np.size(data, axis=1)
        elif diffR < 0:
            data = data[:, :diffR]
        data = data[np.newaxis, :, :]
        return tc.from_numpy(data).type(tc.FloatTensor), self.train_labels[index]

    def __len__(self):
        return self.count


def main():
    # global args, best_prec_eva, best_prec_train, best_eer2_eva, logname, figname, duration, test_freq
    # args = parser.parse_args()
    # args.save = args.save + '_1x' + args.filter_sizes + 'w_' + args.dilation + 'd_' + args.channels + 'c'
    # # args.imdb_path = os.path.join(args.save, args.imdb_path)
    # filter_sizes = map(int, args.filter_sizes.split('_'))
    # channels = map(int, args.channels.split('_'))
    # dilation = map(int, args.dilation.split('_'))
    # test_freq = args.test_freq
    #
    # print( args, '\n')
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # learn_rate_array = args.lr*(np.hstack((np.ones([1, args.lr_step]), 0.1*np.ones([1, max(0, (args.epochs-args.lr_step))]))))
    # logname = os.path.join(args.save, 'log'+'.txt')
    # figname = os.path.join(args.save, 'fig'+'.png')
    # if not os.path.exists(args.save):
    #     os.makedirs(args.save)
    #
    # with open(logname, 'a') as f:
    #     f.write('\n Begin Time: ')
    #     f.write(strftime("%Y-%m-%d %H:%M:%S\n", localtime()))

    # #######################need create the path of data set.################################################

    # im = tc.load(args.imdb_path)
    # if not os.path.exists(args.imdb_path):
    #     im = genRoiImdb(args.imdb_path, args.data_root, args.min_duration, args.max_duration, args.egs_num, args.gender,
    #                     args.select_num, args.sre_set, args.is_discard, args.discard_gate, logname, args.num_repeat,
    #                     args.egs_root)
    # else:
    #     im = tc.load(args.imdb_path)
    #     if args.data_root != im['data_root'] or args.min_duration != im['min_duration'] or args.max_duration != im['max_duration'] or args.egs_num != im['egs_num'] or args.gender != im['gender'] or args.select_num != im['select_num']  or args.sre_set != im['sre_set'] or args.is_discard != im['is_discard'] or args.discard_gate != im['discard_gate'] or args.num_repeat != im['num_repeat']:
    #         im = genRoiImdb(args.imdb_path, args.data_root, args.min_duration, args.max_duration, args.egs_num,
    #                         args.gender, args.select_num, args.sre_set, args.is_discard, args.discard_gate, logname,
    #                         args.num_repeat, args.egs_root)

    # ndim = im['ndim']
    # channel = im['channel']
    # spk_num = im['spk_num']
    # egs_train = im['egs_train']
    # egs_val = im['egs_val']
    # egs_num = im['egs_num']

    # nevals = eva_label.shape[0]
    # ndevs = dev_label.shape[0]
    kwargs = {'num_workers': 16, 'pin_memory': True}
    data_set = DataGet(P.TRAIN_DIR)
    train_loader = tud.DataLoader(
        data_set,
        batch_size=64, shuffle=True,
        **kwargs)
    data_set_val = DataGet(P.VALIDATION_DIR)
    val_loader = tud.DataLoader(
        data_set_val,
        batch_size=64, shuffle=False, **kwargs)

    # dev_loader = tud.DataLoader(
    #     getBatchVal(dev_list, dev_label, args.data_root),
    #     batch_size=1, shuffle=False, **kwargs)

    model = network.net(P.net_kernel_sizes, P.net_channels, P.net_num_classes, P.net_in_channel)
    # model = resnet.resnet18(num_classes=num_classes)
    # model = testnet.net(num_classes=num_classes)
    # create model
    # model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
    #                      bottleneck=args.bottleneck, dropRate=args.droprate)
    model.cuda()
    if P.LOAD_MODEL:
        model.load_state_dict(tc.load(P.SAVE))

    # if not os.path.exists(args.save):
    #     os.mkdir(args.save)

    # optionally resume from a che
    # ckpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = tc.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         args.start_ep = checkpoint['ep']
    #         best_prec_eva = checkpoint['best_prec_eva']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = tc.optim.Adam(model.parameters(), args.lr)

    optimizer = tc.optim.Adam(model.parameters(), P.LR, weight_decay=P.WEIGHT_DECAY)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 6,9], gamma=0.1)

    # with open(logname, 'a') as f:
    #     f.write('args: {0} \nnetwork: {1}\n'.format(args, model))

    for epoch in range(P.EPOCH):
        data_set.epoch = epoch
        # for it in range(args.start_ep, egs_num):
        # data_set.it = it
        # ep = epoch*egs_num + it
        # with open(logname, 'a') as f:
        #     f.write('ep:{0} '.format(ep))
        # adjust_learning_rate(optimizer, learn_rate_array, epoch)
        learning_rate_step(scheduler, epoch)
        # train for one epoch
        train(tqdm(train_loader), model, criterion, optimizer,epoch)
        if np.mod(epoch, P.TEST_FREQUENCY) == 0:
            # evaluate on validation set
            test(tqdm(val_loader), model, epoch)
            # remember best prec@1 and save checkpoint
            # is_best = prec_eva > best_prec_eva
            # is_best_train = best_prec_train > prec_train
            # best_prec_eva = max(prec_eva, best_prec_eva)
            #
            # save_checkpoint({
            #         'epoch': epoch + 1,
            #         'ep': ep + 1,
            #         'state_dict': model.state_dict(),
            #         'best_prec_eva': best_prec_eva,
            #     }, is_best, is_best_train)
            #
            # print('Best accuracy on eva: ', best_prec_eva)
            # with ut.Log('| Best Accuracy: ({0}) \n'.format(best_prec_eva), 'a'):
            #     pass
        # else:
        #     prec_eva, loss_eva = 0, 0
        # plot figure
        # plot_figure(prec_train, loss_train, prec_eva, loss_eva)
            modelSave(model.state_dict(),epoch)

def modelSave(save,epoch):
    if P.SAVE_MODEL:
        if os._exists(P.SAVE):
            os.system("rm " + P.SAVE)
        tc.save(save, P.SAVE+str(epoch))
# def plot_figure(epoch, prec_train, loss_train, prec_eva, loss_eva):
#
#     global acc_train_array, loss_train_array, acc_eva_array, loss_eva_array
#     if epoch == args.start_epoch:
#         # for plot figure
#         acc_train_array = np.empty([1, 0])
#         loss_train_array = np.empty([1, 0])
#         acc_eva_array = np.empty([1, 0])
#         loss_eva_array = np.empty([1, 0])
#     # plot figure
#     acc_train_array = np.hstack((acc_train_array, np.array(prec_train).reshape(1,1)))
#     loss_train_array = np.hstack((loss_train_array, np.array(loss_train).reshape(1,1)))
#     if np.mod(epoch, test_freq) == 0:
#         acc_eva_array = np.hstack((acc_eva_array, np.array(prec_eva).reshape(1,1)))
#         loss_eva_array = np.hstack((loss_eva_array, np.array(loss_eva).reshape(1,1)))
#
#     fig = plt.figure(1)
#     fig.clear()
#     plt.subplot(1, 2, 1)
#     plt.plot(np.arange(0, loss_train_array.size), loss_train_array[0])
#     plt.plot(np.arange(0, loss_eva_array.size*test_freq, step=test_freq), loss_eva_array[0])
#     plt.title('loss')
#     plt.legend(['train', 'val'], loc=2)
#     plt.grid(True)
#     plt.subplot(1, 2, 2)
#     plt.plot(np.arange(0, acc_train_array.size), acc_train_array[0])
#     plt.plot(np.arange(0, acc_eva_array.size*test_freq, step=test_freq), acc_eva_array[0])
#     plt.title('acc')
#     plt.legend(['train', 'val'], loc=2)
#     plt.grid(True)
#
#     #plt.show(block=False)
#     fig.savefig(figname)
@ut.timing(None)
def train(train_loader, model, criterion, optimizer,epoch):
    # type: (object, object, object, object, object) -> object
    """Train for one epoch on the training set"""
    # losses = AverageMeter()
    # top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(async=True)
        input = input.cuda()
        input_var = tc.autograd.Variable(input)
        target_var = tc.autograd.Variable(target)

        # compute output
        (output, embedding1, embedding2) = model(input_var)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target, topk=(1,))[0]
        # losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            pred_y = tc.max(output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == target) / float(target.size(0))
            with ut.Log("[train] epoch:%d | Loss:%f | accuracy:%f\n" % (epoch,loss, accuracy)):
                pass
        # if i % 100 == 0:
        #     print('Training\n Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'accuracy@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))

        # measure elapsed time

    # batch_time.update(time.time() - end)
    with ut.Log("[train] epoch :%d finished | Loss:%f | accuracy:%f\n" % (epoch,loss, accuracy)):
        pass
    # print('Training\n Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))
    #
    # # if args.save:
    # with ut.Log('Training: '
    #             'Loss {loss.avg:.4f}, '
    #             'Prec@1 {top1.avg:.3f} '.format(loss=losses,
    #                                             top1=top1)):
    #     pass
    # return top1.avg, losses.avg


def test(train_loader, model, epoch):
    """Test for one epoch on the training set"""
    # batch_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()

    # switch to test mode
    model.eval()
    right = 0
    total = 0
    ss=[]
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = tc.autograd.Variable(input)
        target_var = tc.autograd.Variable(target)

        # compute output
        (output, embedding1, embedding2) = model(input_var)
        if isinstance(epoch,str):
            #extract data
            target.cpu()
            tar=tc.autograd.Variable(target.type(tc.FloatTensor).resize_((len(target),1))).cuda()
            ss.append(torch.cat((tar,embedding1, embedding2), 1).data.cpu())
        else:
            output=functional.log_softmax(output,1)
            # measure accuracy and record loss
            pred_y = tc.max(output, 1)[1].data.squeeze()
            right = right + sum(pred_y == target)
            total = total + target.size(0)
    if isinstance(epoch,str):
        name = re.findall(".*mfcc/([a-z]*)/", epoch)[0]
        np.savetxt("./" + name + ".txt", np.concatenate(ss, 0))
    else:
        accuracy = right / total
        strings="[=========test=========] | epoch:%d | accuracy:%f\n" % (epoch, accuracy)
        with ut.Log(strings):
            print(strings)
        #
        # prec1 = accuracy(output.data, target, topk=(1,))[0]
        # losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))

    # batch_time.update(time.time() - end)
    # end = time.time()
    #
    # print('Test\n Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))


# def save_checkpoint(state, is_best, is_best_train, filename='checkpoint.pth.tar'):
#     """Saves checkpoint to disk"""
#     filename = os.path.join(args.save, filename)
#     bfilename = os.path.join(args.save, 'model_best.pth.tar')
#     bfilename_eer2 = os.path.join(args.save, 'model_best_train_acc.pth.tar')
#     tc.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, bfilename)
#     if is_best_train:
#         shutil.copyfile(filename, bfilename_eer2)


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


def learning_rate_step(scheduler, epoch):
    scheduler.step(epoch)
    lr = scheduler.get_lr()
    print('Learning rate:{0}\n'.format(lr))
    # print('Epoch:{0}, ' 'Learning rate:{1} |'.format(epoch, lr))
    with ut.Log('Learning rate:{0}\n'.format(lr)):
        pass

@ut.timing("extract_feature")
def extract_feature():
    model = network.net(P.net_kernel_sizes, P.net_channels, P.net_num_classes, P.net_in_channel)
    model.cuda()
    model.load_state_dict(tc.load(P.SAVE))
    kwargs = {'num_workers': 16, 'pin_memory': True}
    for i in [P.TRAIN_DIR,P.VALIDATION_DIR,P.ENROLL_DIR,P.TEST_DIR]:
        data_set = DataGet(i)
        val_loader = tud.DataLoader(data_set,batch_size=64, shuffle=False,**kwargs)
        test(tqdm(val_loader), model, i)

if __name__ == '__main__':
    with ut.Log("================================================================================\n"):
        # main()
        extract_feature()