from __future__ import division
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from gao.getbatch import getBatchTrn, getBatchVal
import os
import shutil
import time
from time import strftime, localtime
import numpy as np
#from eer import cal_eer_matlab
import matplotlib.pyplot as plt
import argparse
from gao import network, lr_scheduler
from tqdm import tqdm

# import network_kaldi


parser = argparse.ArgumentParser(description='PyTorch LID-net training')
parser.add_argument('--imdb-path', default='./nist04_nist05_nist06_nist08_swbd1_swbd2_swbdCellP2_70e_50r_imdb_mfcc.pkl', type=str,
                    help='imdb of index of data')
parser.add_argument('--data-root', default='/home/gaozf/SRE/data/mfcc23_vad_mat/train', type=str,
                    help='root dir of data')
# parser.add_argument('--num-classes', default=10, type=int,
#                     help='num_classes')
# parser.add_argument('--time', default='3s', type=str,
#                     help='how long time of data')
parser.add_argument('--min-duration', default=200, type=int,
                    help='min_duration: 3000, 1000, 300')
parser.add_argument('--max-duration', default=400, type=int,
                    help='min_duration: 3000, 1000, 300')
parser.add_argument('--egs-root', default='/home/gaozf/SRE/data/nist04_nist05_nist06_nist08_swbd1_swbd2_swbdCellP2_70e_50r_egs', type=str,
                    help='egs_root: 50')
parser.add_argument('--egs-num', default=70, type=int,
                    help='egs_num: 50-150')
parser.add_argument('--num-repeat', default=50, type=int,
                    help='num-repeat: 50')
parser.add_argument('--gender', default='f_m', type=str,
                    help='gender: male(m,M), female(f,F), f_m')
parser.add_argument('--sre-set', default='nist04_nist05_nist06_nist08_swbd1_swbd2_swbdCellP2', type=str,
                    help='sre sets: nist04_nist05_nist06_nist08_nist10_swbd1_swbd2_swbdCellP2')
parser.add_argument('--select_num', default=8, type=int,
                    help='select_num: select the spk whos utt num >= select_num')
parser.add_argument('--is-discard', default=True, type=bool,
                    help='whether to discard utterance which is shorter than discard_gate')
parser.add_argument('--discard-gate', default=1000, type=int,
                    help='whether to discard utterance which is shorter than discard_gate')
parser.add_argument('--filter-sizes', default='5_3_3_1_1_1_1', type=str,
                    help='the width of filter size: 3, 5,7 ..')
parser.add_argument('--channels', default='512_512_512_512_1636_512_300', type=str,
                    help='the numbers of filter out channels: 64, 128, 256 ..')
parser.add_argument('--dilation', default='1_2_4_1_1_1_1', type=str,
                    help='the dilation of filter: 1, 1, 2, 4, ..')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--epochs', default=6, type=int,
                    help='number of total epochs to run')
parser.add_argument('--test-freq', default=1, type=int,
                    help='how many train epoch test once:1')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start-ep', default=0, type=int,
                    help='manual ep number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--lr-step', default=2, type=int,
                    help='step decay learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--num-workers', default=12, type=int,
                    help='num-workers:1')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id: 0')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='results', type=str,
                    help='save parameters and logs in this folder')
best_prec_eva = 0.
best_prec_train = 0.
best_eer2_eva = 1.



def main():
    global args, best_prec_eva, best_prec_train, best_eer2_eva, logname, figname, duration, test_freq
    args = parser.parse_args()
    args.save = args.save + '_1x' + args.filter_sizes + 'w_' + args.dilation + 'd_' + args.channels + 'c'
    # args.imdb_path = os.path.join(args.save, args.imdb_path)
    filter_sizes = map(int, args.filter_sizes.split('_'))
    channels = map(int, args.channels.split('_'))
    dilation = map(int, args.dilation.split('_'))
    test_freq = args.test_freq

    print( args, '\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    learn_rate_array = args.lr*(np.hstack((np.ones([1, args.lr_step]), 0.1*np.ones([1, max(0, (args.epochs-args.lr_step))]))))
    logname = os.path.join(args.save, 'log'+'.txt')
    figname = os.path.join(args.save, 'fig'+'.png')
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    with open(logname, 'a') as f:
        f.write('\n Begin Time: ')
        f.write(strftime("%Y-%m-%d %H:%M:%S\n", localtime()))

    # #######################need create the path of data set.################################################

    im = torch.load(args.imdb_path)
    # if not os.path.exists(args.imdb_path):
    #     im = genRoiImdb(args.imdb_path, args.data_root, args.min_duration, args.max_duration, args.egs_num, args.gender,
    #                     args.select_num, args.sre_set, args.is_discard, args.discard_gate, logname, args.num_repeat,
    #                     args.egs_root)
    # else:
    #     im = torch.load(args.imdb_path)
    #     if args.data_root != im['data_root'] or args.min_duration != im['min_duration'] or args.max_duration != im['max_duration'] or args.egs_num != im['egs_num'] or args.gender != im['gender'] or args.select_num != im['select_num']  or args.sre_set != im['sre_set'] or args.is_discard != im['is_discard'] or args.discard_gate != im['discard_gate'] or args.num_repeat != im['num_repeat']:
    #         im = genRoiImdb(args.imdb_path, args.data_root, args.min_duration, args.max_duration, args.egs_num,
    #                         args.gender, args.select_num, args.sre_set, args.is_discard, args.discard_gate, logname,
    #                         args.num_repeat, args.egs_root)

    ndim = im['ndim']
    channel = im['channel']
    spk_num = im['spk_num']
    egs_train = im['egs_train']
    egs_val = im['egs_val']
    egs_num = im['egs_num']

    #nevals = eva_label.shape[0]
    #ndevs = dev_label.shape[0]
    num_classes = spk_num

    data_set = getBatchTrn(egs_train, ndim, args.data_root, egs_num)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
                    data_set,
                    batch_size=args.batch_size, shuffle=True,
                    **kwargs)
    data_set_val = getBatchVal(egs_val, ndim, args.data_root, 1)
    val_loader = torch.utils.data.DataLoader(
                    data_set_val,
                    batch_size=args.batch_size, shuffle=False,**kwargs)

    # dev_loader = torch.utils.data.DataLoader(
    #     getBatchVal(dev_list, dev_label, args.data_root),
    #     batch_size=1, shuffle=False, **kwargs)

    model = network.net(filter_sizes, channels, dilation, num_classes, ndim, channel)
    # model = resnet.resnet18(num_classes=num_classes)
    # model = testnet.net(num_classes=num_classes)
    # create model
    # model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
    #                      bottleneck=args.bottleneck, dropRate=args.droprate)
    model.cuda()

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    # optionally resume from a che
    # ckpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.start_ep = checkpoint['ep']
            best_prec_eva = checkpoint['best_prec_eva']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_step, args.lr_step + 1, args.lr_step + 2], gamma=0.1)
    with open(logname, 'a') as f:
        f.write('args: {0} \nnetwork: {1}\n'.format(args, model))

    for epoch in range(args.start_epoch, args.epochs):
        data_set.epoch = epoch
        for it in range(args.start_ep, egs_num):
            data_set.it = it
            ep = epoch*egs_num + it
            with open(logname, 'a') as f:
                f.write('ep:{0} '.format(ep))
            # adjust_learning_rate(optimizer, learn_rate_array, epoch)
            learning_rate_step(scheduler, epoch)
            # train for one epoch
            prec_train, loss_train = train(tqdm(train_loader), model, criterion, optimizer, ep)
            if np.mod(ep, test_freq) == 0:
                # evaluate on validation set
                prec_eva, loss_eva = test(tqdm(val_loader), model, criterion, ep)
                # remember best prec@1 and save checkpoint
                is_best = prec_eva > best_prec_eva
                is_best_train = best_prec_train > prec_train
                best_prec_eva = max(prec_eva, best_prec_eva)

                save_checkpoint({
                        'epoch': epoch + 1,
                        'ep': ep + 1,
                        'state_dict': model.state_dict(),
                        'best_prec_eva': best_prec_eva,
                    }, is_best, is_best_train)

                print('Best accuracy on eva: ', best_prec_eva)
                with open(logname, 'a') as f:
                        f.write('| Best Accuracy: ({0}) \n'.format(best_prec_eva))
            else:
                prec_eva, loss_eva = 0, 0
            # plot figure
            plot_figure(ep, prec_train, loss_train, prec_eva, loss_eva)

def plot_figure(epoch, prec_train, loss_train, prec_eva, loss_eva):

    global acc_train_array, loss_train_array, acc_eva_array, loss_eva_array
    if epoch == args.start_epoch:
        # for plot figure
        acc_train_array = np.empty([1, 0])
        loss_train_array = np.empty([1, 0])
        acc_eva_array = np.empty([1, 0])
        loss_eva_array = np.empty([1, 0])
    # plot figure
    acc_train_array = np.hstack((acc_train_array, np.array(prec_train).reshape(1,1)))
    loss_train_array = np.hstack((loss_train_array, np.array(loss_train).reshape(1,1)))
    if np.mod(epoch, test_freq) == 0:
        acc_eva_array = np.hstack((acc_eva_array, np.array(prec_eva).reshape(1,1)))
        loss_eva_array = np.hstack((loss_eva_array, np.array(loss_eva).reshape(1,1)))

    fig = plt.figure(1)
    fig.clear()
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, loss_train_array.size), loss_train_array[0])
    plt.plot(np.arange(0, loss_eva_array.size*test_freq, step=test_freq), loss_eva_array[0])
    plt.title('loss')
    plt.legend(['train', 'val'], loc=2)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, acc_train_array.size), acc_train_array[0])
    plt.plot(np.arange(0, acc_eva_array.size*test_freq, step=test_freq), acc_eva_array[0])
    plt.title('acc')
    plt.legend(['train', 'val'], loc=2)
    plt.grid(True)

    #plt.show(block=False)
    fig.savefig(figname)

def train(train_loader, model, criterion, optimizer, epoch):
    # type: (object, object, object, object, object) -> object
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        (output, embedding1, embedding2) = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Training\n Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))

        # measure elapsed time

    batch_time.update(time.time() - end)
    end = time.time()

    print('Training\n Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))

    if args.save:
        with open(logname, 'a') as f:
            f.write('Training: '
                    'Time {batch_time.val:.3f}, '
                    'Loss {loss.avg:.4f}, '
                    'Prec@1 {top1.avg:.3f} '.format(batch_time=batch_time, loss=losses,
                                                    top1=top1))
    return top1.avg, losses.avg


def test(train_loader, model, criterion, epoch):
    # type: (object, object, object, object, object) -> object
    """Test for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to test mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        (output, embedding1, embedding2) = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))


    batch_time.update(time.time() - end)
    end = time.time()

    print('Test\n Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))

    if args.save:
        with open(logname, 'a') as f:
            f.write('| Test: '
                    'Time {batch_time.val:.3f}, '
                    'Loss {loss.avg:.4f}, '
                    'Prec@1 {top1.avg:.3f} '.format(batch_time=batch_time, loss=losses,
                                                    top1=top1))
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, is_best_train, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    filename = os.path.join(args.save, filename)
    bfilename = os.path.join(args.save, 'model_best.pth.tar')
    bfilename_eer2 = os.path.join(args.save, 'model_best_train_acc.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bfilename)
    if is_best_train:
        shutil.copyfile(filename, bfilename_eer2)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, learn_rate_array, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    #lr = args.lr * ((0.5 ** int(epoch >= 10)) * (0.5 ** int(epoch >= 40)) * (0.5 ** int(epoch >= 80)))
    #lr = args.lr
    #lr = learn_rate_array[0, epoch]
    if epoch < args.lr_step:
        lr = args.lr
        momentum = args.momentum
    else:
        lr = args.lr * (1- (epoch - args.lr_step)/(args.epochs - args.lr_step) )
        momentum = 0.5
    print('Epoch:{0}, ' 'Learning rate:{1} |'.format(epoch, lr))
    if args.save:
        with open(logname, 'a') as f:
            f.write('Epoch:{0}, ' 'Learning rate:{1} |'.format(epoch,lr))
        f.close()

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['momentum'] = momentum

def learning_rate_step(scheduler, epoch):
    scheduler.step(epoch)
    lr = scheduler.get_lr()
    print('Learning rate:{0} |'.format(lr))
    # print('Epoch:{0}, ' 'Learning rate:{1} |'.format(epoch, lr))
    if args.save:
        with open(logname, 'a') as f:
            f.write('Learning rate:{0} |'.format(lr))
        f.close()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
