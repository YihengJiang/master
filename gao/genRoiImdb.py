import scipy.io as io
import numpy as np
import torch


def genRoiImdb(imdb_path, data_root, time, duration, stride):
    # im = torch.load(imdb_path)
    im = io.loadmat('/home/jiangyiheng/matlabProjects/xiangmu/mfcc/01_35240000_2015_03_20_07.htk_1_.8.mat', squeeze_me=True, struct_as_record=False)['ff']
    nutters = im.key.train.size
    rois = np.empty([0, 3], dtype=np.int32)

    global c
    train_list = np.empty([0, 1], dtype=np.object)
    for i in range(nutters):
        utterance_name = data_root + im.list.train[i]
        utterance = io.loadmat(utterance_name)['out']
        # utterance = utterance.reshape(1, utterance.shape[0], utterance.shape[1])
        h, w, c = utterance.shape
        temp_list = np.array(utterance_name, dtype=np.object)

        # 30s
        if w < duration:
            print (('%s is shorter than %d ' % (im.list.train[i], duration)))
            continue
        roi = np.empty([0, 3], dtype=np.int64)
        for st_frm in range(0, w, stride):
            end_frm = min(st_frm + duration, w)
            if (end_frm - st_frm) == duration:
                temp = np.array([i, st_frm, end_frm], dtype=np.int64)
                roi = np.vstack((roi, temp))
            elif (end_frm - st_frm) > duration / 2:
                temp = np.array([i, max(0, end_frm - duration), end_frm], dtype=np.int64)
                roi = np.vstack((roi, temp))
                break

        rois = np.vstack((rois, roi))
        train_list = np.vstack((train_list, temp_list))

    dev_arr = {'1s': im.list.dev_1s,
               '3s': im.list.dev_3s,
               'all': im.list.dev_all
               }
    dev_key_arr = {'1s': im.key.dev_1s,
                   '3s': im.key.dev_3s,
                   'all': im.key.dev_all
                   }
    featrue = {}
    featrue['data_root'] = data_root
    featrue['time'] = time
    featrue['duration'] = duration
    featrue['stride'] = stride
    featrue['train_list'] = train_list.reshape(train_list.size)
    featrue['train_label'] = im.key.train - 1
    featrue['dev_list'] = data_root + dev_arr[time]
    featrue['dev_label'] = dev_key_arr[time] - 1
    featrue['eva_list'] = []
    featrue['eva_label'] = []
    featrue['rois'] = rois
    featrue['rois_label'] = im.key.train[rois[:, 0]] - 1
    featrue['ndim'] = c

    torch.save(featrue, imdb_path)
    print ('generate imdb have done\n')
    return featrue


def test():
    imdb_path = './imdb_AP17_DBF.pkl'
    data_root = '/home/gaozf/disk/gaozf/AP17/data/bn_mat_50c_pad_300/'
    stride = 100
    duration = 280
    time = '3s'
    im = genRoiImdb(imdb_path, data_root, time, duration, stride)
    a = 1


if __name__ == '__main__':
    test()
