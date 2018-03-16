from __future__ import division
import torch
import os
import random
import scipy.io as scio
import multiprocessing

def select_list_num(utt2key_dict, num_spk):

    key2utt_dict = utt2key_to_key2utt(utt2key_dict)
    key2utt_dict_new = {}
    key2utt_dict_delete = {}
    for i, (key, utts) in enumerate(key2utt_dict.items()):
        if len(utts) < num_spk:
            # print('delete %s, %s\n' % (spkid, list[spkid]))
            key2utt_dict_delete[key] = utts
            continue
        key2utt_dict_new[key] = utts

    utt2key_dict_new = key2utt_to_utt2key(key2utt_dict_new)
    utt2key_dict_delete = key2utt_to_utt2key(key2utt_dict_delete)
    print('select speaker number:\n')
    print('\tbefore select numbers >= %d, utterance number: %d, speaker number: %d\n' % (num_spk, cal_num_of_utt_spk(utt2key_dict)[0], cal_num_of_utt_spk(utt2key_dict)[1]))
    print('\tafter select numbers >= %d, utterance number: %d, speaker number: %d\n' % (num_spk, cal_num_of_utt_spk(utt2key_dict_new)[0], cal_num_of_utt_spk(utt2key_dict_new)[1]))
    return utt2key_dict_new, utt2key_dict_delete

def utt2key_to_key2utt(utt2key_dict):
    key2utt_dict = {}
    for i, (utt, key) in enumerate(utt2key_dict.items()):
        if key in key2utt_dict:
            key2utt_dict[key].append([utt])
        else:
            key2utt_dict[key] = [[utt]]
    return key2utt_dict

def key2utt_to_utt2key(key2utt_dict):

    utt2key_dict = {}
    for i, (key, utts) in enumerate(key2utt_dict.items()):
        for ix, utt in enumerate(utts):
            utt2key_dict[utt[0]] = key

    return utt2key_dict

def cal_num_of_utt_spk(utt2key_dict):

    return len(utt2key_dict), len(utt2key_to_key2utt(utt2key_dict))

def select_list_gender(utt2key_dict):

    utt2key_male_dict = {}
    utt2key_female_dict = {}
    for i, (utt, key) in enumerate(utt2key_dict.items()):

        if 'm' in utt.split('-'):

            if utt in utt2key_male_dict:
                utt2key_male_dict[utt].extend(key)
            else:
                utt2key_male_dict[utt] = key
        elif 'f' in utt.split('-'):

            if utt in utt2key_female_dict:
                utt2key_female_dict[utt].extend(key)
            else:
                utt2key_female_dict[utt] = key
    print('select gender:\n')
    print('\tmale: utterance numbers: %d, speaker numbers: %d\n' % (cal_num_of_utt_spk(utt2key_male_dict)))
    print('\tfemale: utterance numbers: %d, speaker numbers: %d\n' % (cal_num_of_utt_spk(utt2key_female_dict)))

    return utt2key_male_dict, utt2key_female_dict
def select_sre_set(utt2key_dict, sre_set):

    sets = sre_set.split('_')
    sets_del = []
    utt2key_dict_new = {}
    for i, (utt, key) in enumerate(utt2key_dict.items()):
        data_set = utt.split('/')[-1].split('-')[1]
        if data_set in sets:
            utt2key_dict_new[utt] = key
        else:
            sets_del.append(data_set)

    print('select sre data sets: %s\n' % sets)
    print('\tbefore select sre sets, utterance number: %d, speaker number: %d\n' % (cal_num_of_utt_spk(utt2key_dict)[0], cal_num_of_utt_spk(utt2key_dict)[1]))
    print('\tafter select sre sets, utterance number: %d, speaker number: %d\n' % (cal_num_of_utt_spk(utt2key_dict_new)[0], cal_num_of_utt_spk(utt2key_dict_new)[1]))
    return utt2key_dict_new

def spkid_to_key(utt2spk_dict):

    spk2utt_dict = utt2key_to_key2utt(utt2spk_dict)
    spkid2key_dict = {}
    key2spkid_dict = {}
    for i, spkid in enumerate(sorted(spk2utt_dict.keys())):
        spkid2key_dict[spkid] = i
        key2spkid_dict[i] = spkid

    utt2key_dict = {}
    for i, record in enumerate(sorted(utt2spk_dict.keys())):
        spkid = utt2spk_dict[record]
        utt2key_dict[record] = spkid2key_dict[spkid]
    return utt2key_dict, spkid2key_dict, key2spkid_dict

def utt2key_to_utt2spkid(utt2key_dict, key2spkid_dict):

    utt2spkid_dict = {}
    for i, (utt, key) in enumerate(utt2key_dict.items()):
        utt2spkid_dict[utt] = key2spkid_dict[key]

    return utt2spkid_dict

def select_list_num_fr_utt2roi(utt2roi_dict, select_num):

    spkid2utt, spkid2rois = {}, {}
    for i, (utt, rois) in enumerate(utt2roi_dict.items()):

        if rois[0][0] in spkid2utt:
            spkid2utt[rois[0][0]].append([utt])
            spkid2rois[rois[0][0]].append(rois)
        else:
            spkid2utt[rois[0][0]] = [[utt]]
            spkid2rois[rois[0][0]] = [rois]
    spkid2rois_new = {}
    spkid2rois_delete = {}
    for i, (spkid, rois) in enumerate(spkid2rois.items()):
        if len(rois) < select_num:
            spkid2rois_delete[spkid] = rois
        else:
            spkid2rois_new[spkid] = rois

    utt2rois_dict, rois_list = {}, []
    for i, (spkid, rois) in enumerate(spkid2rois_new.items()):

        for roi in enumerate(rois):
            rois_list.append(roi)
            if roi[1] in utt2rois_dict:
                utt2rois_dict[roi[1]].append(roi[1])
            else:
                utt2rois_dict[roi[1]] = [roi[1]]

    return

def get_data_list(data_root):
    data_list = []
    for root, dirs, files in os.walk(data_root):
        for file in sorted(files):
            if '.mat' in file:
                data_list.append(os.path.join(root, file))

    return data_list

def gen_utt2spkid_dict(data_list):
    utt2key_dict = {}

    for i, item in enumerate(data_list):
        items = item.split('/')[-1].split('-')
        spkid = items[0]
        utt2key_dict[item] = spkid

    return utt2key_dict
def gen_utt2len(utt2key_dict, egs_root):

    if os.path.exists(os.path.join(egs_root, 'utt2len')):
        utt2len = {}
        utt2len_f = open(os.path.join(egs_root, 'utt2len'), 'r')
        utt2len_list = utt2len_f.read().split('\n')
        if '' in utt2len_list:
            utt2len_list.remove('')

        for i, item in enumerate(utt2len_list):
            items= item.split(' ')
            utt = items[0]
            l = items[1]
            utt2len[utt] = int(l)
        utt2len_f.close()
        return utt2len

    utt2len = {}
    for i, utt in enumerate(utt2key_dict.keys()):
        if i % 10 == 0:
            print('%d/%d' % (i, len(utt2key_dict.keys())))
        data = scio.loadmat(utt)['out']
        w, h = data.shape
        utt2len[utt] = w
    utt2len_f = open(os.path.join(egs_root, 'utt2len'), 'w')
    for i, (utt, l) in enumerate(utt2len.items()):
        utt2len_f.write(utt + ' ' + str(l) + '\n')
    utt2len_f.close()
    return utt2len
def gen_egs_val(key2utt, utt2len, duration, egs_dir):
    key2utt_new = {}
    key2utt_val = {}
    for i, (key, utts) in enumerate(key2utt.items()):
        utt_s = []
        utt_s.extend(x[0] for x in utts if utt2len[x[0]] > duration)
        if len(utt_s) == 0:
            continue
        index = random.randint(0, len(utt_s) - 1)
        utt_val = utt_s[index]
        key2utt_val[key] = [[utt_val]]
        key2utt_new[key] = []
        key2utt_new[key].extend(x for x in utts if x[0] != utt_val)

    key_list = key2utt_val.keys()*2
    # split_data(key_list, key2utt_val, duration, utt2len, egs_dir, 0, 0)
    nj = 6
    n = len(key_list) // nj
    key_list_s = []
    for i in range(nj):
        key_list_s.append(key_list[i * n:(i + 1) * n])
    key_list_s.append(key_list[nj * n:])

    muti_pro = []
    for i, utt_ss in enumerate(key_list_s):
        # split_data(utt_ss, key2utt_val, duration, utt2len, egs_dir, 0, i)
        muti_pro.append(multiprocessing.Process(target=split_data, args=(utt_ss, key2utt_val, duration, utt2len, egs_dir, 0, i)))
    for proc in muti_pro:
        proc.start()

    return key2utt_new
def gen_egs(utt2key_dict, min_duration, max_duration, egs_num, num_repeat, key2spkid_dict, egs_root):

    utt2len = gen_utt2len(utt2key_dict, egs_root)
    key2utt = utt2key_to_key2utt(utt2key_dict)
    key2utt = gen_egs_val(key2utt, utt2len, (min_duration + max_duration)/2, os.path.join(egs_root, 'val'))

    egs_duration = []
    # num_utt = len(key2utt) * num_repeat
    key_list = key2utt.keys() * num_repeat
    nj = 8
    n = len(key_list)//nj
    key_list_s = []
    for i in range(nj):
        key_list_s.append(key_list[i*n:(i+1)*n])
    key_list_s.append(key_list[nj*n:])

    for it in range(egs_num):

        duration = random.randint(min_duration, max_duration)
        egs_duration.append(duration)


        muti_pro = []
        for i, utt_ss in enumerate(key_list_s):
            # split_data(utt_ss, key2utt, duration, utt2len, os.path.join(egs_root, 'train'), it, i)
            muti_pro.append(multiprocessing.Process(target=split_data, args=(utt_ss, key2utt, duration, utt2len, os.path.join(egs_root, 'train'), it, i)))
        for proc in muti_pro:
            proc.start()

    return egs_duration

def split_data(key_list, key2utt, duration, utt2len, egs_root, it, pid):
    print('start pid %d' % pid)
    for i, key in enumerate(key_list):
        try:
            utt = get_random_utt(key2utt, key, duration, utt2len)
        except:
            continue
        gen_random_split(utt, key, duration, egs_root, it, pid, i)
    print('finish pid %d' % pid)

def gen_random_split(utt, key, duration, egs_root, it, pid, i):
    data = scio.loadmat(utt)['out']
    w, h = data.shape
    free_length = w - duration
    str_ = random.randint(0, free_length)
    end_ = int(str_ + duration)
    out = data[str_:end_, :]
    out_dir = os.path.join(egs_root, 'egs_' + str(it))
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except:
            pass
    name = utt.split('/')[-1]
    out_name = os.path.join(out_dir, str(key) + '-' + name.replace('.mat', '-' + str(w) + '-' + str(it) + '-' + str(duration) + '-' + str(str_) + '-' + str(end_) + '-' + str(pid) + '-' + str(i) + '.mat'))
    scio.savemat(out_name, {'out': out, 'w': out.shape[0], 'h': out.shape[1], 'key': key}, appendmat=False, format='4')
    return

def get_random_offset(utt, duration):
    data = scio.loadmat(utt)['out']
    w, h = data.shape
    if duration > w:
        import sys
        sys.exit("code error: duration > w")
    free_length = w - duration

    offset = random.randint(0, free_length)
    return offset, offset + duration
def get_random_utt(key2utt, key, duration, utt2len):
    utts = key2utt[key]
    utt_s = []

    utt_s.extend(x[0] for x in utts if utt2len[x[0]] > duration)
    if len(utt_s) == 0:
        print('spkid: %d is short than %d' % (key, duration))
        exit()
    index = random.randint(0, len(utt_s)-1)
    return utt_s[index]

def get_egs_list(egs_root):
    egs = []
    for root, dirs, files in sorted(os.walk(egs_root)):
        if 'egs_' in root:
            egs_cur = []
            for file in sorted(files):
                if '.mat' in file:
                    record = os.path.join(root, file)
                    key = file.split('-')[0]
                    spkid = file.split('-')[1]
                    egs_cur.append([key, spkid, record])

            egs.append(egs_cur)
    return egs
def select_list_duration(utt2spkid_dict, max_duration, egs_root):
    utt2len = gen_utt2len(utt2spkid_dict, egs_root)
    utt2spkid_dict_new = {}
    utt2spkid_dict_delete = {}
    for i, (utt, spkid) in enumerate(utt2spkid_dict.items()):
        if utt2len[utt] >= max_duration:
            utt2spkid_dict_new[utt] = spkid
        else:
            utt2spkid_dict_delete[utt] = spkid
    print('select duration: >= %d \n' % max_duration)
    print ('\tbefore select duration, utterance number: %d, speaker number: %d\n' % (
    cal_num_of_utt_spk(utt2spkid_dict)[0], cal_num_of_utt_spk(utt2spkid_dict)[1]))
    print ('\tafter select duration, utterance number: %d, speaker number: %d\n' % (
    cal_num_of_utt_spk(utt2spkid_dict_new)[0], cal_num_of_utt_spk(utt2spkid_dict_new)[1]))
    return utt2spkid_dict_new, utt2spkid_dict_delete

def genRoiImdb(imdb_path, data_root, min_duration, max_duration, egs_num, gender, select_num, sre_set, is_discard, discard_gate, logname, num_repeat = 50, egs_root='/home/gaozf/SRE/data/mfcc23_mat/egs/'):

    print('start generate imdb ... \n')
    #im = torch.load(imdb_path)
    data_list = get_data_list(data_root)
    utt2spkid_dict = gen_utt2spkid_dict(data_list)
    utt2spkid_dict_total = utt2spkid_dict.copy()

    # select sre_set num gender
    utt2spkid_dict = select_sre_set(utt2spkid_dict, sre_set)
    utt2spkid_male_dict, utt2spkid_female_dict = select_list_gender(utt2spkid_dict)
    if gender == 'm' or gender == 'M':
        print('\tselected male!\n')
        utt2spkid_dict = utt2spkid_male_dict
    elif gender == 'f' or gender == 'F':
        print('\tselected female!\n')
        utt2spkid_dict = utt2spkid_female_dict
    else:
        print('\tselected male and female!\n')

    utt2spkid_dict, utt2spkid_dict_delete = select_list_num(utt2spkid_dict, select_num)
    utt2spkid_dict, utt2spkid_dict_delete = select_list_duration(utt2spkid_dict, max_duration, egs_root)
    # spkid to key
    utt2key_dict, spkid2key_dict, key2spkid_dict = spkid_to_key(utt2spkid_dict)

    # egs_duration = gen_egs(utt2key_dict, min_duration, max_duration, egs_num, num_repeat, key2spkid_dict, egs_root)
    egs_train = get_egs_list(os.path.join(egs_root, 'train'))
    egs_val = get_egs_list(os.path.join(egs_root, 'val'))

    featrue = {}
    featrue['data_root'] = data_root
    featrue['min_duration'] = min_duration
    featrue['max_duration'] = max_duration
    # featrue['egs_duration'] = egs_duration
    featrue['egs_num'] = egs_num
    featrue['ndim'] = 23
    featrue['channel'] = 1
    featrue['spk_num'] = cal_num_of_utt_spk(utt2key_dict)[1]
    featrue['utt_num'] = cal_num_of_utt_spk(utt2key_dict)[0]
    featrue['is_discard'] = is_discard
    featrue['gender'] = gender
    featrue['select_num'] = select_num
    featrue['sre_set'] = sre_set
    featrue['is_discard'] = is_discard
    featrue['discard_gate'] = discard_gate
    featrue['utt2key_dict'] = utt2key_dict
    featrue['spkid2key_dict'] = spkid2key_dict
    featrue['egs_train'] = egs_train
    featrue['egs_val'] = egs_val
    featrue['egs_root'] = egs_root
    featrue['num_repeat'] = num_repeat


    torch.save(featrue, imdb_path)
    print('generate imdb have done\n')
    print('Train set: total number of utterance: %d, discard number: %d, actual number: %d\n' % (len(data_list), len(data_list) - len(egs_train[0]), len(egs_train[0])))
    print('Train set: total number of speaker: %d, discard number: %d, actual number: %d\n' % (cal_num_of_utt_spk(utt2spkid_dict_total)[1], cal_num_of_utt_spk(utt2spkid_dict_total)[1] - cal_num_of_utt_spk(utt2key_dict)[1], cal_num_of_utt_spk(utt2key_dict)[1]))

    # with open(logname, 'a') as f:
    #     f.write('number of utterance: {0}, number of speaker: {1}\n'
    #             .format(len(egs_train[0]), cal_num_of_utt_spk(utt2key_dict)[1]))
    # f.close
    return featrue

def test():

    imdb_path = './sre_imdb_mfcc.pkl'
    data_root = '/home/gaozf/SRE/data/mfcc23_vad_mat/train'
    gender = 'm_f'
    select_num = 8
    # sre_set = 'nist04_nist05_nist06_nist08_nist10_swbd1_swbd2_swbdCellP2'
    sre_set = 'nist10'
    is_discard = True
    discard_gate = 200
    min_duration = 200
    max_duration = 400
    egs_root = '/home/gaozf/SRE/data/nist10_egs/'
    logname = 'egs_log.txt'
    egs_num = 40
    num_repeat = 50

    genRoiImdb(imdb_path, data_root, min_duration, max_duration, egs_num, gender, select_num, sre_set, is_discard, discard_gate, logname, num_repeat, egs_root)
    pass

if __name__ == '__main__':
    test()
