import numpy as np
import cv2
import math
import random

def other_maps(g,n_aucs_maps,shape_r,shape_c):
    """Sample reference maps for s-AUC"""
    while True:
        this_map = np.zeros((shape_r,shape_c))
        fix_nrs = random.sample(
            g.file_, n_aucs_maps)
        #print(fix_nrs)
        for map_idx, fix_nr in enumerate(fix_nrs):
            img_nr = fix_nr.split('.')[0]
            fix_file = f'{g.fix_p}/{g.phase}/{img_nr}.png'
            this_this_map = cv2.imread(str(fix_file), cv2.IMREAD_GRAYSCALE)/255.
            #this_this_map = resize_fixation(this_this_map,shape_r,shape_c).reshape(shape_r,shape_c)
            this_map += this_this_map
        this_map = np.clip(this_map, 0, 1)
        yield this_map

def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)))
    return norm_s_map


def auc_judd(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    s_map = normalize_map(s_map)
    assert np.max(gt) == 1.0,\
        'Ground truth not discretized properly max value > 1.0'
    assert np.max(s_map) == 1.0,\
        'Salience map not normalized properly max value > 1.0'

    # thresholds are calculated from the salience map,
    # only at places where fixations are present
    thresholds = s_map[gt > 0].tolist()

    num_fixations = len(thresholds)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map,
        # keep only those pixels with values above threshold
        temp = s_map >= thresh
        num_overlap = np.sum(np.logical_and(temp, gt))
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap
        # with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / (np.prod(gt.shape[:2]) - num_fixations)

        area.append((round(tp, 4) ,round(fp, 4)))

    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list, fp_list = list(zip(*area))
    return np.trapz(np.array(tp_list), np.array(fp_list))


def auc_shuff_acl(s_map, gt, other_map, n_splits=100, stepsize=0.1):

    # If there are no fixations to predict, return NaN
    if np.sum(gt) == 0:
        print('no gt')
        return None

    # normalize saliency map
    s_map = normalize_map(s_map)

    S = s_map.flatten()
    F = gt.flatten()
    Oth = other_map.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)

    # for each fixation, sample Nsplits values from the sal map at locations
    # specified by other_map

    ind = np.where(Oth > 0)[0]  # find fixation locations on other images

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.full((Nfixations_oth, n_splits), np.nan)

    for i in range(n_splits):
        # randomize choice of fixation locations
        randind = np.random.permutation(ind.copy())
        # sal map values at random fixation locations of other random images
        randfix[:, i] = S[randind[:Nfixations_oth]]

    # calculate AUC per random split (set of random locations)
    auc = np.full(n_splits, np.nan)
    for s in range(n_splits):

        curfix = randfix[:, s]

        allthreshes = np.flip(np.arange(0, max(np.max(Sth), np.max(curfix)), stepsize))
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1
        fp[-1] = 1

        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth

        auc[s] = np.trapz(np.array(tp), np.array(fp))

    return np.mean(auc)


def similarity(s_map, gt):
    # here gt is not discretized nor normalized
    s_map = normalize_map(s_map)
    gt = normalize_map(gt)
    s_map = s_map/(np.sum(s_map)*1.0)
    gt = gt/(np.sum(gt)*1.0)

    return np.sum(np.minimum(s_map, gt))
def nss(s_map,gt):
	s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)

	x,y = np.where(gt==1)
	temp = []
	for i in zip(x,y):
		temp.append(s_map_norm[i[0],i[1]])
	return np.mean(temp)

def cc(s_map,gt):
    s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)
    gt_norm = (gt - np.mean(gt))/np.std(gt)
    a = s_map_norm
    b= gt_norm
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

def kldiv(s_map,gt):
    s_map = s_map/(np.sum(s_map)*1.0)
    gt = gt/(np.sum(gt)*1.0)
    eps = 1e-9
    return np.sum(gt * np.log(eps + gt/(s_map + eps)))

if __name__ == "__main__":
    phase = 'val'
    shape_r,shape_c = 224,320
    batch_size = 16
    '''
    g = generator_SAL(phase,shape_r,shape_c,batch_size)
    o_g = other_maps(g,10,shape_r,shape_c)
    a = next(o_g)
    '''


    Metric_ = {'AUC_J':0.0,'SIM':0.0,'s-AUC':0.0,'CC':0.0,'NSS':0.0,'KL':0.0}
    map_p = np.random.randint(255,size=(8,224,320))/255.
    fix_g = map_p > 0.9
    map_g = map_p
    N = 8
    epoch = 0
    for k in range(N):
        print(k,end='\b\r')
        # Metric_['AUC_J'] += auc_judd(map_p[k],fix_g[k])
        # Metric_['s-AUC'] += auc_shuff_acl(map_p[k],fix_g[k],fix_g[k])
        # Metric_['NSS'] += nss(map_p[k],fix_g[k])
        # Metric_['SIM'] += similarity(map_p[k],map_g[k])
        Metric_['CC'] += cc(map_p[k],map_g[k])
        Metric_['KL'] += kldiv(map_p[k],map_g[k])
        #print(nss(map_p[k],fix_g[k]),(s_map - np.mean(s_map))/np.std(s_map))
    for key in Metric_:
        print(key)
        Metric_[key] /= N
    print(f"eval : {epoch} , AUC_J: {Metric_['AUC_J']:4.4f} , s-AUC: {Metric_['s-AUC']:4.4f} , NSS: {Metric_['NSS']:4.4f}")
    print(f"eval : {epoch} , SIM  : {Metric_['SIM']:4.4f} , CC   : {Metric_['CC']:4.4f} , KL : {Metric_['KL']:4.4f}")