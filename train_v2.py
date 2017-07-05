import sys
import numpy as np
import logging
from collections import namedtuple
import wf2array
import itertools
import operator
import scipy.cluster.hierarchy as hcluster

from scipy.spatial.distance import pdist
import scipy.sparse as sp
from numpy import array, ceil, sqrt, empty
from sklearn.model_selection import KFold
from sklearn import metrics

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC

import argparse

args = None

dt2 = np.dtype([('apxy', 'S20'), ('pid', 'S30'), ('x', 'f4'), ('y', 'f4'), ('wflist', 'S600')])
dt = np.dtype([('pid', 'S30'), ('apxy', 'S30'), ('x', 'f4'), ('y', 'f4'), ('wflist', 'S600'), ('tag', 'S10')])

dt_pu2 = np.dtype([('grid', 'S20'), ('pid', 'S30'), ('ap', 'S20'), ('sig', 'i4'), ('ts', 'S20'), ('tag', 'S20'), ('x', 'f4'), ('y', 'f4'), ('wflist', 'S600')])

dt_pu = np.dtype([('apxy', 'S20'), ('sig', 'i4'), ('ts', 'S20'), ('pid', 'S30'), ('tag', 'S20'), ('x', 'f4'), ('y', 'f4'), ('wflist', 'S600')])

dt_pu = np.dtype([('pid', 'S30'), ('xy', 'S20'), ('x', 'f4'), ('y', 'f4'), ('wflist', 'S600'), ('tag', 'S20')])
dt_pu = np.dtype([('apxy', 'S30'), ('pid', 'S30'), ('tag', 'S10'), ('x', 'f4'), ('y', 'f4'), ('wflist', 'S600')] )

class PUClassifier(object):
    def __init__(self, trad_clf=None, n_folds=5):
        self.trad_clf = trad_clf
        self.n_folds = n_folds

    def fit(self, X, s):
        if self.trad_clf is None:
            #self.trad_clf = GridSearchCV(SGDClassifier(loss="log", penalty="l2"), param_grid={"alpha": np.logspace(-4, 0, 10)})
            #self.trad_clf = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid={'C': np.logspace(0, 1, 2)})
            self.trad_clf = SVC(C=1, kernel='rbf', probability=True)

        c = np.zeros(self.n_folds)
        s0 = enumerate(StratifiedKFold(s[s==0], n_folds=self.n_folds,shuffle=True))
        s1 = enumerate(StratifiedKFold(s[s==1], n_folds=self.n_folds,shuffle=True))

        for i, (itr, ite) in s0:
            for j, (jtr, jte) in s1:
                if i == j:
                    try:
                        X_train = np.r_[X[itr], X[jtr]]
                        y_train = np.r_[s[itr], s[jtr]]
                        y_train.shape = (y_train.shape[0], 1)

                        self.trad_clf.fit(X_train, y_train)
                        c[i] = self.trad_clf.predict_proba(X[ite][s[ite]==1])[:,1].mean()
                        c[i] += self.trad_clf.predict_proba(X[jte][s[jte]==0])[:,1].mean()
                    except Exception, e:
                        logging.warning(e)

        self.c = c.mean()
        return self

    def sample(self, X, s):
        (fill, ij, ap2idx, idx2ap, size) = wf2array.get_fill_ij(X,30,convert_sig,True)
        (fill, ij, ap2idx, idx2ap) = feature_selection((fill, ij), idx2ap, ap2idx)
        X = sp.csr_matrix((fill,ij),dtype=np.float32,shape=(size,ij[1,:].max() + 1)).todense()

        if not hasattr(self, 'c'):
            self.fit(X, s)
        X_positive = X[s==1]
        X_unlabeled = X[s==0]
        X_negative_test = X[s==20]
        n_positive = X_positive.shape[0]
        n_unlabeled = X_unlabeled.shape[0]

        X_positive_test = X[s==11]
        X_unlabeled_test = X[s==10]
        n_positive_test = X_positive_test.shape[0]
        n_unlabeled_test = X_unlabeled_test.shape[0]

        X_train = np.r_[X_positive, X_unlabeled, X_unlabeled]
        y_train = np.concatenate([np.repeat(1, n_positive), np.repeat(1, n_unlabeled), np.repeat(0, n_unlabeled)])
        X_test = np.r_[X_positive_test, X_unlabeled_test]
        y_test = np.concatenate([np.repeat(1, n_positive_test), np.repeat(0,n_unlabeled_test)])

        self.trad_clf.fit(X, s)
        p_unlabeled = self.trad_clf.predict_proba(X_unlabeled)[:,1]
        w_positive = ((1 - self.c) / self.c) * (p_unlabeled / (1 - p_unlabeled))
        w_negative = 1 - w_positive
        sample_weight = np.concatenate([np.repeat(1.0, n_positive), w_positive, w_negative])
        return X_train, y_train, sample_weight, X_test, y_test, X_negative_test


def get_data(lines, sep='\t'):
    for line in lines:
        line = line.strip('\n').split(sep)
        yield line

def build_and_fit(name, X, Y):
    if name == 'LR':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1.0, multi_class='ovr', penalty='l2')
        model.fit(X, np.array(Y))
        return model
    elif name == 'LSVM':
        from sklearn.svm import LinearSVC
        model = LinearSVC(C=1.0, multi_class='ovr', penalty='l2', dual=False)
        model.fit(X, np.array(Y))
        return model
    elif name == 'OSVC':
        from sklearn.svm import OneClassSVM
        #model = OneClassSVM(kernel='sigmoid', nu=0.1, gamma=0.1)
        #model = OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)
        model = OneClassSVM(kernel='rbf', nu=args.nu, gamma=args.gamma)
        model.fit(X)
        return model
    return None


def process_wifi(wflist, topk):
    wflist = wflist.strip().split('|')
    return '|'.join(wflist[:topk])


def eva_oc(maxlen):
    topk = 100 if args.topk == None else args.topk

    group_data = np.zeros((maxlen,),dtype=object)
    size = 0
    oldlabel = ''
    newlabel = ''
    for pid, g in itertools.groupby(get_data(sys.stdin), operator.itemgetter(0)):
        for data in g:
            data[4] = process_wifi(data[4], topk)
            data = np.array(tuple(data), dtype=dt)

            if size == 0:
                oldlabel = data['pid']
                group_data[size] = data
                size += 1
            else:
                newlabel = data['pid']
                if newlabel != oldlabel:
                    oneclass(group_data[:size,].astype(dt, copy=False))

                    size = 0
                    group_data[size] = data
                    oldlabel = data['pid']
                else:
                    group_data[size] = data
                    size += 1
        oneclass(group_data[:size,].astype(dt, copy=False))


def pu_train(positive_data, negative_data, unlabeled_data):
    num_p = positive_data.shape[0]
    num_n = negative_data.shape[0]
    num_u = unlabeled_data.shape[0]

    #print num_p, num_n, num_u
    if num_p < 10 or num_n < 10 or num_u < 10:
        return
    num_p_train = int(num_p * 0.8)
    num_u_train = int(num_u * 0.8)
    num_p_test = num_p - num_p_train
    num_u_test = num_u - num_u_train
    num_n_test = num_n

    p_X_train = positive_data[:num_p_train,].astype(dt_pu, copy=False)['wflist']
    p_X_test = positive_data[num_p_train:,].astype(dt_pu, copy=False)['wflist']
    u_X_train = unlabeled_data[:num_u_train,].astype(dt_pu, copy=False)['wflist']
    u_X_test = unlabeled_data[num_u_train:,].astype(dt_pu, copy=False)['wflist']
    #negative_data.astype(dt_pu, copy=False)
    n_X_test = negative_data.astype(dt_pu, copy=False)['wflist']

    X = np.r_[p_X_train, u_X_train, p_X_test, u_X_test, n_X_test]
    s = np.concatenate([np.repeat(1, num_p_train), np.repeat(0, num_u_train),\
        np.repeat(11, num_p_test), np.repeat(10, num_u_test),\
        np.repeat(20, num_n_test)])

    pu = PUClassifier(n_folds=3)
    X_train, y_train, sample_weight, X_test, y_test, X_negative_test = pu.sample(X, s)

    n_folds = 3
    #alphas = np.logspace(0, 1, 2)
    alphas = [0.1, 0.5, 1]
    class_weights = [{1:1}]
    best_score = -np.inf
    best_alpha = None
    best_class_weight = None
    for alpha, class_weight in itertools.product(alphas, class_weights):
        scores = np.zeros(n_folds)
        for i, (itr, ite) in enumerate(StratifiedKFold(y_train, n_folds=n_folds, shuffle=True)):
            #clf = SGDClassifier(loss="log", penalty="l2", alpha=alpha, class_weight=class_weight)
            clf = SVC(C=alpha, class_weight=class_weight, kernel='rbf')
            clf.fit(X_train[itr], y_train[itr], sample_weight=sample_weight[itr])
            ypred = clf.predict(X_train[ite])
            scores[i] = metrics.accuracy_score(y_train[ite], ypred, sample_weight=sample_weight[ite])

        this_score = scores.mean()
        if this_score > best_score:
            best_score = this_score
            best_alpha = alpha
            best_class_weight = class_weight
    #clf = SGDClassifier(loss="log", penalty="l2", alpha=best_alpha, class_weight=best_class_weight).fit(X_train, y_train, sample_weight=sample_weight)
    clf = SVC(C=best_alpha,class_weight=class_weight,kernel='rbf').fit(X_train, y_train, sample_weight=sample_weight)
    #for x in clf.support_vectors_:
    #    print x

    p_Z_train = clf.predict(X_train[:num_p_train])
    p_Z_test = clf.predict(X_test[:num_p_test])
    n_Z_test = clf.predict(X_negative_test)
    #p_Z_train = pu.trad_clf.predict(X_train[:num_p_train])
    #p_Z_test = pu.trad_clf.predict(X_test[:num_p_test])
    #n_Z_test = pu.trad_clf.predict(X_negative_test)
    D_train = clf.decision_function(X_train[:num_p_train])

    p_train_accuracy = metrics.accuracy_score([1]*num_p_train, p_Z_train)
    p_test_accuracy = metrics.accuracy_score([1]*num_p_test, p_Z_test)
    n_test_accuracy = metrics.accuracy_score([0]*num_n_test, n_Z_test)

    p_train_num = len(p_Z_train)
    p_train_err_num = p_train_num - p_train_num * p_train_accuracy

    p_test_num = len(p_Z_test)
    n_test_num = len(n_Z_test)
    p_test_err_num = p_test_num - p_test_num * p_test_accuracy
    n_test_err_num = n_test_num - n_test_num * n_test_accuracy

    print '%d\t%d\t%d\t%d\t%d\t%d' % (p_train_err_num, p_train_num, p_test_err_num, p_test_num, n_test_err_num, n_test_num)


def eva_puc(maxlen):
    topk = args.topk
    positive_data = np.zeros((maxlen,),dtype=object)
    negative_data = np.zeros((maxlen,),dtype=object)
    unlabeled_data = np.zeros((maxlen,),dtype=object)

    p_size = 0
    n_size = 0
    u_size = 0
    size = 0

    def add_data(tag, data, positive_data, negative_data, unlabeled_data, p_size, n_size, u_size):
        if tag == '0':
            unlabeled_data[u_size] = data
            u_size += 1
        elif tag == '1':
            positive_data[p_size] = data
            p_size += 1
        elif tag == '-1':
            negative_data[n_size] = data
            n_size += 1
        return positive_data, negative_data, unlabeled_data, p_size, n_size, u_size


    old_apxy = None
    new_apxy = None
    old_pid = None
    new_pid = None
    for data in sys.stdin:
        data = data.strip('\n').split('\t')
        data[4] = process_wifi(data[4], topk)

        data = np.array(tuple(data), dtype=dt_pu)
        pid = data['pid']
        tag = data['tag']
        apxy = data['apxy']

        new_apxy = apxy
        if old_apxy == None:
            old_apxy = apxy
            #add_data(tag, data, positive_data, negative_data, unlabeled_data, p_size, n_size, u_size)
            positive_data, negative_data, unlabeled_data, p_size, n_size, u_size = add_data(tag, data, positive_data, negative_data, unlabeled_data, p_size, n_size, u_size)
            continue
        elif new_apxy != old_apxy:
            pu_train(positive_data[:p_size,],\
                    negative_data[:n_size,],\
                    unlabeled_data[:u_size,])
            positive_data = np.zeros((maxlen,),dtype=object)
            negative_data = np.zeros((maxlen,),dtype=object)
            unlabeled_data = np.zeros((maxlen,),dtype=object)
            p_size = 0
            n_size = 0
            u_size = 0

            old_apxy = new_apxy
            #add_data(tag, data, positive_data, negative_data, unlabeled_data, p_size, n_size, u_size)
            positive_data, negative_data, unlabeled_data, p_size, n_size, u_size = add_data(tag, data, positive_data, negative_data, unlabeled_data, p_size, n_size, u_size)
            continue

        new_pid = pid
        if old_pid == None or pid == '' or old_pid == '' or old_pid == new_pid:
            old_pid = pid
            positive_data, negative_data, unlabeled_data, p_size, n_size, u_size = add_data(tag, data, positive_data, negative_data, unlabeled_data, p_size, n_size, u_size)
            continue
        elif new_pid != old_pid and old_pid != '':
            pu_train(positive_data[:p_size,],\
                    negative_data[:n_size,],\
                    unlabeled_data[:u_size,])
            positive_data = np.zeros((maxlen,),dtype=object)
            negative_data = np.zeros((maxlen,),dtype=object)
            p_size = 0
            n_size = 0
            #add_data(tag, data, positive_data, negative_data, unlabeled_data, p_size, n_size, u_size)
            positive_data, negative_data, unlabeled_data, p_size, n_size, u_size = add_data(tag, data, positive_data, negative_data, unlabeled_data, p_size, n_size, u_size)
            continue
            
    #try:
    pu_train(positive_data[:p_size,],\
            negative_data[:n_size,],\
            unlabeled_data[:u_size,])
    #except:
        #pass

def feature_selection(m, idx2ap, ap2idx):
    from collections import Counter
    n_feature = args.nfea

    (fill, ij) = m
    ap2count = Counter(ij[1,:])
    ms = ap2count.most_common()
    occuNum = [ c for (apidx,c) in ms ]

    thr = ms[:n_feature][-1][1]

    smallap = set([ apidx for (apidx,c) in ms if c < thr ])
    new_ap2idx = dict()
    new_idx2ap = dict()
    for ap in ap2idx:
        if ap2idx[ap] not in smallap:
            idx = len(new_ap2idx)
            new_ap2idx[ap] = idx 
            new_idx2ap[idx] = ap
    lineno2newlinenp = dict()
    fill_idx = 0 
    for i in range(fill.shape[0]):
        if ij[1,i] in smallap:
            continue
        fill[fill_idx] = fill[i]
        if ij[0,i] not in lineno2newlinenp:
            lineno2newlinenp[ij[0,i]] = len(lineno2newlinenp)
        ij[0,fill_idx] = lineno2newlinenp[ij[0,i]]
        ij[1,fill_idx] = new_ap2idx[idx2ap[ij[1,i]]]
        fill_idx += 1
    fill = fill[:fill_idx]
    ij = ij[:,:fill_idx]
    
    return (fill,ij,new_ap2idx,new_idx2ap)


def convert_sig(x):
    x = int(x)
    return x
    if x > 90:
        return 0
    else:
        return 100-x

def oneclass(data):
    L = data.shape[0]
    if L < 10:
        return
    (fill, ij, ap2idx, idx2ap, size) = wf2array.get_fill_ij(data['wflist'],30,convert_sig,True)
    (fill, ij, ap2idx, idx2ap) = feature_selection((fill, ij), idx2ap, ap2idx)

    m = sp.csr_matrix((fill,ij),dtype=np.float32,shape=(size,ij[1,:].max() + 1))

    L = m.shape[0]
    L0 = m[data['tag']=='0'].shape[0] # 0 means out
    L1 = m[data['tag']=='1'].shape[0] # 1 means in
    if L1 < 30:
        return
    if L0 < 5:
        return

    idx = range(L1)
    idx_bound = int(L1 * 0.8)
    train_err_num = 0
    train_num = 0
    test_err_num = 0
    test_num = 0
    testacc_err_num = 0
    testacc_num = 0

    train_idx = idx[:idx_bound]
    test_idx = idx[idx_bound:]
    X_train, X_test, X_test_acc = m[data['tag']=='1'][train_idx], m[data['tag']=='1'][test_idx], m[data['tag']=='0']

    Y_train, Y_test, Y_test_acc = data[data['tag']=='1']['pid'][train_idx], data[data['tag']=='1']['pid'][test_idx], data[data['tag']=='0']

    model = build_and_fit(args.model, X_train, Y_train)
    if model == None:
        logging.warning('Failed to build model')
        return
    Z_train = model.predict(X_train)
    Z_test = model.predict(X_test)
    Z_test_acc = model.predict(X_test_acc)

    D_train = model.decision_function(X_train)
    D_test = model.decision_function(X_test)
    D_test_acc = model.decision_function(X_test_acc)

    for i in range(len(Y_train)):
        if 1 != Z_train[i]:
            train_err_num += 1
            #print 'wr', data['wflist'][i], D_train[i]
        #else:
            #print 'ac', data['wflist'][i], D_train[i]
        train_num += 1
    for i in range(len(Y_test)):
        if 1 != Z_test[i]:
            test_err_num += 1
            #print 'wr2', data['wflist'][i+len(Y_train)], D_test[i]
        #else:
            #print 'ac2', data['wflist'][i+len(Y_train)], D_test[i]
        test_num += 1
    for i in range(len(Y_test_acc)):
        if -1 != Z_test_acc[i]:
            testacc_err_num += 1
            #print 'wr', data['wflist'][data['tag']=='0'][i], D_test_acc[i]
        #else:
            #print 'ac', data['wflist'][data['tag']=='0'][i], D_test_acc[i]
        testacc_num += 1
    print train_err_num, train_num, test_err_num, test_num, testacc_err_num, testacc_num


def eva_method(groupid, k, data):
    data = data.astype(dt)

    kf = KFold(n_splits=2, shuffle=True)
    train_err_num = 0
    test_err_num = 0
    train_num = 0
    test_num = 0

    #(m,idx2ap,ap2idx) = dm2.normwf_and_getspswf(data['wflist'], 100, normed=True)
    (m,idx2ap,ap2idx) = dm2.normwf_and_getspswf(data['wflist'], 100, normed=False)
    for train_idx, test_idx in kf.split(data):
            X_train, X_test = m[train_idx], m[test_idx]
            #W_train, W_test = data['weight'][train_idx], data['weight'][test_idx]
            Y_train, Y_test = data['pid'][train_idx], data['pid'][test_idx]

            coorx_train, coorx_test = data['x'][train_idx], data['x'][test_idx]
            coory_train, coory_test = data['y'][train_idx], data['y'][test_idx]

            model = build_and_fit(args.model, X_train, Y_train)
            if model == None:
                logging.warning("Failed to build model")
                continue
            Z_train = model.predict(X_train)
            Z_test = model.predict(X_test)
            D_train = model.decision_function(X_train)
            D_test = model.decision_function(X_test)

            ptr = ''

            # test error
            for i in range(len(Y_train)):
                if Y_train[i] != Z_train[i]:
                    train_err_num += 1
                train_num += 1
            # train err
            for i in range(len(Y_test)):
                if Y_test[i] != Z_test[i]:
                    test_err_num += 1
                test_num += 1

            # tmp
            savemodel = False
            if savemodel:
                coefs = model.coef_.flatten()
                idxs = []
                ncoefs = []
                for i in range(coefs.shape[0]):
                    #if coefs[i] != 0:
                    idxs.append(i)
                    ncoefs.append(coefs[i])

                feas = ','.join(['%d' % idx2ap[i]  for i in range(model.coef_.shape[1]) ])
                print '%s\trefine\tMULTI\t%s\t%s\t%s\t%s\t%s' % (groupid,\
                    ','.join(model.classes_),\
                    ','.join(map(str,model.intercept_.flatten())),\
                    feas,\
                    ','.join(map(str,idxs)),\
                    ','.join(map(str,ncoefs)))
        #except Exception, e:
            #print Exception, " ", e

    #print '%s\t%s\t%s\t%s' % (train_err_num, train_num, test_err_num, test_num)


def kde(data):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    L = data.shape[0]
    if L < 10:
        return

    (fill, ij, ap2idx, idx2ap, size) = wf2array.get_fill_ij(data['wflist'],30,convert_sig,False)
    (fill, ij, ap2idx, idx2ap) = feature_selection((fill, ij), idx2ap, ap2idx)
    m = sp.csr_matrix((fill,ij),dtype=np.float32,shape=(size,ij[1,:].max() + 1)).todense()
    print data['wflist']
    print m

    #pca = PCA(n_components=15, whiten=True)
    #pcad_m = pca.fit_transform(m)

    params = {'bandwidth': np.logspace(-1, 1, 20)}
    params = {'bandwidth': [0.1]}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(m)
    print 'bandwidth', grid.best_estimator_.bandwidth

    kde = grid.best_estimator_

    new_wifis = kde.sample(2, random_state=0)
    #new_wifis = pca.inverse_transform(new_wifis)
    for sigs in new_wifis:
        ptr = []
        for i in idx2ap:
            ap = '%012x' % (idx2ap[i])
            ptr.append(ap+';'+str(sigs[i]))
        print '|'.join(ptr)



def gan(maxlen):
    topk = 100 if args.topk == None else args.topk

    group_data = np.zeros((maxlen,),dtype=object)
    size = 0
    oldlabel = ''
    newlabel = ''
    for pid, g in itertools.groupby(get_data(sys.stdin), operator.itemgetter(0)):
        for data in g:
            data = np.array(tuple(data), dtype=dt)

            if size == 0:
                oldlabel = data['pid']
                group_data[size] = data
                size += 1
            else:
                newlabel = data['pid']
                if newlabel != oldlabel:
                    kde(group_data[:size].astype(dt, copy=False))
                    size = 0
                    group_data[size] = data
                    oldlabel = data['pid']
                else:
                    group_data[size] = data
                    size += 1
        kde(group_data[:size].astype(dt, copy=False))


if __name__ == '__main__':
    global args
    argsparser =  argparse.ArgumentParser()
    argsparser.add_argument("-nu",type=float)
    argsparser.add_argument("-gamma",type=float)
    argsparser.add_argument("-model",type=str)
    argsparser.add_argument("-topk",type=int,default=100)
    argsparser.add_argument("-mode",type=str,default='oc')
    argsparser.add_argument("-nfea",type=int,default=10)
    args = argsparser.parse_args()

    if args.mode == 'oc':
        eva_oc(10000)
    elif args.mode == 'test':
        gan(10000)
    else:
        eva_puc(10000)
