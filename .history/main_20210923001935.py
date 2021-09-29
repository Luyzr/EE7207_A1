import numpy as np
import math
import scipy.io
from sklearn.model_selection import KFold

# data_train.shape => (330, 33)
# label_train.shape => (330, 1)
# data_test.shape => (21, 33)

def Gaussian(x, c, t):
    return math.exp(-1 * np.sum(np.square(x - c))/(2 * t**2))

def EstimateC(data, label, pn=30, pretrained=False):
    print('Getting center vector...')
    if pretrained:
        return np.load('{}_center_vector.npy'.format(pn))
    n, d = data.shape
    e = np.zeros(n)
    candi = [i for i in range(0, 330)]
    for i in range(0, n):
        c = data[i]
        o, w = EstimateOW(data, c, 0.707, label)
        f = np.dot(o, w)
        e[i] = 1/2 * np.sum(np.square(f - label))
    first = np.argmin(e)
    err = e[first]
    old_err = np.Inf
    c = data[first].reshape((1,-1))
    candi.pop(first)
    m = 1
    # print('round:{}  error:{:.2f}\n'.format(m, err))
    while m < pn and err <= old_err and np.abs(err - old_err) > 0.15:
        m += 1
        old_err = err
        e = np.Inf * np.ones(n)
        for k in range(0, n - m):
            i = candi[k]
            nc = np.concatenate((c, data[i].reshape(1,-1)), axis=0)
            t = EstimateT(nc, m)
            o, w = EstimateOW(data, nc, t, label)
            f = np.dot(o, w)
            e[i] = 1/2 * np.sum(np.square(f - label))
        first = np.argmin(e)
        err = e[first]
        c = np.concatenate((c, data[first].reshape(1,-1)), axis=0)
        candi.pop(candi.index(first))
        # print('round:{}  error:{:.2f}\n'.format(m, err))
    print('Number of center vector:{}, saving'.format(m))
    np.save('{}_center_vector.npy'.format(m), c)
    return c

def EstimateT(c, m):
    # Estimate the parameter of Gaussian
    dis = [0]*m
    for i in range(m):
        for j in range(i, m):
            dis[j] = max(dis[j], np.sqrt(np.sum(np.square(c[i] - c[j]))))
    t = max(dis)/np.sqrt(2*m)
    return t

def getO(data, c, t):
    m = c.shape[0]
    n, d = data.shape
    o = [[0]*m for i in range(n)]
    for i in range(n):
        for j in range(m):
            o[i][j] = Gaussian(data[i], c[j], t)
    o = np.array(o).reshape(n, m)
    return o

def EstimateOW(data, c, t, label):
    # Estimate W
    n, d = data.shape
    m = c.shape[0]
    o = getO(data, c, t)
    w = np.dot(np.dot(np.linalg.pinv((np.dot(o.T,o))),o.T),np.array(label))
    return o, w

def LinearRBF(data, label, pn, pretrained=False):
    c = EstimateC(data, label, pn=pn, pretrained=pretrained)
    m, _ = c.shape
    t = EstimateT(c, m)
    o, w = EstimateOW(data, c, t, label)
    return c, w, t

def Dataloader():
    train_data = scipy.io.loadmat('data_train.mat')['data_train']
    train_label = scipy.io.loadmat('label_train.mat')['label_train']
    test_data = scipy.io.loadmat('data_test.mat')['data_test']
    return train_data, train_label, test_data

def Train(data_train, label_train, pn=4, pretrained=False):

    c, w, t = LinearRBF(data_train, label_train, pn=pn , pretrained=pretrained)

    m, d = c.shape

    o = getO(data_train, c, t)
    f = np.dot(o, w)
    label_train = np.heaviside(label_train, 0.5)
    f = np.heaviside(f, 0.5)
    err = 0
    n, _ = label_train.shape
    for i in range(0, n):
        if label_train[i] != f[i]:
            err += 1
    print('Train accuracy is {:.2f}%'.format(100 * (1 - err/n)))
    return c, w, t

def Evaluate(data_test, label_test, c, w, t, mode='t'):
    o = getO(data_test, c, t)
    f = np.dot(o, w)
    f = np.heaviside(f, 0.5)
    err = 0
    if mode == 't':
        label_test = np.heaviside(label_test, 0.5)
        print('Truth is {}'.format(label_test.reshape(1, -1)))
        print('Result is {}'.format(f.reshape(1,-1)))
        n, _ = label_test.shape
        for i in range(0, n):
            if label_test[i] != f[i]:
                err += 1
        print('Test accuracy is {:.2f}%'.format(100 * (1 - err/n)))
        return 1 - err/n
    if mode == 'e':
        print('Result is {}'.format(f.reshape(1,-1)))
        return

if __name__ == "__main__":
    '''
    参数都是一脉相承的
    pn 表示设定的CV的数量
    m  是计算过程中实际用到的CV的数量
    c  是CV
    w  是权重
    t  是高斯参数
    '''
    print('Loading data')
    train_data, train_label, test_data = Dataloader()
    
    # 调参用这个
    getCV = False
    pn = 2

    # # 得结果用这个(你得先调过参才行)
    # getCV = True
    # pn = 15

    kf = KFold(5, shuffle=True, random_state=42)
    rr = 1
    if getCV:
        best_pn = 0
        best_score = 0
        for pn in range(2, 20):
            scores = []
            rr = 1
            for train_index, test_index in kf.split(train_data):
                print('========================== The {}th experiment with pn={} =========================='.format(rr, pn))
                rr += 1
                data_train, label_train = train_data[train_index], train_label[train_index]
                data_test, label_test = train_data[test_index], train_label[test_index]

                print('Start Training...')
                c, w, t = Train(data_train, label_train, pn, pretrained=False)

                print('Start Evaluating..')
                score = Evaluate(data_test, label_test, c, w, t, mode='t')
                scores.append(score)
            mean_score = np.mean(np.array(scores))
            print('The mean score with pn={} is {}\n'.format(pn, mean_score))
            if mean_score > best_score:
                best_pn = pn
                best_score = mean_score
        print('The best pn is {}, with the best score: {}'.format(best_pn, best_score))
    else:
        c, w, t = Train(train_data, train_label, pn, pretrained=True)
        print('pn is: {}; t is: {:.4f}'.format(pn, t))
        Evaluate(test_data, None, c, w, t, mode='e')