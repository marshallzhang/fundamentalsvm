import numpy as np
import random

class optStruct:
    def __init__(self, dataIn, classLabels, C, tolerance):
        self.data = dataIn
        self.labels = classLabels
        self.C = C
        self.tol = tolerance
        self.m = np.shape(dataIn)[0]
        self.alphas = np.zeros((self.m))
        self.b = 0
        self.errorCache = np.zeros((self.m,2))

def calcKthError(oS, k):
    kthPredic = float(np.dot((oS.alphas * oS.labels).T, np.dot(oS.data, oS.data[k,:].T))) + oS.b
    kthError = kthPredic - float(oS.labels[k])
    return kthError

def selectJ(i, oS, ithError):
    maxK = -1
    maxDeltaError = 0
    jthError = 0

    oS.errorCache[i] = [1, ithError]

    validErrorCacheList = np.nonzero(oS.errorCache[:,0])[0]

    if len(validErrorCacheList) > 1:
        for k in validErrorCacheList:
            if k == i:
                continue
            kthError = calcKthError(oS, k)
            deltaError = abs(ithError - kthError)
            if (deltaError > maxDeltaError):
                maxK = k
                maxDeltaError = deltaError
                jthError = kthError
        return maxK, jthError

    else:
        j = selectRandomJ(i, oS.m)
        jthError = calcKthError(oS, j)
    return j, jthError

def updateKthError(oS, k):
    kthError = calcKthError(oS, k)
    oS.errorCache[k] = [1, kthError]

def selectRandomJ(i,m):
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj
