import svm_support
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

class Platt():

    def __init__(self):
        
        self.data = None
        self.labels = None
        self.oS = None
        self.weights = None

    def loadData(self,dataIn,classLabelsIn):
        self.data = np.array(dataIn, dtype=float)
        self.data[np.isnan(self.data)] = 0 

        self.labels = np.array(classLabelsIn)
        self.labels = np.transpose(self.labels)
    
    def innerLoop(self, i):
        ithError = svm_support.calcKthError(self.oS, i)
        if ((self.oS.labels[i] * ithError < -self.oS.tol) and (self.oS.alphas[i] < self.oS.C)) or\
            ((self.oS.labels[i] * ithError > self.oS.tol) and (self.oS.alphas[i] > 0)):
            j, jthError = svm_support.selectJ(i, self.oS, ithError)
            oldIthAlpha = self.oS.alphas[i].copy()
            oldJthAlpha = self.oS.alphas[j].copy()
 
            if (self.oS.labels[i] != self.oS.labels[j]):
                L = max(0, self.oS.alphas[j] - self.oS.alphas[i])
                H = min(self.oS.C, self.oS.C + self.oS.alphas[j] - self.oS.alphas[i])
            else:
                L = max(0, self.oS.alphas[j] + self.oS.alphas[i] - self.oS.C)
                H = min(self.oS.C, self.oS.alphas[j] + self.oS.alphas[i])
 
            if L == H:
                return 0

            eta = 2.0 * np.dot(self.oS.data[i,:], self.oS.data[j,:].T) - np.dot(self.oS.data[i,:], self.oS.data[i,:].T) - np.dot(self.oS.data[j,:], self.oS.data[j,:].T)
 
            if eta >= 0:
                return 0
 
            self.oS.alphas[j] -= self.oS.labels[j] * (ithError - jthError) / eta
 
            self.oS.alphas[j] = svm_support.clipAlpha(self.oS.alphas[j], H, L)
            svm_support.updateKthError(self.oS, j)
 
            if (abs(self.oS.alphas[j] -oldJthAlpha)) < 0.00001:
                return 0
 
            self.oS.alphas[i] += self.oS.labels[j] * self.oS.labels[i] * (oldJthAlpha - self.oS.alphas[j])
 
            svm_support.updateKthError(self.oS, i)

            b1 = self.oS.b - ithError - self.oS.labels[i] * (self.oS.alphas[i] - oldIthAlpha) *\
                 np.dot(self.oS.data[i,:], self.oS.data[i,:].T) - self.oS.labels[j] *\
                 (self.oS.alphas[j] - oldJthAlpha) * np.dot(self.oS.data[i,:], self.oS.data[j,:].T)
           
            b2 = self.oS.b - jthError - self.oS.labels[i] * (self.oS.alphas[i] - oldIthAlpha) *\
                 np.dot(self.oS.data[i,:], self.oS.data[j,:].T) - self.oS.labels[j] *\
                 (self.oS.alphas[j] - oldJthAlpha) * np.dot(self.oS.data[j,:], self.oS.data[j,:].T)
            
            if 0 < self.oS.alphas[i] and self.oS.C > self.oS.alphas[i]:
                self.oS.b = b1
            elif 0 < self.oS.alphas[j] and self.oS.C > self.oS.alphas[i]:
                self.oS.b = b2
            else:
                self.oS.b = (b1 + b2) / 2.0

            return 1
        else:
            return 0

    def train(self, C, tolerance, maxIterations, kTup=('lin', 0)):
        self.oS = svm_support.optStruct(self.data, self.labels, C, tolerance)

        iterations = 0

        entireSet = True

        alphaPairsChanged = 0

        while iterations < maxIterations and (alphaPairsChanged > 0 or entireSet):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(self.oS.m):
                    alphaPairsChanged += self.innerLoop(i)
                iterations += 1

            else:
                nonBoundIs = np.nonzero((self.oS.alphas > 0) * (self.oS.alphas < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerLoop(i)
                    iterations += 1
            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True

            print "Iteration: " + str(iterations)

        self.calcWeights()

    def calcWeights(self):
        m,n = np.shape(self.data)
        self.weights = np.zeros(n)
        for i in range(m):
            scalar = np.zeros((n,n))
            np.fill_diagonal(scalar, (self.oS.alphas[i] * self.oS.labels[i]))
            self.weights += np.dot(self.oS.data[i,:], scalar).T

    def classify(self, dataPoint):
        point = np.array(dataPoint, dtype=float)
        point[np.isnan(point)] = 0
        if np.dot(point, self.weights) + self.oS.b > 0:
            return 1.0
        else:
            return -1.0
