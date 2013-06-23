import svm_support
import numpy as np

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
#        if np.dot(point, self.weights) + self.oS.b > 0:
#            return 1.0
#        else:
#            return -1.0
        return np.dot(point, self.weights) + self.oS.b
