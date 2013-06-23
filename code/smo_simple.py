import numpy as np
import numpy.ma as ma 
import svm_helper as helper

class SMOSimple():

    def __init__(self):
        
        self.b = 0
        self.alphas = None
        self.weights = None
        self.data = None
        self.labels = None

    def loadData(self,dataIn,classLabelsIn):
        self.data = np.array(dataIn, dtype=float)
        self.data[np.isnan(self.data)] = 0 

        self.labels = np.array(classLabelsIn)
        self.labels = np.transpose(self.labels)

    def trainSMO(self, C, tolerance, maxIter):
        
        num_companies,num_signals = np.shape(self.data)

        # create column vector of 0s, one for each point
        self.alphas = np.zeros((num_companies,1))

        iterations = 0

        while iterations < maxIter:
            
            alphaPairsChanged = 0

            for companyi in range(num_companies):
                
                # get prediction and error
                intermediate = np.dot(self.data, np.transpose(self.data[companyi,:]))
                ithPredic = float(np.dot(self.alphas.T * self.labels, intermediate)) + self.b
                ithPredicError = ithPredic - float(self.labels[companyi])
                
                
                # can alpha be optimized? (if the error is large)
                if ((self.labels[companyi] * ithPredicError < -tolerance) and (self.alphas[companyi] < C)) or\
                   ((self.labels[companyi] * ithPredicError >  tolerance) and (self.alphas[companyi] > 0)):
                    
                    # get new company that's not current one
                    companyj = helper.selectRandomJ(companyi,num_companies)
                    jthPredic = float(np.dot(self.alphas.T * self.labels, np.dot(self.data, self.data[companyj,:].T))) + self.b
                    # jthPredic = float((np.multiply(self.alphas,self.labels).T) * (self.data * self.dataMat[companyj,:].T)) + b
                    jthPredicError = jthPredic - float(self.labels[companyj])
                    
                    # save old alphas
                    ithAlphaOld = self.alphas[companyi].copy()
                    jthAlphaOld = self.alphas[companyj].copy()
                    
                    # ???????????????????????????????????
                    if self.labels[companyi] != self.labels[companyj]:
                        L = max(0, self.alphas[companyj] - self.alphas[companyi])
                        H = min(C, C + self.alphas[companyj] - self.alphas[companyi])
                    else:
                        L = max(0, self.alphas[companyj] + self.alphas[companyi] - C)
                        H = min(C, self.alphas[companyj] + self.alphas[companyi])

                    if L == H: 
                        print "L == H, going to next company..."
                        continue
                    
                    dAlpha = 2.0 * np.dot(self.data[companyi,:], self.data[companyj,:].T)\
                             - np.dot(self.data[companyi,:], self.data[companyi,:].T)\
                             - np.dot(self.data[companyj,:], self.data[companyj,:].T)

                    if dAlpha == 0:
                        print "dAlpha == 0, going to next company..."
                        continue

                    self.alphas[companyj] -= self.labels[companyj] * (ithPredicError - jthPredicError) / dAlpha
                    self.alphas[companyj] = helper.clipAlpha(self.alphas[companyj],H,L)

                    if abs(self.alphas[companyj] - jthAlphaOld) < 0.00001:
                        print "jth alpha didn't move around enough, going to next company..."
                        continue

                    self.alphas[companyi] += (self.labels[companyj] * self.labels[companyi]) * (jthAlphaOld - self.alphas[companyj])
                    
                    b1 = self.b - ithPredicError - \
                         self.labels[companyi] * (self.alphas[companyi] - ithAlphaOld) * \
                         np.dot(self.data[companyi,:], self.data[companyi,:].T) - \
                         self.labels[companyj] * (self.alphas[companyj] - jthAlphaOld) * \
                         np.dot(self.data[companyi,:], self.data[companyj,:].T)

                    b2 = self.b - jthPredicError - \
                         self.labels[companyi] * (self.alphas[companyi] - ithAlphaOld) * \
                         np.dot(self.data[companyi,:], self.data[companyj,:].T) - \
                         self.labels[companyj] * (self.alphas[companyj] - jthAlphaOld) * \
                         np.dot(self.data[companyj,:], self.data[companyj,:].T)

                    if (0 < self.alphas[companyi]) and (C > self.alphas[companyi]):
                        self.b = b1
                    elif (0 < self.alphas[companyj]) and (C > self.alphas[companyj]):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    alphaPairsChanged += 1

                    print "Changed alpha pair."
                    
            if alphaPairsChanged  == 0:
                iterations += 1
            else:
                iterations = 0

            print "Iteration", iterations
        
        self.calcWeights()

    def calcWeights(self):
        m,n = np.shape(self.data)
        self.weights = np.zeros(n)
        for i in range(m):
            scalar = np.zeros((n,n))
            np.fill_diagonal(scalar, (self.alphas[i] * self.labels[i]))
            self.weights += np.dot(self.data[i,:], scalar).T

    def classify(self, dataPoint):
        point = np.array(dataPoint, dtype=float)
        point[np.isnan(point)] = 0
        if np.dot(point, self.weights) + self.b > 0:
            return 1.0
        else:
            return -1.0
