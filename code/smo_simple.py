import numpy as np
import svm_helper as helper

def smoSimple(dataMatIn, classLabels, C, tolerance, maxIter):
    
    # import data and get shape of data
    dataMatrix = np.mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0 
    num_companies,num_signals = np.shape(dataMatrix)

    # create column vector of 0s, one for each point
    alphas = mat(zeros(num_companies,1))

    iterations = 0

    while iterations < maxIter:
        
        alphaPairsChanged = 0

        for companyi in range(num_companies):
            
            # get prediction and error
            ithPredic = float((multiply(alphas,labelMat).T) * (dataMatrix * dataMatrix[companyi,:].T)) + b
            ithPredicError = ithPredic - float(labelMat[companyi])
            
            # can alpha be optimized? (if the error is large)
            if ((labelMat[companyi] * ithPredicError < -toler) and (alphas[companyi] < C)) or\
               ((labelMat[companyi] * ithPredicError >  toler) and (alphas[companyi] > 0)):

                # get new company that's not current one
                companyj = helper.selectRandomJ(companyi,m)
                jthPredic = float((multiply(alphas,labelMat).T) * (dataMatrix * dataMatrix[companyj,:].T)) + b
                jthPredicError = jthPredic - float(labelMat[companyj])
                
                # save old alphas
                ithAlphaOld = alphas[companyi].copy()
                jthAlphaOld = alphas[companyj].copy()
                
                # ???????????????????????????????????
                if labelMat[companyi] != labelMat[companyj]:
                    L = max(0, alphas[companyj] - alphas[companyi])
                    H = min(C, C + alphas[companyj] - alphas[companyi])
                else:
                    L = max(0, alphas[companyj] + alphas[companyi] - C)
                    H = min(C, alphas[companyj] + alphas[companyi])

                if L == H: 
                    print "L == H, going to next company..."
                    continue
                
                dAlpha = 2.0 * dataMatrix[companyi,:] * dataMatrix[companyj,:].T\
                         - dataMatrix[companyi,:] * dataMatrix[companyi,:].T\
                         - dataMatrix[companyj,:] * dataMatrix[companyj,:].T

                if dAlpha == 0:
                    print "dAlpha == 0, going to next company..."
                    continue

                alphas[companyj] -= labelMat[companyj] * (ithPredicError - jthPredicError) / dAlpha
                alphas[companyj] = helper.clipAlpha(alphas[companyj],H,L)

                if abs(alphas[companyj] - jthAlphaOld) < 0.00001:
                    print "jth alpha didn't move around enough, going to next company..."
                    continue

                alphas[companyi] += (labelMat[companyj] * labelMat[companyi]) * (jthAlphaOld - alphas[companyj])
                
                b1 = b - ithPredicError - \
                     labelMat[companyi] * (alphas[companyi] - ithAlphaOld) * \
                     dataMatrix[companyi,:] * dataMatrix[companyi,:].T - \
                     labeMat[companyj] * (alphas[companyj] - jthAlphaOld) * \
                     dataMatrix[companyi,:] * dataMatrix[copmanyj,:].T

                b2 = b - jthPredicError - \
                     labelMat[companyi] * (alphas[companyi] - ithAlphaOld) * \
                     dataMatrix[companyi,:] * dataMatrix[companyj,:].T - \
                     labeMat[companyj] * (alphas[companyj] - jthAlphaOld) * \
                     dataMatrix[companyj,:] * dataMatrix[copmanyj,:].T

                if (0 < alphas[companyi]) and (C > alphas[companyi]):
                    b = b1
                elif (0 < alphas[companyj]) and (C > alphas[companyj]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1

                print "Changed alpha pair."

        if alphaPairsChanged  == 0:
            iter += 1
        else:
            iter = 0

        print "Iteration", iter

    return b, alphas
