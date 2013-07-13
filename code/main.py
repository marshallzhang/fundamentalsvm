import data_loader as data
import platt
import operator
import adaboost

START = 0
END = 200

companies = data.companyDataLoader('../data/final_company_data', False, START, END)
dataMatrix, labelMatrix = companies.getMatrices()

testcompanies = data.companyDataLoader('../data/final_company_data', False, END + 1, END + (END - START) + 1)
testData, testLabels = testcompanies.getMatrices()

def test(C,toler,maxIter):
    svm = platt.Platt()
    svm.loadData(dataMatrix,labelMatrix)
    svm.train(C, toler, maxIter)

    i = 0

    result = {}
    for weight in svm.weights:
        i += 1
        result[i] = weight


    correct = 0
    total = 0

    for i in range(len(testLabels)):
        if svm.classify(testData[i]) == testLabels[i]:
            correct += 1
            total += 1
        else:
            total += 1
    print "Correct: " +str(float(correct) / float(total)) + "%"

    return float(correct) / float(total)
def testadaboost(iterations):
    adabooster = adaboost.adaBoost()
    adabooster.loadData(dataMatrix,labelMatrix)
    adabooster.boost(iterations)
    
    correct = 0
    total = 0
    for i in range(len(testLabels)):
        if adabooster.classify(testData[i]) == testLabels[i]:
            correct += 1
            total += 1
        else:
            total += 1
    print "Correct: ", str(float(correct) / float(total)) + "%"


testadaboost(25)
#bestC = 0
#bestT = 0
#bestCorrect = 0
#for C in range(0, 50, 5):
#    for toler in range(1, 100, 10):
#        print "C",float(C)/ 10.0
#        print "toler",float(toler)/100
#        curCorrect = test(float(C) / 10.0, float(toler) / 100, 40) 
#        
#        if curCorrect > bestCorrect:
#            bestC = C
#            bestT = toler
#            bestCorrect = curCorrect
#
#        print "best C", bestC
#        print "best T", bestT
#        print "best correct", bestCorrect
#         
