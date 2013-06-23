import data_loader as data
import platt
import operator

START = 0
END = 450 

companies = data.companyDataLoader('../data/company_data', False, START, END)
dataMatrix, labelMatrix = companies.getMatrices()

svm = platt.Platt()
svm.loadData(dataMatrix,labelMatrix)
svm.train(0.8,0.001,40)

i = 0

result = {}
for weight in svm.weights:
    i += 1
    result[i] = weight

testcompanies = data.companyDataLoader('../data/company_data', False, END + 1, END + (END - START) + 1)
testData, testLabels = testcompanies.getMatrices()

correct = 0
total = 0

for i in range(len(testLabels)):
    print svm.classify(testData[i]), testLabels[i]
    if svm.classify(testData[i]) == testLabels[i]:
        correct += 1
        total += 1
    else:
        total += 1

print "Correct: " + str(float(correct) / float(total)) + "%"
