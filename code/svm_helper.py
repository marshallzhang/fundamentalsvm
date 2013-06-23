from company_snapshot import CompanySnapshot
from custom_exceptions import BadCompanyData
import numpy as np
import os
import random

def loadData(directory):
    dataMat = []; labelMat = []
    for file in os.listdir(directory):
        try:
            company = CompanySnapshot(file)
            dataMat.append(company.signals)
            labelMat.append(company.earnings)
        except BadCompanyData:
            pass
    return dataMat, labelMat

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

# data, labels = loadData('../data/company_data')
# print len([x for x in labels if x == 1.0])
# print len(labels)
# print loadData('../data/company_data')
