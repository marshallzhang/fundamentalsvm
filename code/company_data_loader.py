from company_snapshot import CompanySnapshot
from custom_exceptions import BadCompanyData
import os

class companyDataLoader():
    
    # initalize empty matrices
    def __init__(self,directory,complete,start,toload):
        self.dataMatrix = []
        self.labelMatrix = []
        self.complete = complete 
        self.directory = directory
        self.start = start
        self.toload = toload

        self.initHelper()

    # fill in matrices with first half of companies
    def initHelper(self):
        loaded = 0

        for file in os.listdir(self.directory):
            
            # get only half the companies
            if not self.complete:
                if loaded > self.start and loaded < self.toload:
                    try:
                        company = CompanySnapshot(self.directory,file)
                        loaded += 1
                    except BadCompanyData:
                        pass

                    self.dataMatrix.append(company.signals)
                    self.labelMatrix.append(company.earnings)

                else:
                    loaded += 1


            # get all the companies
            else:
                try:
                    company = CompanySnapshot(self.directory,file)
                except BadCompanyData:
                    pass

                self.dataMatrix.append(company.signals)
                self.labelMatrix.append(company.earnings)

    # return data and labels
    def getMatrices(self):
        return self.dataMatrix,self.labelMatrix
