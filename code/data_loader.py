from snapshot import CompanySnapshot
import os
import re
import time

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
                        self.dataMatrix.append(company.signals)
                        self.labelMatrix.append(company.earnings)
                    except BadCompanyData:
                        print "BAD COMPANY DATA"
                        pass


                else:
                    loaded += 1


            # get all the companies
            else:
                try:
                    company = CompanySnapshot(self.directory,file)
                    self.dataMatrix.append(company.signals)
                    self.labelMatrix.append(company.earnings)
                except BadCompanyData:
                    pass


    # return data and labels
    def getMatrices(self):
        return self.dataMatrix,self.labelMatrix

class CompanySnapshot():
    
    def __init__(self,directory,filename):
        self.name = None
        self.date = None
        self.signals = []
        self.earnings = None
        
        self.directory = directory

        self.loadData(filename)
        self.setEarnings()
        self.dataCheck()
    
    # populate instance variables
    def loadData(self,filename):
        
        print "Loading " + filename + '...'
        
        # get name and date
        parse_filename = re.search('([\w\d-]*)_([\w\d-]*).txt',filename)
        self.name = parse_filename.group(1)
        self.date = time.strptime(parse_filename.group(2),"%d-%b-%Y")
        
        # get signals and earnings
        f = open(self.directory + '/' + filename,'r')
        for line in f.readlines():
            if line == '\n':
                self.signals.append(None)
            else:
                self.signals.append(float(line.rstrip()))
        f.close()
    
    # get earnings +/- from last element of signals
    def setEarnings(self):
        try:
            self.earnings = self.signals.pop()
        except IndexError:
            print "no earnings"
            raise BadCompanyData

    # check if company has any signals
    def dataCheck(self):
        if None in self.signals:
            print "none"
            raise BadCompanyData
        if float(self.earnings) != -1.0 and float(self.earnings) != 1.0:
            print "bad earnings"
            raise BadCompanyData

class BadCompanyData(Exception):
    pass
