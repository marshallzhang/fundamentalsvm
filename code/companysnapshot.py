import re, os
import time

class NoSignals(Exception):
    pass

class CompanySnapshot():
    
    def __init__(self,filename):
        self.name = None
        self.date = None
        self.signals = []
        self.earnings = None
        
        self.loadData(filename)
        self.setEarnings()
        self.noneCheck()
    
    # populate instance variables
    def loadData(self,filename):
        
        print "Loading " + filename + '...'
        
        # get name and date
        parse_filename = re.search('([\w\d-]*)_([\w\d-]*).txt',filename)
        self.name = parse_filename.group(1)
        self.date = time.strptime(parse_filename.group(2),"%d-%b-%Y")
        
        # get signals and earnings
        f = open('../data/company_data/' + filename,'r')
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
            raise NoSignals

    # check if company has any signals
    def noneCheck(self):
        if not [x for x in self.signals if x != None]:
            raise NoSignals
