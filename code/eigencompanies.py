import numpy as np
from company_data import DataLoader

class EigenCompanies():

    # init
    def __init__(self, num_eigenvecs, directory, complete=True, start=0, end=0):
        self.dataMatrix = []
        self.covarMatrix = []
        self.meanMatrix = []
        self.eigenValues = []
        self.eigenVectors = []
        self.eigenCompanies = []
        self.labelMatrix = []

        # row is one feature, column is one company
        data = DataLoader(directory, complete, start, end)
        self.dataMatrix, self.labelMatrix = data.getMatrices()
        self.dataMatrix = np.transpose(np.array(self.dataMatrix))

        # compute eigencompanies
        self.getCovariance()
        self.getEigenVs()
        self.getEigenCompanies(num_eigenvecs)

    def getCovariance(self):
        
        for feature in self.dataMatrix:
            
            # compute average of feature
            mean = np.mean(feature)
            mean_vector = np.empty((np.shape(self.dataMatrix)[1],))
            mean_vector[:] = mean
            mean_feature_vector = feature - mean_vector
            self.meanMatrix.append(mean_feature_vector)

        self.meanMatrix = np.array(self.meanMatrix)
        self.covarMatrix = np.cov(self.meanMatrix)

    def getEigenVs(self):

        self.eigenValues, self.eigenVectors = np.linalg.eig(self.covarMatrix)

    def getEigenCompanies(self, num_eigenvecs):
        principal_basis = []
        
        # create num_eigenvecs x num_features matrix
        for i in range(num_eigenvecs):
            principal_basis.append(np.transpose(self.eigenVectors[:,i]))

        # eigenCompanies has one row for company and one column for feature
        for feature in np.transpose(self.meanMatrix):
            new_feature = np.dot(principal_basis, feature)
            self.eigenCompanies.append(np.transpose(new_feature)) 

    def getMatrices(self):
        return self.eigenCompanies, self.labelMatrix
