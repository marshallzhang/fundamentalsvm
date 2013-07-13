import numpy as np
import math
from collections import Counter
import os
from eigencompanies import EigenCompanies

class adaBoost():

	def __init__(self, num_features):

		self.data 			 = np.matrix([])
		self.labels 		 = np.matrix([])
		self.classifierArray = np.matrix([])
		self.NUM_FEATURES    = num_features 
		self.featuresMatrix  = np.zeros((1,self.NUM_FEATURES))

        def loadData(self,dataIn,classLabelsIn):
            self.data = np.array(dataIn, dtype=float)
            self.data[np.isnan(self.data)] = 0 
            self.featuresMatrix = self.data
            print np.shape(self.featuresMatrix)
            self.labels = np.array(classLabelsIn)
            self.labels = np.transpose(self.labels)

	# weak classification given a weak classifier
	def guessClass(self,featuresMatrix,feature,threshold,inequality):
		# get dimensions of features matrix
		try:
			n,m = np.shape(featuresMatrix)
		except ValueError:
			n = 1

		classed = (np.ones((1,n)))[0]

		# weak guesser for one image
		if n == 1:
			if inequality == '<=':
					if featuresMatrix[feature] <= threshold:
						classed = -1
			else:
					if featuresMatrix[feature] > threshold:
						classed = -1

		# weak guesser for a set of images
		else:
			if inequality == '<=':
				for img in range(0,n):
					if featuresMatrix[img][feature] <= threshold:
						classed[img] = -1
			else:
				for img in range(0,n):
					if featuresMatrix[img][feature] > threshold:
						classed[img] = -1
		
		return classed

	# trains a weak classifier
	def trainClassifier(self,data,labels,weights,steps,weakClassGuessers):

		# setup
		labelMatrix = np.matrix(labels).T
		try:
			n,m,o = np.shape(data)
		except ValueError:
			try: 
				n,m = np.shape(data)
			except:
				n = len(data)
		bestClassifier = {}
		bestClassGuess = np.zeros((n,1))
		minError = float('inf')

		# main training loop
		for feature in range(0,self.NUM_FEATURES):

			# min and max of one column of featuresMatrix, representing the values
			# of one particular feature across all the images in our training set
			rangeMin = self.featuresMatrix[:,feature].min()
			rangeMax = self.featuresMatrix[:,feature].max()

			# calculate our stepsize based on max and min
			stepSize = (rangeMax - rangeMin) / float(steps)

			# for every little subdivision of the range between min and max
			for j in range(-1,steps + 1):
				for inequality in ['<=','>']:

					# set the threshold value by incrementing one step over minrange
					# and classify based on the threshold
					threshold = (rangeMin + (j * stepSize))
					classGuess = self.guessClass(self.featuresMatrix,feature,threshold,inequality)
					
					# make a vector to record errors
					errorArray = np.ones((n,1))

					# if correct prediction, set error for that image to 0
					for img in range(0,len(self.featuresMatrix)):
						if classGuess[img] == labelMatrix[img]:
							errorArray[img] = 0

					# get weighted error
					weightedError = np.matrix(weights).T * np.matrix(errorArray)

					# if weighted error is smallest, then the feature,
					# threshold, and inequality into our temporary bestClassifier
					if weightedError < minError:

						# also ensure if we have any weakGuessers already,
						# the new classifier should be based on a different feature
						if weakClassGuessers:
							if feature not in [weakClassGuessers[x]['feature'] for x in range(0,len(weakClassGuessers))]:
								minError = weightedError
								bestClassGuess = classGuess.copy()
								bestClassifier['feature'] = feature
								bestClassifier['threshold'] = threshold
								bestClassifier['inequality'] = inequality

						else:
							minError = weightedError
							bestClassGuess = classGuess.copy()
							bestClassifier['feature'] = feature
							bestClassifier['threshold'] = threshold
							bestClassifier['inequality'] = inequality

		return bestClassifier,minError,bestClassGuess

	# creates a strong classifier from maxFeatures weak classifiers
	def boost(self,maxFeatures):

		# setup
		self.classifierArray = []
		weakClassGuessers = []
		try:
			n,m,o = np.shape(self.data)
		except ValueError:
			try: 
				n,m = np.shape(self.data)
			except ValueError:
				n = len(self.data)
		weights = np.ones((n,1))
		weights = weights * (1. / n)
		aggregateClassGuess = np.zeros((n,1))

		# main boosting loop
		for i in range (0,maxFeatures):
			
			print "Computing weak classifier #", i+1, "of", maxFeatures, "for inclusion in our strong classifier"

			# train best classifier for current set of image weights
			bestClassifier,error,classGuess = self.trainClassifier(self.data,self.labels,weights,10,weakClassGuessers)
                        print bestClassifier
			# calculate weight of the classifier
			alpha = float(math.log(1.0 - error) / max(error,1e-16))
			bestClassifier['alpha'] = alpha

			# add classifier to weakClassGuessers
			weakClassGuessers.append(bestClassifier)

			# calculate new weights
			exponent = np.multiply(alpha * np.matrix(self.labels), classGuess)
			weights = np.multiply(weights,np.exp(exponent.T))
			weights = weights * (1 / weights.sum())

			# update aggregateClassGuess with newest classifier's guess
			aggregateClassGuess = aggregateClassGuess + np.matrix((-1 * alpha * classGuess)).T

			# calculate error of aggregateClassGuess and errorRate
			aggregateErrors = np.multiply(np.sign(aggregateClassGuess) != np.matrix(self.labels).T, np.ones((n,1)))
			errorRate = aggregateErrors.sum() / n

			if errorRate == 0.0: 
				break

		self.classifierArray = weakClassGuessers

	# classifies an inputted image using our strong boosted classifier
	def classify(self,featuresMatrix):

		# compute features vector of input
		aggregateClassGuess = 0

		# for every classifier we train, use it to classguess and then scale by
		# alpha and add to aggregate guess
		for classifier in range (0,len(self.classifierArray)):
			classGuess = self.guessClass(featuresMatrix,self.classifierArray[classifier]['feature'],self.classifierArray[classifier]['threshold'],self.classifierArray[classifier]['inequality'])
			aggregateClassGuess = aggregateClassGuess + (self.classifierArray[classifier]['alpha'] * classGuess)

		try:
                    return np.sign(aggregateClassGuess)[0]
                except IndexError:
                    return np.sign(aggregateClassGuess)

class Cascade:

	def __init__(self):

		self.subwindow 				= []
		self.falsePositiveRate 	    = 1.0
		self.detectionRate          = 1.0
		self.positiveSet 		    = self.loadPositives()
		self.negativeSet  	        = self.loadNegatives()
		self.cascadedClassifier     = {}

	# load initial set of positive images
	def loadPositives(self,positiveDir="../data/sameface/"):

		positiveSet = []

		positiveImages = os.listdir(positiveDir)

		positiveImages.pop(0)

		# add each vector to the list
		for i in positiveImages:
			positiveSet.append(get_frame_vector(positiveDir + i,False))

		return positiveSet

	# load initial set of negative images
	def loadNegatives(self,negativeDir="../data/randombg/"):

		negativeSet = []

		negativeImages = os.listdir(negativeDir)

		negativeImages.pop(0)

		# add each vector to the list
		for i in negativeImages:
			negativeSet.append(get_frame_vector(negativeDir + i,False))

		return negativeSet

	# classifies a set of images and returns a dictionary with their classification
	def cascadedClassifierGuess(self,data,adabooster):

		classifiedDict = {}

		# for every image
		for i in data:

			# initialize guess to true
			classifiedDict[i] = 1

			# setup
			features = Features(i)
			featuresMatrix = features.f
			n = len(data)
			aggregateClassGuess = 0

			# for each strong classifier in our cascaded classifier
			for layer,classifier in self.cascadedClassifier.items():

				# for each weak classifier in our strong classifier
				for x in range (0,len(classifier)):

					# use the weak classifier to guessClass and update aggregateClassGuess
					classGuess = adabooster.guessClass(featuresMatrix,classifier[x]['feature'],classifier[x]['threshold'],classifier[x]['inequality'])
					aggregateClassGuess = aggregateClassGuess + (-1 * classifier[x]['alpha'] * classGuess)

				# if any layer returns a negative result, automatically return negative
				if np.sign(aggregateClassGuess) == -1:
					classifiedDict[i] = -1
					break

		return classifiedDict

	# adjust threshold of classifier to improve detection rate
	def adjustThreshold(self,classifier,n):

		# adjust threshold for each weakclassifier in strong classifier
		for i in range(0,n):
			if classifier[n][i]['inequality'] == "<=":
				classifier[n][i]['threshold'] -= 2
			else:
				classifier[n][i]['threshold'] += 2

	# train a cascadedClassifier based on target rates for detection and false positives
	def trainCascadedClassifier(self,f,d,Ftarget):

		# initialize an adaBoost instance and load positive/negative training set
		adabooster = adaBoost()
		adabooster.loadData()

		# while our overall false positive rate is too high
		while self.falsePositiveRate > Ftarget:

			n = 0
			newFalsePositiveRate = self.falsePositiveRate

			# while the false positive rate for our current set of negative images 
			# (our "local" false positive rate) is too high
			while newFalsePositiveRate > (f * self.falsePositiveRate):

				n += 1

				print "Training cascaded classifier layer #", n

				# on our second pass, reload our positive and negative sets
				if n > 1:
					adabooster.loadDataFromMatrices(self.positiveSet,self.negativeSet)

				# create a strong classifier made of n weak classifiers based on 
				# current training sets
				adabooster.boost(n)

				# add our new classifier to our cascadedClassifier
				self.cascadedClassifier[n] = adabooster.classifierArray

				print "Computing current cascaded classifier's false positive rate..."

				# find our classifier's false positive and detection rate
				negativeSetGuesses = self.cascadedClassifierGuess(self.negativeSet,adabooster)
				ncnt = Counter()
				for k,v in negativeSetGuesses.items():
					ncnt[v] += 1

				# "local" false positive rate
				newLocalFalsePositiveRate = float(ncnt[1]) / float(len(negativeSetGuesses.items()))

				print "Computing current cascaded classifier's detection rate..."

				positiveSetGuesses = self.cascadedClassifierGuess(self.positiveSet,adabooster)
				pcnt = Counter()
				for k,v in positiveSetGuesses.items():
					pcnt[v] += 1
				newDetectionRate = float(pcnt[1]) / float(len(positiveSetGuesses.items()))

				# adjust the most recently added classifier to improve detection rate
				while newDetectionRate < d * self.detectionRate:

					print "Adjusting our latest strong classifier to achieve a detection rate of",d,"..."

					self.adjustThreshold(self.cascadedClassifier,n)

					print self.cascadedClassifier

					# re-guess
					positiveSetGuesses = self.cascadedClassifierGuess(self.positiveSet,adabooster)

					# compute new detection rate
					cnt = Counter()
					for k,v in positiveSetGuesses.items():
						cnt[v] += 1
					newDetectionRate = float(cnt[1]) / float(len(positiveSetGuesses.items()))

					print newDetectionRate
			
				tempNegativeSet = []

				if newFalsePositiveRate > f:

					print "Resetting our negative training set to our cascaded classifier's false positives..."

					# replace our current negative training set with only our current false positives
					negativeSetGuesses = self.cascadedClassifierGuess(self.negativeSet,adabooster)
					for (k,v) in negativeSetGuesses.items():
						if v == 1:
							tempNegativeSet.append(k)

					self.negativeSet = tempNegativeSet

				# compute new "local" false positive rate
				newFalsePositiveRate = newFalsePositiveRate * newLocalFalsePositiveRate

				print "Current cascaded classifier's false positive rate is",newFalsePositiveRate
				print "Current cascaded classifier's detection rate is",newDetectionRate

			# overall false positive rate
			self.falsePositiveRate = newFalsePositiveRate

		print "Done training cascaded classifier! Ready to classify."

	# classify one image using cascaded classifier
	def cascadedClassify(self, i):

		adabooster = adaBoost()
		result = self.cascadedClassifierGuess([i],adabooster)
		return ([v for (k,v) in result.items()][0] == -1)
