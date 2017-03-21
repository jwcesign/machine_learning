from numpy import *
import operator

group = array([[1,1.1],[1,1],[0,0],[0,0.1]])
labels = ['A','A','B','B']

def knn(intX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(intX,(dataSetSize,1))-dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

