import numpy as np
import math

def selectSplitting(attr, data, thresh, ratio):
	dEntropy = findEntropy(data)
	print(dEntropy)
	dSize2 = data.shape[0]
	gain = []
	#iterate through each attribute
	for i in range(0,len(attr)):
		attrEntropy = 0
		#iterate through each value of an attribute
		for uniqueValue in np.unique(data[:, i]):
			dataWithValue = data[data[:,i] == uniqueValue]
			attrEntropy += len(dataWithValue)/dSize2*findEntropy(dataWithValue)
		curGain = dEntropy - attrEntropy
		if(ratio):
			gain.append(curGain/attrEntropy)
		else:
			gain.append(curGain)
		print(attrEntropy)
	bestIndex = gain.index(max(gain))
	if(gain[bestIndex] > thresh):
		return bestIndex
	else:
		return -1

def findEntropy(data):
	uniqueClassifier = np.unique(data[:,-1])
	entropy = 0
	dSize = data.shape[0]
	for c in uniqueClassifier:
		isC = data[data[:,-1] == c]
		lengthSplit = isC.shape[0]
		entropy += (lengthSplit/dSize)*math.log(lengthSplit/dSize, 2)
	return -entropy

def findMostFrequent(data):
	uniqueClassifiers = np.unique(data[:,-1])
	countEach = {}
	for item in uniqueClassifiers:
		count = len(data[data[:,-1] == item])
		countEach[item] = count
	bestTuple = max(countEach.items(), key = lambda x: x[1])
	return (bestTuple[0], bestTuple[1]/len(data))

	


def main():
	test = np.array([[3,"N","T","S","N"],
					 [3,"Y","T","S","Y"],
					 [3,"Y","O","N","N"],
					 [3,"Y","T","N","N"],
					 [3,"N","O","N","N"],
					 [3,"Y","T","S","Y"],
					 [3,"Y","O","S","N"],
					 [3,"N","T","S","N"],
					 [4,"N","T","S","Y"],
					 [4,"Y","O","N","N"],
					 [4,"Y","O","S","Y"],
					 [4,"N","T","N","N"],
					 [4,"N","O","S","Y"],
					 [4,"Y","O","S","Y"],
					 [4,"N","T","N","N"],
					 [4,"Y","O","N","N"]])
	attr = np.array(["1","2","3","4'"])
	classifiers = {"N": 1, "Y": 2}
	Tree = []
	if(len(np.unique(test[:,-1])) == 1):
		Tree.append(Leaf(1, test[0,-1], 1))
	elif(len(attr) == 0):
		freq = findMostFrequent(test)
		Tree.append(Leaf(classifiers[freq[0]], freq[0], freq[1]))
	print(selectSplitting(attr, test, .1, 0))

class Leaf:
	def __init__(self, decision, label, p):
		self.decision = decision
		self.label = label
		self.p = p

class Node:
	def __init__(self, attName, data):
		self.edges = []
		self.attName = attName
		self.data = data
		

if __name__ == "__main__":
	main()