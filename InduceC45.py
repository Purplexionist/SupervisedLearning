import numpy as np
import math

def selectSplitting(attr, data, thresh, ratio):
	dEntropy = findEntropy(data)
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
	bestIndex = gain.index(max(gain)))
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
selectSplitting(attr, test, .5, 0)

