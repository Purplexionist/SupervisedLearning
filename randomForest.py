#Random forest

#How to run:
# py randomForest.py data thresh ratio_flag n m k N [restrictions.csv]

import numpy as np
import math
import xml.etree.ElementTree as ET
import sys


def read_csv_numbers(filepath):
	file = open(filepath, 'r')
	lines = file.readlines()
	file.close()
	attNames = lines[0].rstrip().split(",")[1:]
	attLevels = list(map(int, lines[1].rstrip().split(",")[1:]))
	classifier = lines[2].rstrip()

	arr = np.empty((len(lines)-3,len(attLevels)))

	for i in range(len(lines)-3):
		arr[i] = lines[3+i].rstrip().split(",")[1:]

	return(arr,attNames[0:-1])




def selectSplitting(attr, data, thresh, isNumeric):
	dEntropy = findEntropy(data)
	#print(dEntropy)
	dSize2 = data.shape[0]
	gain = []
	#iterate through each attribute
	if(isNumeric == 0):
		for i in range(0,len(attr)):
			attrEntropy = 0
			gainEntropy = 0
			#iterate through each value of an attribute
			for uniqueValue in np.unique(data[:, i]):
				dataWithValue = data[data[:,i] == uniqueValue]
				attrEntropy += len(dataWithValue)/dSize2*findEntropy(dataWithValue)
				gainEntropy += len(dataWithValue)/dSize2*math.log(len(dataWithValue)/dSize2, 2)
			gainEntropy = -gainEntropy
			if(gainEntropy == 0):
				gainEntropy = 1
			curGain = dEntropy - attrEntropy
			if(sys.argv[4] == 1):
				gain.append(curGain/gainEntropy)
			else:
				gain.append(curGain)
			#print(curGain/gainEntropy)
	else:
		for i in range(0, len(attr)):
			gainArray = []
			for num in np.unique(data[:, i]):
				attrEntropy = 0
				gainEntropy = 0
				dataUnder = data[data[:, i] <= num]
				dataAbove = data[data[:, i] > num]
				if(len(dataUnder) != 0):
					attrEntropy += len(dataUnder)/dSize2*findEntropy(dataUnder)
				if(len(dataAbove) != 0):
					attrEntropy += len(dataAbove)/dSize2*findEntropy(dataAbove)
				if(len(dataUnder) != 0):
					gainEntropy += len(dataUnder)/dSize2*math.log(len(dataUnder)/dSize2, 2)
				if(len(dataAbove) != 0):
					gainEntropy += len(dataAbove)/dSize2*math.log(len(dataAbove)/dSize2, 2)
				gainEntropy = -gainEntropy
				if(gainEntropy == 0):
					gainEntropy = 1
				curGain = dEntropy - attrEntropy
				if(sys.argv[4] == 1):
					gainArray.append([num, curGain/gainEntropy])
				else:
					gainArray.append([num, curGain])
			numMax = -999
			alphaBest = -1
			for inner in gainArray:
				if(inner[1] > numMax):
					numMax = inner[1]
					alphaBest = inner[0]
			gain.append([i, alphaBest, numMax])
	if(isNumeric == 1):
		curIndex = -1
		curBestAlpha = -99
		curBestNum = -99
		for miniList in gain:
			if(miniList[2] > curBestNum):
				curBestNum = miniList[2]
				curIndex = miniList[0]
				curBestAlpha = miniList[1]
		if(curBestNum > float(thresh)):
			return curIndex, curBestAlpha
		else:
			return -1, 0
	else:
		bestIndex = gain.index(max(gain))
		if(gain[bestIndex] > float(thresh)):
			return bestIndex, 0
		else:
			return -1, 0

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

def C45(test, attr, RootNode, classifiers, isNumeric):

	if(len(np.unique(test[:,-1])) == 1):
		#print(classifiers[test[0,-1]])
		RootNode.leaf = Leaf(classifiers[test[0,-1]], test[0,-1], 1)
		
	elif(len(attr) == 0):
		freq = findMostFrequent(test)
		#print(classifiers[test[0,-1]])
		RootNode.leaf = Leaf(classifiers[freq[0]], freq[0], freq[1])
		
	else:
		splitNum, alpha = selectSplitting(attr, test, sys.argv[2], isNumeric)
		if(splitNum == -1):
			freq = findMostFrequent(test)
			RootNode.leaf = Leaf(classifiers[freq[0]], freq[0], freq[1])
		else:
			RootNode.attName = attr[splitNum]
			
			
			if(isNumeric == 0):
				for v in np.unique(test[: ,splitNum]):
					#print(v)
					Dv = test[test[:, splitNum] == v]
					Dv = np.delete(Dv, splitNum, axis = 1)
					curAttr = np.delete(attr, splitNum)
					tempNode = Node("")
					C45(Dv, curAttr, tempNode, classifiers,isNumeric)
					newEdge = Edge(v, tempNode)
					RootNode.edges.append(newEdge)
			else:
				Dv1 = test[test[:, splitNum] <= alpha]
				#print(len(Dv1))
				if(len(np.unique(Dv1[:, splitNum]) == 1)):
					Dv1 = np.delete(Dv1, splitNum, axis = 1)
					curAttr = np.delete(attr, splitNum)
				tempNode = Node("")
				C45(Dv1, curAttr, tempNode, classifiers,isNumeric)
				newEdge = Edge(-alpha, tempNode)
				RootNode.edges.append(newEdge)
				Dv2 = test[test[:, splitNum] > alpha]
				if(len(Dv2) != 0):
					if(len(np.unique(Dv2[:, splitNum]) == 1)):
						Dv2 = np.delete(Dv2, splitNum, axis = 1)
						curAttr2 = np.delete(attr, splitNum)

					tempNode2 = Node("")
					C45(Dv2, curAttr2, tempNode2, classifiers,isNumeric)
					newEdge2 = Edge(alpha, tempNode2)
					RootNode.edges.append(newEdge2)

def read_iris(filepath):
	file = open(filepath,'r')
	lines = file.readlines()
	file.close()
	attNames = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
	arr = np.empty((len(lines),len(attNames)+1),dtype = "float64")

	labels = {"Iris-setosa":0.0,"Iris-versicolor":1.0,"Iris-virginica":2.0}
	classifiers = {0.0:"Iris-setosa",1.0:"Iris-versicolor",2.0:"Iris-virginica"}

	for i in range(len(lines)):
		line = lines[i].rstrip().split(",")
		sl = float(line[0])
		sw = float(line[1])
		pl = float(line[2])
		pw = float(line[3])
		iris = float(labels[line[4]])
		arr[i] = [sl,sw,pl,pw,iris]
	return arr,attNames,classifiers,labels

def findClass(row, rootNode, myDict, attr, flag, labels, label):
	if(rootNode.leaf != None):
		if(flag == 1):
			label.append(rootNode.leaf.label)	
			#print("Row:",str(row[0:-1]), ", Predicted:",rootNode.leaf.label)
		else:
			label.append(rootNode.leaf.decision)
	else:
		for i in rootNode.edges:
			if(flag == 1):
				if(i.choice < 0):
					if(float(row[attr.index(rootNode.attName)]) <= abs(i.choice)):
						findClass(row, i.Node, myDict, attr, 1,labels,label)
				else:
					if(float(row[attr.index(rootNode.attName)]) > abs(i.choice)):
						findClass(row, i.Node, myDict, attr, 1,labels,label)
			else:
				if(float(i.choice) == row[attr.index(rootNode.attName)]):
					findClass(row, i.Node, myDict, attr, 0,labels,label)


def false_positive(conf):
	sum = 0
	for i in range(len(conf)):
		for j in range(len(conf)):
			sum += conf[i,j]



def true_positive(conf):
	sum = 0
	for i in range(len(conf)):
		sum += conf[i,i]
	return(sum)

def removeDataForForest(curTrain,attr,m,kNum):
	mList = np.random.choice(list(range(len(attr))),m,False)
	mList.sort()

	tempTrain = curTrain
	newAttr = []
	for i in mList:
		newAttr.append(attr[i])
	for num in reversed(list(range(len(attr)))):
		if num not in mList:
			tempTrain = np.delete(tempTrain, num, axis = 1)
	
	kList = np.random.choice(list(range(len(curTrain))),kNum,False)
	kList.sort()
	for num in reversed(list(range(len(curTrain)))):
		if num not in kList:
			tempTrain = np.delete(tempTrain, num, axis = 0)
	return(tempTrain,newAttr)	
	


def main():
	#randomForest.py data thresh ratio_flag n m k N [restrictions.csv]

	m = int(sys.argv[5])
	kNum = int(sys.argv[6])
	nTree = int(sys.argv[7])
	
	isNumeric = 0
	#flag indicating this is numerical data; i.e iris dataset
	if "iris" in sys.argv[1]:
		isNumeric = 1
		train,attr,classifiers,labels = read_iris(sys.argv[1])
	#else, this program can read any categorical numbers csv file
	else:
		train,attr = read_csv_numbers(sys.argv[1])
		classes = np.unique(train[:,-1])
		classifiers = {}
		labels = {}

		for i in classes:
			classifiers[int(i)] = str(i)
			labels[str(i)] = int(i)

	RootNode = Node("")
	
	try:
		restrictionsFile = open(sys.argv[8])
		restrictions = restrictionsFile.readlines()[0].split(",")[1:]
		for i in range(len(restrictions)-1,-1,-1):
			if restrictions[i] == '0':
				train = np.delete(train,i,axis=1)
				del attr[i]

	except:
		print("No restrictions file found/inputted")

	np.random.shuffle(train)
	trees = []
	testData = []
	k = int(sys.argv[4])
	if(k == -1):
		k = len(train)
	if(k == 0 or k == 1):
		RootNode = Node("")
		curTrain = train
		C45(curTrain, attr, RootNode, classifiers, isNumeric)
		testData.append(train)
		trees.append(RootNode)
	else:
		lengthLeft = len(train)
		cumSum = 0
		foldsLeft = k
		n = len(train)
		for i in range(0, k):
			start = cumSum
			end = cumSum + lengthLeft//foldsLeft
			cumSum += lengthLeft//foldsLeft
			lengthLeft -= lengthLeft//foldsLeft
			foldsLeft -= 1
			testData.append(train[start : end])
			curTrain = np.append(train[0 : start], train[end : n], 0)
			foldTrees = []
			for numTrees in range(nTree):
				RootNode = Node("")
				tempData,tempAttr = removeDataForForest(curTrain,attr,m,kNum)
				C45(tempData, tempAttr, RootNode, classifiers,isNumeric)
				foldTrees.append(RootNode)
			trees.append(foldTrees)

	confusion_matrix = np.zeros((len(classifiers),len(classifiers)), dtype = 'int64')
	averages = []

	for i in range(len(testData)):
		tree = trees[i]
		testRows = testData[i]
		answerCollection = {}
		answerCollection["total"] = 0
		answerCollection["wrong"] = 0
		answerCollection["right"] = 0
		for row in testRows:
			label = []
			for miniTree in tree:
				findClass(row,miniTree,answerCollection,attr,isNumeric,labels, label)
			if(len(label) == 0):
				continue
			most =int(float(max(set(label), key = label.count)))
			real = int(row[-1])
			if isNumeric == 0:
				confusion_matrix[real-1][most-1] += 1
			else:
				confusion_matrix[real][most] += 1
			if(most == real):
				answerCollection["right"] += 1
			else:
				answerCollection["wrong"] += 1
			answerCollection["total"] += 1
		averages.append(answerCollection)

	if isNumeric == 0:
		save1 = confusion_matrix[0][0]
		confusion_matrix[0][0] = confusion_matrix[1][1]
		confusion_matrix[1][1] = save1
		save1 = confusion_matrix[1][0]
		confusion_matrix[1][0] = confusion_matrix[0][1]
		confusion_matrix[0][1] = save1
	if isNumeric == 0:
		print("McCain Obama")
	else:
		print("Iris-setoa Iris-versicolor Iris-virginica")
	print(confusion_matrix)
	if isNumeric == 0:
		tp = true_positive(confusion_matrix)	
		fp = confusion_matrix[0][1]
		fn = confusion_matrix[1][0]
		tn = confusion_matrix[0][0]
		accuracy = (tn+tp)/(tn+tp+fn+fp)
		recall = tp/(tp+fn)
		precision = tp/(tp+fp)
		pf = fp/(fp+tn)
		f = (2*precision*recall)/(precision+recall)
		print("Recall: "+str(recall))
		print("Precision: "+str(precision))
		print("pf: "+str(pf))
		print("f-measure: "+str(f))
		print("Overall Accuracy: "+str(accuracy))
		print("Overall Error: "+str(1-accuracy))
	if isNumeric == 1:
		right = confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2]
		total = (right + confusion_matrix[1][0] + confusion_matrix[2][0] + confusion_matrix[2][1] +
			confusion_matrix[0][1]+ confusion_matrix[0][2] + confusion_matrix[1][2])
		thisAcc = right / total
		print("Overall Accuracy: "+str(thisAcc))
		print("Overall Error: "+str(1-thisAcc))
	totCorrect = 0
	tot = 0
	avg = 0
	for i in averages:
		totCorrect += i["right"]
		tot += i["total"]
		avg += totCorrect/tot
	print("Average accuracy: "+str(avg/len(averages)))
	print("Average error: "+str(1-avg/len(averages)))

	

class Leaf:
	def __init__(self, decision, label, p):
		self.decision = decision
		self.label = label
		self.p = p

class Edge:
	def __init__(self, choice, Node):
		self.choice = choice
		self.Node = Node

class Node:
	def __init__(self, attName):
		self.edges = []
		self.attName = attName
		self.leaf = None
		

if __name__ == "__main__":
	main()

