import numpy as np
import math
import xml.etree.ElementTree as ET
import sys

#run program with this:
#	python InduceC45.py <domainFile.xml> <TrainingSetFile.csv> <threshold_float> [<restrictionsFile>]
 
	

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

def indent(indent_counter):
	return '	'*indent_counter

#csv_number_labels is None for numeric
#isNumeric equals 1 for numeric data
def C45(test, attr, RootNode, classifiers, indent_counter, csv_number_labels):
	isNumeric = 0
	if(csv_number_labels == None):
		isNumeric = 1
	if(len(np.unique(test[:,-1])) == 1):
		#print(classifiers[test[0,-1]])
		RootNode.leaf = Leaf(classifiers[test[0,-1]], test[0,-1], 1)
		print(indent(indent_counter) + '<decision end = '+classifiers[test[0,-1]]+' choice = "'+str(test[0,-1])+'" p = "1.00"/>')
	elif(len(attr) == 0):
		freq = findMostFrequent(test)
		#print(classifiers[test[0,-1]])
		RootNode.leaf = Leaf(classifiers[freq[0]], freq[0], freq[1])
		print(indent(indent_counter) + '<decision end = '+classifiers[freq[0]]+' choice = "'+str(freq[0])+'" p = "'+str(freq[1])+'"/>')
	else:
		splitNum, alpha = selectSplitting(attr, test, sys.argv[3], isNumeric)
		if(splitNum == -1):
			freq = findMostFrequent(test)
			RootNode.leaf = Leaf(classifiers[freq[0]], freq[0], freq[1])
			print(indent(indent_counter) + '<decision end = '+classifiers[freq[0]]+' choice = "'+str(freq[0])+'" p = "'+str(freq[1])+'"/>')
		else:
			RootNode.attName = attr[splitNum]
			#print(RootNode.attName)
			print(indent(indent_counter)+'<node var = "'+RootNode.attName+'">')
			indent_counter += 1
			if(isNumeric == 0):
				for v in np.unique(test[: ,splitNum]):
					#print(v)
					Dv = test[test[:, splitNum] == v]
					Dv = np.delete(Dv, splitNum, axis = 1)
					curAttr = np.delete(attr, splitNum)
					tempNode = Node("")
					print(indent(indent_counter) + '<edge var = "'+v+'" num="'+find_num(csv_number_labels,v)+'">')
					C45(Dv, curAttr, tempNode, classifiers, indent_counter+1, csv_number_labels)
					print(indent(indent_counter) + "</edge>")
					newEdge = Edge(v, tempNode)
					RootNode.edges.append(newEdge)
				print(indent(indent_counter-1) + "</node>")
			else:
				Dv1 = test[test[:, splitNum] <= alpha]
				#print(len(Dv1))
				if(len(np.unique(Dv1[:, splitNum]) == 1)):
					Dv1 = np.delete(Dv1, splitNum, axis = 1)
					curAttr = np.delete(attr, splitNum)
				tempNode = Node("")
				print(indent(indent_counter) + '<edge var = "<= ' + str(alpha) +'">' )
				C45(Dv1, curAttr, tempNode, classifiers, indent_counter+1, None)
				print(indent(indent_counter) + "</edge>")
				newEdge = Edge(-alpha, tempNode)
				RootNode.edges.append(newEdge)
				Dv2 = test[test[:, splitNum] > alpha]
				if(len(Dv2) != 0):
					if(len(np.unique(Dv2[:, splitNum]) == 1)):
						Dv2 = np.delete(Dv2, splitNum, axis = 1)
						curAttr2 = np.delete(attr, splitNum)

					tempNode2 = Node("")
					print(indent(indent_counter) + '<edge var = "> ' + str(alpha) +'">' )
					C45(Dv2, curAttr2, tempNode2, classifiers, indent_counter+1, None)
					print(indent(indent_counter) + "</edge>")
					newEdge2 = Edge(alpha, tempNode2)
					RootNode.edges.append(newEdge2)
				print(indent(indent_counter-1) + "</node>")

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
	return arr,attNames,classifiers


def parse_xml(filepath, attr):
	csv_number_labels = []
	tree = ET.parse(filepath)
	root = tree.getroot()
	classifiers = {}
	for child in root:
		labels = [child.attrib['name']]
		if(child.tag == "Category"):
			for subcat in child:
				classifiers[subcat.attrib['name']] = subcat.attrib['type']
		for subchild in child:

			labels.append(subchild.attrib['name'])
		csv_number_labels.append(labels)
	return csv_number_labels,classifiers

def find_num(csv_number_labels, v):
	i = 0
	for x in csv_number_labels:
		j = 0
		for y in x:
			if y == v:
				return str(j)
			j+=1
		i += 1


def main():
	indent_counter = 0
	print('<Tree name = "test">')
	indent_counter += 1

	#flag indicating this is numerical data; i.e iris dataset
	if sys.argv[1] == "NULL":
		labeled_data,attr,classifiers = read_iris(sys.argv[2])
		csv_number_labels = None
	#else, this program can read any categorical numbers csv file
	else:
		test,attr = read_csv_numbers(sys.argv[2])
		csv_number_labels, classifiers = parse_xml(sys.argv[1], attr)
		labeled_data = np.empty(test.shape, dtype = "object")
		for col in range(test.shape[1]):
			for row in range(test.shape[0]):
				labeled_data[row,col] = csv_number_labels[col][int(test[row,col])]
	RootNode = Node("")
	
	try:
		restrictionsFile = open(sys.argv[5])
		restrictions = restrictionsFile.readlines()[0].split(",")[1:]
		print(len(restrictions))
		for i in range(len(restrictions)-1,-1,-1):
			print(i)
			if restrictions[i] == '0':
				labeled_data = np.delete(labeled_data,i,axis=1)
				attr = np.delete(attr,i)
			print(attr)

	except:
		print("No restrictions file found/inputted")

	#csv_number_labels is null for numeric
	C45(labeled_data, attr, RootNode, classifiers, indent_counter,csv_number_labels)
	print("</tree>")

	

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
		leaf = None
		

if __name__ == "__main__":
	main()