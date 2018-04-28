import numpy as np
import math
import xml.etree.ElementTree as ET
import sys

#run program with this:
#	python InduceC45.py <domainFile.xml> <TrainingSetFile.csv> <threshold_float> [<restrictionsFile>]
 
	

def selectSplitting(attr, data, thresh):
	dEntropy = findEntropy(data)
	#print(dEntropy)
	dSize2 = data.shape[0]
	gain = []
	#iterate through each attribute
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
		#gainEntropy = -gainEntropy
		#print(findEntropy(dataWithValue))
		if(sys.argv[4] == 1):
			gain.append(curGain/gainEntropy)
		else:
			gain.append(curGain)
		#print(curGain/gainEntropy)
	bestIndex = gain.index(max(gain))
	if(gain[bestIndex] > float(thresh)):
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

def indent(indent_counter):
	return '	'*indent_counter


def C45(test, attr, RootNode, classifiers, indent_counter, csv_number_labels):


	if(len(np.unique(test[:,-1])) == 1):
		#print(classifiers[test[0,-1]])
		RootNode.leaf = Leaf(classifiers[test[0,-1]], test[0,-1], 1)
		print(indent(indent_counter) + '<decision end = '+classifiers[test[0,-1]]+' choice ="'+test[0,-1]+'" p = "1.00"/>')
	elif(len(attr) == 0):
		freq = findMostFrequent(test)
		#print(classifiers[test[0,-1]])
		RootNode.leaf = Leaf(classifiers[freq[0]], freq[0], freq[1])
		print(indent(indent_counter) + '<decision end = '+classifiers[freq[0]]+' choice ="'+freq[0]+'" p = "'+str(freq[1])+'"/>')
	else:
		splitNum = selectSplitting(attr, test, sys.argv[3])
		if(splitNum == -1):
			freq = findMostFrequent(test)
			RootNode.leaf = Leaf(classifiers[freq[0]], freq[0], freq[1])
			print(indent(indent_counter) + '<decision end = '+classifiers[freq[0]]+' choice ="'+freq[0]+'" p = "'+str(freq[1])+'"/>')
		else:
			RootNode.attName = attr[splitNum]
			#print(RootNode.attName)
			print(indent(indent_counter)+'<node var = "'+RootNode.attName+'">')
			indent_counter += 1
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
	attNames = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
	arr = np.empty((len(lines),len(attNames)),dtype = "float64")

	classifiers = {"Iris-setosa":0.0,"Iris-versicolor":1.0,"Iris-virginica":2.0}

	for i in range(len(lines)):
		line = lines[i].rstrip().split(",")
		sl = float(line[0])
		sw = float(line[1])
		pl = float(line[2])
		pw = float(line[3])
		iris = float(classifiers[line[4]])
		arr[i] = [sl,sw,pl,pw,iris]
	return(arr,attNames)


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
		test,attr = read_iris(sys.argv[2])
		print(test)
		return

	test,attr = read_csv_numbers(sys.argv[2])
	csv_number_labels, classifiers = parse_xml(sys.argv[1], attr)
	


	labeled_data = np.empty(test.shape, dtype = "object")
	for col in range(test.shape[1]):
		for row in range(test.shape[0]):
			labeled_data[row,col] = csv_number_labels[col][int(test[row,col])]
	RootNode = Node("")
	



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