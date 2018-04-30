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

# py Validation.py data thresh ratio_flag n [restriction]
def main():
	indent_counter = 0
	indent_counter += 1

	#flag indicating this is numerical data; i.e iris dataset
	if sys.argv[1] == "iris.data.txt":
		test,attr,classifiers = read_iris(sys.argv[1])
		csv_number_labels = None
	#else, this program can read any categorical numbers csv file
	else:
		test,attr = read_csv_numbers(sys.argv[1])
		classes = np.unique(test[:,-1])
		classifiers = {}

		for i in classes:
			classifiers[int(i)] = str(i)

	RootNode = Node("")
	
	try:
		restrictionsFile = open(sys.argv[5])
		restrictions = restrictionsFile.readlines()[0].split(",")[1:]
		for i in range(len(restrictions)-1,-1,-1):
			print(i)
			if restrictions[i] == '0':
				labeled_data = np.delete(labeled_data,i,axis=1)
				attr = np.delete(attr,i)

	except:
		print("No restrictions file found/inputted")

	#csv_number_labels is null for numeric
	#C45(labeled_data, attr, RootNode, classifiers, indent_counter,csv_number_labels, f2)

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