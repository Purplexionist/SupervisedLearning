import numpy as np
import math
import xml.etree.ElementTree as ET
import sys

def main():
	indent_counter = 0
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
		for i in range(len(restrictions)-1,-1,-1):
			print(i)
			if restrictions[i] == '0':
				labeled_data = np.delete(labeled_data,i,axis=1)
				attr = np.delete(attr,i)

	except:
		print("No restrictions file found/inputted")

	#csv_number_labels is null for numeric
	C45(labeled_data, attr, RootNode, classifiers, indent_counter,csv_number_labels, f2)

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