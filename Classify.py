import sys
import csv
import numpy as np
import xml.etree.ElementTree as ET

def read_csv(filepath):
	file = open(filepath, 'r')
	lines = file.readlines()
	file.close()
	attr = {}
	if(filepath == "iris.data.txt"):
		attr["sepal_length"] = 0
		attr["sepal_width"] = 1
		attr["petal_length"] = 2
		attr["petal_width"] = 3
		data = np.empty((len(lines), 5))
		for i in range(0, len(lines)):
			temp = lines[i].rstrip().split(",")
			if(temp[4] == "Iris-setosa"):
				temp[4] = 0.0
			elif(temp[4] == "Iris-versicolor"):
				temp[4] = 1.0
			elif(temp[4] == "Iris-virginica"):
				temp[4] = 2.0
			data[i] = temp
	else:		
		for i in range(0, len(lines[0].rstrip().split(","))-2):
			attr[lines[0].rstrip().split(",")[i+1]] = i
		data = np.empty((len(lines)-3, 11))
		for i in range(len(lines) - 3):
			data[i] = lines[3+i].rstrip().split(",")[1:]
	
	return data, attr

def generateTree(rootXML, rootNode):
	for child in rootXML:
		#if it has a node
		if(child.tag == 'node'):
			rootNode.attName = child.attrib['var']
			generateTree(child, rootNode)
		elif(child.tag == 'edge'):
			tempNode = Node("")
			rootNode.edges.append(Edge(child.attrib['num'], tempNode))
			generateTree(child, tempNode)
		elif(child.tag == 'decision'):
			rootNode.leaf = Leaf(child.attrib['end'], child.attrib['choice'], child.attrib['choice'])

def findClass(row, rootNode, myDict, attr):
	if(rootNode.leaf != None):
		if(float(rootNode.leaf.decision) == float(row[-1])):
			myDict["total"] = myDict["total"] + 1
			myDict["right"] = myDict["right"] + 1
		else:
			myDict["total"] = myDict["total"] + 1
			myDict["wrong"] = myDict["wrong"] + 1
	else:
		for i in rootNode.edges:
			if(float(i.choice) == row[attr[rootNode.attName]]):
				findClass(row, i.Node, myDict, attr)



def main():
	data, attr = read_csv(sys.argv[1])
	tree = ET.parse(sys.argv[2])
	root = tree.getroot()
	rootNode = Node("")
	generateTree(root, rootNode)
	answerCollection = {}
	answerCollection["total"] = 0
	answerCollection["wrong"] = 0
	answerCollection["right"] = 0
	for row in data:
		findClass(row, rootNode, answerCollection, attr)
	print(answerCollection["right"])



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

