import sys
import csv
import numpy as np

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
		print(len(lines[0].rstrip().split(","))-2)
		for i in range(0, len(lines[0].rstrip().split(","))-2):
			attr[lines[0].rstrip().split(",")[i+1]] = i
		data = np.empty((len(lines)-3, 11))
		for i in range(len(lines) - 3):
			data[i] = lines[3+i].rstrip().split(",")[1:]
	
	return data, attr

def main():
	data, attr = read_csv(sys.argv[1])
	print(len(data))
	print(data)
	print(attr)
if __name__ == "__main__":
	main()