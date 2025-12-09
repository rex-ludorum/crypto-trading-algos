import argparse

def writeUpdatedLine(line, outFile):
	data = line.split(",")
	data[6] = data[6].lower()
	data[5] = "{:f}".format(float(data[5]))
	data[4] = "{:.2f}".format(float(data[4]))
	outFile.write(",".join(data))

def convertData():
	with open(inputFile, 'r') as inFile, open(inputFile[:-4] + '-converted.csv', 'w') as outFile:
		first = True
		for line in inFile:
			if first:
				first = False
				outFile.write(line)
				continue
			writeUpdatedLine(line, outFile)

parser = argparse.ArgumentParser(description='Convert booleans to lowercase and floats to fixed point in trading data.')
parser.add_argument('tradesFile', help='the file with data to convert')
args = parser.parse_args()
inputFile = vars(args)['tradesFile']

convertData()
