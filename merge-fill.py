import argparse

def getTimestamp(line):
	data = line.split(",")
	return data[0]

def getId(line):
	data = line.split(",")
	return data[3]

def writeUpdatedLine(line, lastTrade, outFile):
	data = line.split(",")
	recordTime = data[0]
	if lastTrade and int(recordTime) <= int(lastTrade['Time']) + int(lastTrade['offset']) + int(data[3]) - int(lastTrade['tradeId']):
		data[0] = str(int(lastTrade['Time']) + int(lastTrade['offset']) + int(data[3]) - int(lastTrade['tradeId']))
		# print("Time %s for %s conflicts with last trade time (%s with offset %s, tradeId %s), updating to %s" % (recordTime, symbol, lastTrade['Time'], lastTrade['offset'], lastTrade['tradeId'], data[0]))
		lastTrade['offset'] = str(int(lastTrade['offset']) + int(data[3]) - int(lastTrade['tradeId']))
	else:
		lastTrade['Time'] = recordTime
		lastTrade['offset'] = str(0)
	lastTrade['tradeId'] = data[3]
	outFile.write(",".join(data))

def fillGaps():
	with open(inputFile, 'r') as inFile, open(fills, 'r') as fillFile, open(inputFile[:-4] + '-filled.csv', 'w') as outFile:
		lines = inFile.readlines()
		outFile.write(lines.pop(0))
		i = 0
		lastTrade = {'Time': '0', 'offset': '0', 'tradeId': '0'}
		first = True
		for line in fillFile:
			if first:
				first = False
				continue
			while int(getId(lines[i])) < int(getId(line)):
				writeUpdatedLine(lines[i], lastTrade, outFile)
				i += 1
			writeUpdatedLine(line, lastTrade, outFile)
		for idx in range(i, len(lines)):
			writeUpdatedLine(lines[idx], lastTrade, outFile)

parser = argparse.ArgumentParser(description='Fill in gaps in trading data from a given file.')
parser.add_argument('tradesFile', help='the file with gaps to fill in')
parser.add_argument('fillsFile', help='the file with trades to fill with')
args = parser.parse_args()
inputFile = vars(args)['tradesFile']
fills = vars(args)['fillsFile']

fillGaps()
