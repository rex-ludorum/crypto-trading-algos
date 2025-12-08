import argparse

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

def adjustTimestamps():
	with open(inputFile, 'r') as inFile, open(inputFile[:-4] + '-adjusted.csv', 'w') as outFile:
		lastTrade = {'Time': '0', 'offset': '0', 'tradeId': '0'}
		first = True
		for line in inFile:
			if first:
				first = False
				outFile.write(line)
				continue
			writeUpdatedLine(line, lastTrade, outFile)

parser = argparse.ArgumentParser(description='Adjust the timestamps of the trading data in a given file.')
parser.add_argument('tradesFile', help='the file with timestamps to adjust')
args = parser.parse_args()
inputFile = vars(args)['tradesFile']

adjustTimestamps()
