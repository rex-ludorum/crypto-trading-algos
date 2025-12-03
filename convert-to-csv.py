import datetime
import dateutil
import argparse

def getMicrosecondsFromDate(trade):
	date = trade[1] + "T" + trade[2] + "Z"
	parsedDate = dateutil.parser.isoparse(date)
	return round(datetime.datetime.timestamp(parsedDate) * 1000000)

def convert():
	outputFile = inputFile + ".csv"
	with open(inputFile, "r") as readFile, open(outputFile, "w") as writeFile:
		writeFile.write(",".join(["time", "measure_name", "symbol", "tradeId", "price", "size", "isBuyerMaker"]) + '\n')
		for line in readFile:
			data = line.split(" ")
			data[-1] = data[-1][:-1]
			writeFile.write(",".join([str(getMicrosecondsFromDate(data)), "price", symbol, data[0], data[3], data[4], data[5]]) + '\n')

parser = argparse.ArgumentParser(description='Convert a file with trading data to a csv file.')
parser.add_argument('file', help='the file with trading data to convert', type=str)
args = parser.parse_args()
inputFile = vars(args)['file']
if "ETH-USD" in inputFile:
	symbol = "ETH-USD"
else:
	symbol = "BTC-USD"

convert()
