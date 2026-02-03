import argparse

def countTrades():
	lines = 0
	for inputFile in inputFiles:
		tempLines = 0
		with open(inputFile, "r") as file:
			for _ in file:
				tempLines += 1
			print("%s: %d lines" % (inputFile, tempLines))
			lines += tempLines
	print(lines)

parser = argparse.ArgumentParser(description='Count the number of trades in multiple files.')
parser.add_argument('files', nargs="+", help='the files across which to count trades')
args = parser.parse_args()
inputFiles = vars(args)['files']

countTrades()
