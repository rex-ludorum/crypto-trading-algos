import argparse

def checkGaps():
	with open(inputFile, 'r') as f:
		ids = []
		for line in f:
			data = line.split(" ")
			ids.append(int(data[0]))

		missedRanges = getMissedRanges(ids)
		print(missedRanges[0])
		print(missedRanges[1])

def getMissedRanges(ids):
	if len(ids) == 1:
		return []
	if not ids:
		return []

	ranges = []
	diffs = []
	for idx, trade in enumerate(ids):
		if idx == 0:
			continue
		prevId = ids[idx - 1]
		if trade - 1 != prevId:
			if trade - prevId > 2:
				ranges.append(str(prevId + 1) + "-" + str(trade - 1))
			else:
				ranges.append(str(prevId + 1))
			diffs.append(trade - prevId - 1)

	return (ranges, diffs)

parser = argparse.ArgumentParser(description='Show the gaps in trading data collected from Coinbase.')
parser.add_argument('file', help='the file to analyze for gaps')
args = parser.parse_args()
inputFile = vars(args)['file']

checkGaps()
