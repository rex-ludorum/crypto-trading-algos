import argparse

def checkGaps():
	with open(inputFile, 'r') as f:
		ids = []
		s = set()
		for line in f:
			if '.csv' in inputFile and 'time' in line:
				continue

			if '.csv' in inputFile:
				data = line.split(",")
				id = int(data[3])
			else:
				data = line.split(" ")
				id = int(data[0])

			if id in s:
				ids.append(int(data[0]))
			else:
				s.add(int(data[0]))

		missedRanges = getMissedRanges(ids)
		print(missedRanges)

def getMissedRanges(ids):
	if len(ids) == 1:
		return [str(ids[0])]
	if not ids:
		return []

	ranges = []
	lastContiguousId = ids[0]
	for idx, trade in enumerate(ids):
		if idx == 0:
			continue
		prevId = ids[idx - 1]
		if trade - 1 != prevId:
			if lastContiguousId == prevId:
				ranges.append(str(lastContiguousId))
			else:
				ranges.append(str(lastContiguousId) + "-" + str(prevId))
			lastContiguousId = trade
		if idx == len(ids) - 1:
			if lastContiguousId == trade:
				ranges.append(str(lastContiguousId))
			else:
				ranges.append(str(lastContiguousId) + "-" + str(trade))
	return ranges

parser = argparse.ArgumentParser(description='Show the duplicates in trading data collected from Coinbase.')
parser.add_argument('file', help='the file to analyze for duplicates')
args = parser.parse_args()
inputFile = vars(args)['file']

checkGaps()
