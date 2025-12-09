import argparse

def findDiff(file1, file2):
	with open(file1, 'r') as f1, open(file2, 'r') as f2:
		ids1 = []
		ids2 = []
		s1 = set()
		s2 = set()
		idsNotIn1 = []
		idsNotIn2 = []
		for line in f1:
			if 'time' in line:
				continue

			data = line.split(",")
			id = int(data[3])
			ids1.append(id)
			s1.add(id)

		for line in f2:
			if 'time' in line:
				continue

			data = line.split(",")
			id = int(data[3])
			ids2.append(id)
			s2.add(id)

		for id in ids1:
			if id not in s2:
				idsNotIn2.append(id)

		for id in ids2:
			if id not in s1:
				idsNotIn1.append(id)

		print("Trades in %s not in %s: " % (file1, file2) + str(getMissedRanges(idsNotIn2)))
		print("Trades in %s not in %s: " % (file2, file1) + str(getMissedRanges(idsNotIn1)))

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

parser = argparse.ArgumentParser(description='Show how two files differ in the trades they contain.')
parser.add_argument('file1', help='the first file to compare')
parser.add_argument('file2', help='the second file to compare')
args = parser.parse_args()
file1 = vars(args)['file1']
file2 = vars(args)['file2']

findDiff(file1, file2)
