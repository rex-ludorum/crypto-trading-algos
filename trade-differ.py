import argparse

def findDiff(file1, file2):
	with open(file1, 'r') as f1, open(file2, 'r') as f2:
		ids1 = []
		ids2 = []
		s1 = set()
		s2 = set()
		idsNotIn1 = []
		idsNotIn2 = []

		f1.readline()
		line = f1.readline()
		data = line.split(",")
		minId1 = int(data[3])

		f2.readline()
		line = f2.readline()
		data = line.split(",")
		minId2 = int(data[3])

		while minId1 < minId2:
			line = f1.readline()
			data = line.split(",")
			minId1 = int(data[3])

		while minId2 < minId1:
			line = f2.readline()
			data = line.split(",")
			minId2 = int(data[3])

		ids1.append(minId1)
		ids2.append(minId2)

		lastId1 = 1
		lastId2 = 1
		for line in f1:
			data = line.split(",")
			id = int(data[3])
			ids1.append(id)
			lastId1 = id

		for line in f2:
			data = line.split(",")
			id = int(data[3])
			ids2.append(id)
			lastId2 = id

		ids1 = [x for x in ids1 if x <= min(lastId1, lastId2)]
		ids2 = [x for x in ids2 if x <= min(lastId1, lastId2)]
		s1.update(ids1)
		s2.update(ids2)

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
