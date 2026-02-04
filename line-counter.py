import argparse

def countTrades():
	lines = 0
	for inputFile in inputFiles:
		tempLines = 0
		with open(inputFile, "r") as file:
			if inputFile == inputFiles[0]:
				file.readline()
				line = file.readline()
				data = line.split(",")
				firstId = int(data[3])
				file.seek(0, 0)
			elif inputFile == inputFiles[-1]:
				file.seek(0, 2)
				pos = file.tell()
				newlines = 0
				while newlines < 3:
					pos -= 1
					file.seek(pos)
					if file.read(1) == "\n":
						newlines += 1

				file.seek(pos + 1)
				line = file.readline()
				data = line.split(",")
				lastId = int(data[3])

				file.seek(0, 0)

			for line in file:
				tempLines += 1
			tempLines -= 1
			print("%s: %d trades" % (inputFile, tempLines))
			lines += tempLines
	print(lines)
	print("Theoretical number of trades: %d" % (lastId - firstId + 1))
	print("Missed trades: %d" % (lastId - firstId + 1 - lines))

parser = argparse.ArgumentParser(description='Count the number of trades in multiple files.')
parser.add_argument('files', nargs="+", help='the files across which to count trades')
args = parser.parse_args()
inputFiles = vars(args)['files']

countTrades()
