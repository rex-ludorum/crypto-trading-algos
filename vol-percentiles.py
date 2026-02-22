import argparse

fifteenMinInMicroseconds = 15 * 60 * 1000000
windows = [fifteenMinInMicroseconds * i for i in list(range(1, 5))]
timeWindows = [[0, 0] for _ in windows]

buyVols = [0] * len(windows)
sellVols = [0] * len(windows)

trades = []

def calculateIndicators():
	tradeIdx = 0
	first = True
	for inputFile in inputFiles:
		with open(inputFile, "r") as file:
			file.readline()
			for line in file:
				data = line.split(",")
				trades.append(data)
				vol = float(data[5]) * float(data[4])
				timestamp = int(data[0])
				if first:
					for window in timeWindows:
						window[1] = timestamp
					first = False
				isBuyerMaker = data[6].strip().lower() == 'true'
				tradeIdx += 1
				for j, window in enumerate(windows):
					if timestamp - timeWindows[j][1] > window:
						i = timeWindows[j][0]
						while i < tradeIdx - 1:
							newMicroseconds = int(trades[i][0])
							if timestamp - newMicroseconds > windows[j]:
								newVol = float(trades[i][5]) * float(trades[i][4])
								if trades[i][6].strip().lower() == 'true':
									sellVols[j] -= newVol
								else:
									buyVols[j] -= newVol
							else:
								break
							i += 1
						timeWindows[j] = [i, int(trades[i][0])]
					if isBuyerMaker:
						sellVols[j] += vol
					else:
						buyVols[j] += vol
				if tradeIdx % 100000 == 0:
					print("Timestamp: %d" % timestamp)
					print("15 minute buy volume: %f" % buyVols[0])
					print("15 minute sell volume: %f" % sellVols[0])
					print("30 minute buy volume: %f" % buyVols[1])
					print("30 minute sell volume: %f" % sellVols[1])
					print("45 minute buy volume: %f" % buyVols[2])
					print("45 minute sell volume: %f" % sellVols[2])
					print("60 minute buy volume: %f" % buyVols[3])
					print("60 minute sell volume: %f" % sellVols[3])

parser = argparse.ArgumentParser(description='Show cumulative volume calculations for sanity checking.')
parser.add_argument('files', nargs="+", help='the files across which to calculate')
args = parser.parse_args()
inputFiles = vars(args)['files']

calculateIndicators()
