import datetime
import dateutil
import numpy as np

def getMicrosecondsFromDate(trade):
	date = trade[1] + "T" + trade[2] + "Z"
	parsedDate = dateutil.parser.isoparse(date)
	return round(datetime.datetime.timestamp(parsedDate) * 1000000)

trades = []
with open('BTC-USD_2024-07-18-23.00.00.000000000_2024-07-22-02.00.00.000000000', 'r') as f:
	for line in f:
		data = line.split(" ")
		data[-1] = data[-1][:-1]
		trades.append(data)

fifteenMinInMicroseconds = 15 * 60 * 1000000
windows = [fifteenMinInMicroseconds * i for i in list(range(1, 17))]
timeWindows = [[0, getMicrosecondsFromDate(trades[0])] for _ in windows]

buyVols = [0] * len(windows)
sellVols = [0] * len(windows)

allDeltas = [[] for _ in windows]
startCollectingDeltas = [False] * len(windows)

# 5th percentile to 95th percentile in increments of 5 percentiles
deltaPercentiles = [[] for _ in windows]

for idx, trade in enumerate(trades):
	vol = float(trade[4])
	microseconds = getMicrosecondsFromDate(trade)

	for j, window in enumerate(windows):
		if microseconds - timeWindows[j][1] > window:
			startCollectingDeltas[j] = True
			i = timeWindows[j][0]
			while i < idx:
				newMicroseconds = getMicrosecondsFromDate(trades[i])
				if microseconds - newMicroseconds > windows[j]:
					newVol = float(trades[i][4])
					if trades[i][5] == 'false':
						buyVols[j] -= newVol
					else:
						sellVols[j] -= newVol
				else:
					break
				i += 1
			timeWindows[j] = [i, getMicrosecondsFromDate(trades[i])]
		if trade[5] == 'false':
			buyVols[j] += vol
		else:
			sellVols[j] += vol
		if startCollectingDeltas[j]:
			allDeltas[j].append(buyVols[j] - sellVols[j])

for idx, arr in enumerate(allDeltas):
	npArr = np.array(arr)
	for percentile in range(5, 100, 5):
		deltaPercentiles[idx].append(np.percentile(npArr, percentile))

deltaPercentiles = [[x.item() for x in sublist] for sublist in deltaPercentiles]

with open("volDeltas", "w") as f:
	for sublist in deltaPercentiles:
		strlist = [str(x) for x in sublist]
		f.write(" ".join(strlist) + '\n')
