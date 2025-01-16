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

allBuyVols = [[] for _ in windows]
allSellVols = [[] for _ in windows]
startCollectingVols = [False] * len(windows)

# 60th percentile to 95th percentile in increments of 5 percentiles
buyVolPercentiles = [[] for _ in windows]
sellVolPercentiles = [[] for _ in windows]

for idx, trade in enumerate(trades):
	vol = float(trade[4])
	microseconds = getMicrosecondsFromDate(trade)

	for j, window in enumerate(windows):
		if microseconds - timeWindows[j][1] > window:
			startCollectingVols[j] = True
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
		if startCollectingVols[j]:
			allBuyVols[j].append(buyVols[j])
			allSellVols[j].append(sellVols[j])

for idx, arr in enumerate(allBuyVols):
	npArr = np.array(arr)
	for percentile in range(60, 100, 5):
		buyVolPercentiles[idx].append(np.percentile(npArr, percentile))
for idx, arr in enumerate(allSellVols):
	npArr = np.array(arr)
	for percentile in range(60, 100, 5):
		sellVolPercentiles[idx].append(np.percentile(npArr, percentile))

buyVolPercentiles = [[x.item() for x in sublist] for sublist in buyVolPercentiles]
sellVolPercentiles = [[x.item() for x in sublist] for sublist in sellVolPercentiles]

with open("buyPercentiles", "w") as f:
	for sublist in buyVolPercentiles:
		strlist = [str(x) for x in sublist]
		f.write(" ".join(strlist) + '\n')

with open("sellPercentiles", "w") as f:
	for sublist in sellVolPercentiles:
		strlist = [str(x) for x in sublist]
		f.write(" ".join(strlist) + '\n')
