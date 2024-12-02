import datetime
import dateutil
import itertools
import numpy as np

def getMicrosecondsFromDate(trade):
	date = trade[1] + "T" + trade[2]
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
# windows = [fifteenMinInMicroseconds * i for i in list(range(1, 3))]
timeWindows = [[0, getMicrosecondsFromDate(trades[0])] for _ in windows]

stopLosses = [x / 10.0 for x in range(5, 35, 5)]
targets = [x / 10.0 for x in range(10, 55, 5)]
# stopLosses = [x / 10.0 for x in range(10, 15, 5)]
# targets = [x / 10.0 for x in range(15, 20, 5)]

buyVols = [0] * len(windows)
sellVols = [0] * len(windows)

# 60th percentile to 95th percentile in increments of 5 percentiles
buyVolPercentiles = []
sellVolPercentiles = []

with open("buyPercentiles", "r") as f:
	for line in f:
		data = line.split(" ")
		data[-1] = data[-1][:-1]
		buyVolPercentiles.append([float(x) for x in data])

with open("sellPercentiles", "r") as f:
	for line in f:
		data = line.split(" ")
		data[-1] = data[-1][:-1]
		sellVolPercentiles.append([float(x) for x in data])

combos = [list(itertools.product(buyVolPercentiles[i], sellVolPercentiles[i], stopLosses, targets)) for i, _ in enumerate(windows)]
entries = [[[] for _ in combos[i]] for i, _ in enumerate(combos)]
tradeLogs = [[[] for _ in combos[i]] for i, _ in enumerate(combos)]
startingCapitals = [[1] * len(combos[i]) for i, _ in enumerate(combos)]
maxProfits = [[0] * len(combos[i]) for i, _ in enumerate(combos)]
totalTrades = [[0] * len(combos[i]) for i, _ in enumerate(combos)]
wins = [[0] * len(combos[i]) for i, _ in enumerate(combos)]
losses = [[0] * len(combos[i]) for i, _ in enumerate(combos)]

for idx, trade in enumerate(trades):
	vol = float(trade[4])
	price = float(trade[3])
	microseconds = getMicrosecondsFromDate(trade)

	for j, window in enumerate(windows):
		if microseconds - timeWindows[j][1] > window:
			for i in range(timeWindows[j][0], idx):
				newMicroseconds = getMicrosecondsFromDate(trades[i])
				if microseconds - newMicroseconds > windows[j]:
					newVol = float(trades[i][4])
					if trades[i][5] == 'false':
						buyVols[j] -= newVol
					else:
						sellVols[j] -= newVol
				else:
					timeWindows[j] = [i, newMicroseconds]
					break
		if trade[5] == 'false':
			buyVols[j] += vol
		else:
			sellVols[j] += vol

		for i, combo in enumerate(combos[j]):
			buyVolPercentile = combo[0]
			sellVolPercentile = combo[1]
			stopLoss = combo[2]
			target = combo[3]

			if not entries[j][i]:
				if buyVols[j] >= buyVolPercentile:
					maxProfits[j][i] = price
					entries[j][i] = ["LONG", price, trade[1] + "T" + trade[2], trade[0]]
					# print(entries[j])
					tradeLogs[j][i].append(" ".join([str(x) for x in entries[j][i]]))
					totalTrades[j][i] += 1
				elif sellVols[j] >= sellVolPercentile:
					maxProfits[j][i] = price
					entries[j][i] = ["SHORT", price, trade[1] + "T" + trade[2], trade[0]]
					# print(entries[j])
					tradeLogs[j][i].append(" ".join([str(x) for x in entries[j][i]]))
					totalTrades[j][i] += 1
			else:
				if entries[j][i][0] == "LONG":
					maxProfits[j][i] = max(maxProfits[j][i], price)
					profitMargin = (maxProfits[j][i] - entries[j][i][1]) / entries[j][i][1]
					if profitMargin >= target / 100:
						startingCapitals[j][i] *= (1 + profitMargin)
						entries[j][i] = []
						tradeLogs[j][i].append("Profit: " + str(price) + " " + trade[1] + "T" + trade[2] + " " + trade[0])
						tradeLogs[j][i].append("Capital: " + str(startingCapitals[j][i]))
						# print("Profit: " + str(price) + " " + trade[1] + "T" + trade[2])
						# print("Capital: " + str(startingCapitals[j]))
						wins[j][i] += 1
					elif price <= (1 - stopLoss / 100) * entries[j][i][1]:
						startingCapitals[j][i] *= 1 - stopLoss / 100
						entries[j][i] = []
						tradeLogs[j][i].append("Loss: " + str(price) + " " + trade[1] + "T" + trade[2] + " " + trade[0])
						tradeLogs[j][i].append("Capital: " + str(startingCapitals[j][i]))
						# print("Loss: " + str(price) + " " + trade[1] + "T" + trade[2])
						# print("Capital: " + str(startingCapitals[j]))
						losses[j][i] += 1
				elif entries[j][i][0] == "SHORT":
					maxProfits[j][i] = min(maxProfits[j][i], price)
					profitMargin = (entries[j][i][1] - maxProfits[j][i]) / entries[j][i][1]
					if profitMargin >= target / 100:
						startingCapitals[j][i] *= (1 + profitMargin)
						entries[j][i] = []
						tradeLogs[j][i].append("Profit: " + str(price) + " " + trade[1] + "T" + trade[2] + " " + trade[0])
						tradeLogs[j][i].append("Capital: " + str(startingCapitals[j][i]))
						# print("Profit: " + str(price) + " " + trade[1] + "T" + trade[2])
						# print("Capital: " + str(startingCapitals[j]))
						wins[j][i] += 1
					elif price >= (1 + stopLoss / 100) * entries[j][i][1]:
						startingCapitals[j][i] *= 1 - stopLoss / 100
						entries[j][i] = []
						tradeLogs[j][i].append("Loss: " + str(price) + " " + trade[1] + "T" + trade[2] + " " + trade[0])
						tradeLogs[j][i].append("Capital: " + str(startingCapitals[j][i]))
						# print("Loss: " + str(price) + " " + trade[1] + "T" + trade[2])
						# print("Capital: " + str(startingCapitals[j]))
						losses[j][i] += 1

with open("results", "w") as f:
	for i, comboList in enumerate(combos):
		for j, combo in enumerate(comboList):
			print("Time window: " + str(windows[i]))
			f.write("Time window: " + str(windows[i]) + "\n")
			print("Buy vol threshold: " + str(combo[0]))
			f.write("Buy vol threshold: " + str(combo[0]) + "\n")
			print("Sell vol threshold: " + str(combo[1]))
			f.write("Sell vol threshold: " + str(combo[1]) + "\n")
			print("Stop loss: " + str(combo[2]))
			f.write("Stop loss: " + str(combo[2]) + "\n")
			print("Target: " + str(combo[3]))
			f.write("Target: " + str(combo[3]) + "\n")
			print("Total trades: " + str(totalTrades[i][j]))
			f.write("Total trades: " + str(totalTrades[i][j]) + "\n")
			print("Wins: " + str(wins[i][j]))
			f.write("Wins: " + str(wins[i][j]) + "\n")
			print("Losses: " + str(losses[i][j]))
			f.write("Losses: " + str(losses[i][j]) + "\n")
			print("Final capital: " + str(startingCapitals[i][j]))
			f.write("Final capital: " + str(startingCapitals[i][j]) + "\n")
			print("")
			for entry in tradeLogs[i][j]:
				f.write(entry + "\n")
			f.write("\n")

	returns = [x for xs in startingCapitals for x in xs]
	maxIdx = np.array(returns).argmax().item()
	row = 0
	col = 0
	while len(startingCapitals[row]) == 0:
		row += 1
	while (maxIdx > 0):
		if len(startingCapitals[row]) == 0:
			row += 1
			col = 0
		elif maxIdx == len(startingCapitals[row]):
			row += 1
			col = 0
			while len(startingCapitals[row]) == 0:
				row += 1
			break
		else:
			col += min(maxIdx, len(startingCapitals[row]))
			maxIdx -= min(maxIdx, len(startingCapitals[row]))
			if maxIdx != 0:
				row += 1
				col = 0

	print("Maximum return")
	f.write("Maximum return\n")
	print("Time window: " + str(windows[row]))
	f.write("Time window: " + str(windows[row]) + "\n")
	print("Buy vol threshold: " + str(combos[row][col][0]))
	f.write("Buy vol threshold: " + str(combos[row][col][0]) + "\n")
	print("Sell vol threshold: " + str(combos[row][col][1]))
	f.write("Sell vol threshold: " + str(combos[row][col][1]) + "\n")
	print("Stop loss: " + str(combos[row][col][2]))
	f.write("Stop loss: " + str(combos[row][col][2]) + "\n")
	print("Target: " + str(combos[row][col][3]))
	f.write("Target: " + str(combos[row][col][3]) + "\n")
	print("Total trades: " + str(totalTrades[row][col]))
	f.write("Total trades: " + str(totalTrades[row][col]) + "\n")
	print("Wins: " + str(wins[row][col]))
	f.write("Wins: " + str(wins[row][col]) + "\n")
	print("Losses: " + str(losses[row][col]))
	f.write("Losses: " + str(losses[row][col]) + "\n")
	print("Final capital: " + str(startingCapitals[row][col]))
	f.write("Final capital: " + str(startingCapitals[row][col]) + "\n")
	for entry in tradeLogs[row][col]:
		f.write(entry + "\n")
