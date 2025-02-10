import datetime
import dateutil
import numpy as np

THRESHOLD = 0.25

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

priceDiffs = []

'''
increasing = True
pivotPrice = float(trades[0][3])

i = 1
while float(trades[i][3]) == pivotPrice:
	i += 1
if float(trades[i][3]) > pivotPrice:
	increasing = True
else:
	increasing = False

for idx, trade in enumerate(trades):
	if idx == 0:
		continue

	price = float(trade[3])
	prevPrice = float(trades[idx - 1][3])
	if increasing and price < prevPrice:
		priceDiffs.append((prevPrice - pivotPrice) / pivotPrice * 100)
		# print("inc")
		# print(pivotPrice)
		pivotPrice = prevPrice
		# print(prevPrice)
		# print(price)
		increasing = False
	elif not increasing and price > prevPrice:
		priceDiffs.append((pivotPrice - prevPrice) / pivotPrice * 100)
		# print(priceDiffs)
		# print("dec")
		# print(pivotPrice)
		pivotPrice = prevPrice
		# print(prevPrice)
		# print(price)
		increasing = True
'''

pivotPrice = float(trades[0][3])
increasing = None
maxPrice = pivotPrice
minPrice = pivotPrice

for trade in trades:
	price = float(trade[3])
	# print((price - minPrice) / minPrice * 100)
	# print((maxPrice - price) / maxPrice * 100)
	if increasing is not None:
		if increasing:
			maxPrice = max(maxPrice, price)
		else:
			minPrice = min(minPrice, price)
	'''
	if increasing:
		print("inc")
	else:
		print("dec")
	print(maxPrice)
	print(minPrice)
	'''
	if (price - minPrice) / minPrice * 100 >= THRESHOLD:
		if increasing is not None and not increasing:
			priceDiffs.append((pivotPrice - minPrice) / pivotPrice * 100)
			pivotPrice = minPrice
			maxPrice = 1
			increasing = True
		elif increasing is None:
			increasing = True
	elif (maxPrice - price) / maxPrice * 100 >= THRESHOLD:
		if increasing is not None and increasing:
			priceDiffs.append((maxPrice - pivotPrice) / pivotPrice * 100)
			pivotPrice = maxPrice
			minPrice = 10000000
			increasing = False
		elif increasing is None:
			increasing = False

print(len(priceDiffs))
npArr = np.array(priceDiffs)
for percentile in range(5, 100, 5):
	print(np.percentile(npArr, percentile))
