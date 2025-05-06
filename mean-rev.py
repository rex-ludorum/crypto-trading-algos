import argparse
import datetime
import dateutil
import numpy as np
import math
import matplotlib.pyplot as plt

MICROSECONDS_IN_HOUR = 3600000000
MICROSECONDS_IN_DAY = 86400000000
MICROSECONDS_IN_WEEK = 604800000000
ONE_HOUR_BEFORE_CME_CLOSE = 75600000000
CME_CLOSE = 79200000000
CME_OPEN = 82800000000
CME_CLOSE_FRIDAY = 165600000000
CME_OPEN_SUNDAY = 342000000000
MARCH_1_1972_IN_SECONDS = 68256000

DAYS_IN_EACH_MONTH = [31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28]

parser = argparse.ArgumentParser(description='Analyze trading data for mean reversion.')
parser.add_argument('files', nargs="+", help="the files that contain trading data")
args = parser.parse_args()
files = vars(args)['files']

def getMicrosecondsFromDate(date):
	parsedDate = dateutil.parser.isoparse(date)
	return round(datetime.datetime.timestamp(parsedDate) * 1000000)

def isLastFridayOfMonth(ts):
	timestamp = ts // 1000000
	leapYearCycles = (timestamp - MARCH_1_1972_IN_SECONDS) // ((365 * 4 + 1) * 86400)
	days = (timestamp - MARCH_1_1972_IN_SECONDS) // 86400
	daysInCurrentCycle = days % (365 * 4 + 1)
	yearsInCurrentCycle = daysInCurrentCycle // 365
	daysInCurrentYear = daysInCurrentCycle % 365

	marchFirstDayOfWeekInCurrentCycle = leapYearCycles * (365 * 4 + 1) % 7
	marchFirstDayOfWeekInCurrentYear = (marchFirstDayOfWeekInCurrentCycle + yearsInCurrentCycle * 365) % 7

	daysUntilFirstFriday = 0;
	if marchFirstDayOfWeekInCurrentYear > 2:
		daysUntilFirstFriday = 9 - marchFirstDayOfWeekInCurrentYear
	else:
		daysUntilFirstFriday = 2 - marchFirstDayOfWeekInCurrentYear
	
	if yearsInCurrentCycle == 4 and marchFirstDayOfWeekInCurrentCycle == 5:
		return True

	if (daysInCurrentYear - daysUntilFirstFriday) % 7 != 0:
		return False

	i = 0
	while i < len(DAYS_IN_EACH_MONTH):
		if daysInCurrentYear >= DAYS_IN_EACH_MONTH[i]:
			daysInCurrentYear -= DAYS_IN_EACH_MONTH[i]
			i += 1
		else:
			break

	if DAYS_IN_EACH_MONTH[i] - daysInCurrentYear - 1 >= 7:
		return False
	else:
		return True

def isDST(ts):
	timestamp = ts // 1000000
	leapYearCycles = (timestamp - MARCH_1_1972_IN_SECONDS) // ((365 * 4 + 1) * 86400)
	days = (timestamp - MARCH_1_1972_IN_SECONDS) // 86400
	daysInCurrentCycle = days % (365 * 4 + 1)
	yearsInCurrentCycle = daysInCurrentCycle // 365
	daysInCurrentYear = daysInCurrentCycle % 365

	timeInCurrentDay = timestamp % 86400

	marchFirstDayOfWeekInCurrentCycle = leapYearCycles * (365 * 4 + 1) % 7
	marchFirstDayOfWeekInCurrentYear = (marchFirstDayOfWeekInCurrentCycle + yearsInCurrentCycle * 365) % 7

	dstStart = 0;
	if marchFirstDayOfWeekInCurrentYear > 4:
		dstStart = 11 - marchFirstDayOfWeekInCurrentYear + 7
	else:
		dstStart = 4 - marchFirstDayOfWeekInCurrentYear + 7

	dstEnd = dstStart + 238

	if daysInCurrentYear == dstStart:
		return timeInCurrentDay >= 7200
	elif daysInCurrentYear == dstEnd:
		return timeInCurrentDay < 7200
	else:
		return daysInCurrentYear > dstStart and daysInCurrentYear < dstEnd

day = 0
beforeClosePrices = []
beforeCloseTimes = []
afterClosePrices = []
afterCloseTimes = []
minuteIncrements = list(range(60000000, MICROSECONDS_IN_HOUR + 1, 60000000))

allBeforeCloseInterps = []
allAfterCloseInterps = []
allExpiryBeforeCloseInterps = []
allExpiryAfterCloseInterps = []

for file in files:
	with open(file, "r") as f:
		for line in f:
			data = line.split(" ")
			date = data[1] + "T" + data[2] + "Z"
			microseconds = getMicrosecondsFromDate(date)

			dst = isDST(microseconds)
			dayRemainder = microseconds % MICROSECONDS_IN_DAY + dst * MICROSECONDS_IN_HOUR
			weekRemainder = microseconds % MICROSECONDS_IN_WEEK + dst * MICROSECONDS_IN_HOUR
			if dayRemainder < ONE_HOUR_BEFORE_CME_CLOSE or dayRemainder > CME_OPEN:
				if beforeClosePrices:
					beforeCloseTimes.reverse()
					beforeClosePrices.reverse()
					if isLastFridayOfMonth(microseconds):
						allExpiryBeforeCloseInterps.append(np.interp(minuteIncrements, beforeCloseTimes, beforeClosePrices).tolist())
						allExpiryAfterCloseInterps.append(np.interp(minuteIncrements, afterCloseTimes, afterClosePrices).tolist())
					else:
						allBeforeCloseInterps.append(np.interp(minuteIncrements, beforeCloseTimes, beforeClosePrices).tolist())
						allAfterCloseInterps.append(np.interp(minuteIncrements, afterCloseTimes, afterClosePrices).tolist())
					beforeCloseTimes.clear()
					beforeClosePrices.clear()
					afterCloseTimes.clear()
					afterClosePrices.clear()
				continue
			if weekRemainder > CME_CLOSE_FRIDAY + MICROSECONDS_IN_HOUR and weekRemainder < CME_OPEN_SUNDAY:
				continue

			if dayRemainder < CME_CLOSE:
				beforeClosePrices.append(float(data[3]))
				beforeCloseTimes.append(CME_CLOSE - dayRemainder)
			elif dayRemainder > CME_CLOSE:
				afterClosePrices.append(float(data[3]))
				afterCloseTimes.append(dayRemainder - CME_CLOSE)

allDiffs = []
allExpiryDiffs = []

atLeastOnePctDiff = 0
for idx, l in enumerate(allBeforeCloseInterps):
	if (max(l) - min(l)) / math.sqrt(min(l) * max(l)) > 0.01:
		atLeastOnePctDiff += 1
		allDiffs.append([abs(x - y) / x * 100 for x, y in zip(l, allAfterCloseInterps[idx])])

atLeastOnePctDiffExpiry = 0
for idx, l in enumerate(allExpiryBeforeCloseInterps):
	print((max(l) - min(l)) / math.sqrt(min(l) * max(l)))
	if (max(l) - min(l)) / math.sqrt(min(l) * max(l)) > 0.01:
		atLeastOnePctDiffExpiry += 1
		allExpiryDiffs.append([abs(x - y) / x * 100 for x, y in zip(l, allExpiryAfterCloseInterps[idx])])

allDiffs = np.array(allDiffs)
allExpiryDiffs = np.array(allExpiryDiffs)

medianDiffs = np.array(np.median(allDiffs, axis=0))
medianExpiryDiffs = np.array(np.median(allExpiryDiffs, axis=0))

print("Days with at least a one percent change: " + str(atLeastOnePctDiff / len(allBeforeCloseInterps)))
print(medianDiffs)
print("Expiry days with at least a one percent change: " + str(atLeastOnePctDiffExpiry / len(allExpiryBeforeCloseInterps)))
print(medianExpiryDiffs)

x = list(range(1, 61))
plt.plot(x, medianDiffs)
plt.xlabel("Minutes after close")
plt.ylabel("% difference from before close")
plt.title("Median mean reversion on non-expiry days with a significant trend leading up to CME close")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x, medianExpiryDiffs)
plt.xlabel("Minutes after close")
plt.ylabel("% difference from before close")
plt.title("Median mean reversion on expiry days with a significant trend leading up to CME close")
plt.grid(True)
plt.show()
