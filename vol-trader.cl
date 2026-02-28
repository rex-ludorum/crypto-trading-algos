#define MAX_TOTAL_TRADES 20618

#define NUM_WINDOWS 4

#define FIFTEEN_MINUTES_MICROSECONDS 900000000

typedef struct __attribute__ ((packed)) tradeWithoutDate {
	long timestamp;
	double price;
	double qty;
	uchar isBuyerMaker;
} tradeWithoutDate;

typedef struct __attribute__ ((packed)) indicators {
	double vols[2 * NUM_WINDOWS];
} indicators;

typedef struct __attribute__ ((packed)) combo {
	long window;
	double buyVolPercentile;
	double sellVolPercentile;
	double stopLoss;
	double target;
} combo;

typedef struct __attribute__ ((packed)) entry {
	double price;
	bool isLong;
} entry;

typedef struct __attribute__ ((packed)) timeWindow {
	int tradeIdx;
	long timestamp;
} timeWindow;

typedef struct __attribute__ ((packed)) tradeRecord {
	double capital;
	int shorts;
	int shortWins;
	int shortLosses;
	int longs;
	int longWins;
	int longLosses;
} tradeRecord;

typedef struct __attribute__ ((packed)) positionData {
	long timestamp;
	double buyVol;
	double sellVol;
	int tradeIdx;
} positionData;

typedef struct __attribute__ ((packed)) twMetadata {
	int twTranslation;
	int twStart;
} twMetadata;

typedef struct __attribute__ ((packed)) entryData {
	double buyVol;
	double sellVol;
} entryData;

typedef struct __attribute__ ((packed)) entryAndExit {
	double profitMargin;
	int entryIndex;
	int exitIndex;
	entryData e;
	bool isLong;
} entryAndExit;

typedef struct __attribute__((packed)) drawdowns {
	double max;
	double mean;
	double current;
} drawdowns;

typedef struct __attribute__((packed)) drawdownLengths {
	long max;
	double mean;
	long drawdownStart;
} drawdownLengths;

typedef struct __attribute__((packed)) lossStreaks {
	int n;
	int max;
	double mean;
	int current;
} lossStreaks;

typedef struct __attribute__((packed)) tradeDurations {
	int n;
	long max;
	double mean;
	long entryTimestamp;
} tradeDurations;

typedef struct __attribute__((packed)) monthlyReturns {
	int n;
	double current;
	double mean;
	double m2;
	long nextMonth;
} monthlyReturns;

#define MICROSECONDS_IN_HOUR 3600000000
#define MICROSECONDS_IN_DAY 86400000000
#define MICROSECONDS_IN_WEEK 604800000000
#define CME_CLOSE 79200000000
#define CME_OPEN 82800000000
#define CME_CLOSE_FRIDAY 165600000000
#define CME_OPEN_SUNDAY 342000000000

#define MARCH_1_1972_IN_SECONDS 68256000
#define DAYS_IN_LEAP_YEAR_CYCLE 1461
#define SECONDS_IN_DAY 86400

__constant int daysInMonths[12] = { 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28 };

inline int isDST(long ts) {
	int timestamp = ts / 1000000;
	int days = (timestamp - MARCH_1_1972_IN_SECONDS) / SECONDS_IN_DAY;
	int daysInCurrentCycle = days % DAYS_IN_LEAP_YEAR_CYCLE;
	int daysInCurrentYear = daysInCurrentCycle % 365;

	int timeInCurrentDay = timestamp % SECONDS_IN_DAY;

	int marchFirstDayOfWeekInCurrentYear = (days - daysInCurrentYear) % 7;

	int dstStart = 0;
	if (marchFirstDayOfWeekInCurrentYear > 4) {
		dstStart = 11 - marchFirstDayOfWeekInCurrentYear + 7;
	} else {
		dstStart = 4 - marchFirstDayOfWeekInCurrentYear + 7;
	}
	int dstEnd = dstStart + 238;

	if (daysInCurrentYear == dstStart) {
		return timeInCurrentDay >= 7200;
	} else if (daysInCurrentYear == dstEnd) {
		return timeInCurrentDay < 7200;
	} else {
		return daysInCurrentYear > dstStart && daysInCurrentYear < dstEnd;
	}
}

inline long getTsOfNextMonth(long ts) {
	int timestamp = ts / 1000000;
	int days = (timestamp - MARCH_1_1972_IN_SECONDS) / SECONDS_IN_DAY;
	int daysInCurrentCycle = days % DAYS_IN_LEAP_YEAR_CYCLE;
	int daysInCurrentYear = daysInCurrentCycle % 365;
	int daysInCurrentYearCopy = daysInCurrentYear;
	bool isLeapYear = daysInCurrentCycle / 365 >= 3;

	int daysUntilNextMonthFromStartOfCurrentYear = 0;
	if (daysInCurrentCycle != 1460) {
		int monthIdx = 0;
		while (daysInCurrentYear >= 0) {
			daysUntilNextMonthFromStartOfCurrentYear += daysInMonths[monthIdx];
			daysInCurrentYear -= daysInMonths[monthIdx++];
		}
		if (isLeapYear && monthIdx == 12)
			daysUntilNextMonthFromStartOfCurrentYear += 1;
	} else
		daysUntilNextMonthFromStartOfCurrentYear = 1;

	long long daysUntilNextMonthFromMarchOne =
			days - daysInCurrentYearCopy + daysUntilNextMonthFromStartOfCurrentYear;
	return (daysUntilNextMonthFromMarchOne * SECONDS_IN_DAY +
					MARCH_1_1972_IN_SECONDS) *
				 1000000;
}

inline int computeIndicatorIdx(long window) {
	return window / FIFTEEN_MINUTES_MICROSECONDS - 1;
}

__kernel void volTraderWithIndicators(__global int* numTrades, __global tradeWithoutDate* trades, __global indicators* inds, __global combo* combos, __global entry* entries, __global tradeRecord* tradeRecords, __global int* numTradesInInterval) {
	int index = get_global_id(0);
	double capital = tradeRecords[index].capital;
	combo c = combos[index];
	// printf("%d %d %d %d %d\n", sizeof(int), sizeof(double), sizeof(long), sizeof(bool), sizeof(long long));
	// printf("%d %f %f %f %f\n", c.window, c.buyVolPercentile, c.sellVolPercentile, c.stopLoss, c.target);
	// printf("%d\n", numTrades);
	int ls = tradeRecords[index].longs;
	int lw = tradeRecords[index].longWins;
	int ll = tradeRecords[index].longLosses;
	int ss = tradeRecords[index].shorts;
	int sw = tradeRecords[index].shortWins;
	int sl = tradeRecords[index].shortLosses;
	entry e = entries[index];

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

	int tradesInInterval = 0;

	int indicatorIdx = computeIndicatorIdx(c.window);

	for (int i = 0; i < *numTrades; i++) {
		double vol = trades[i].qty;
		double price = trades[i].price;
		double sellVol = inds[i].vols[2 * indicatorIdx];
		double buyVol = inds[i].vols[2 * indicatorIdx + 1];
		// printf("%e %f %lld %d %d\n", vol, price, trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		// printf("%ld %d %d\n", trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		long microseconds = trades[i].timestamp;
		int isDSTInt = isDST(microseconds);
		long dayRemainder = microseconds % MICROSECONDS_IN_DAY + isDSTInt * MICROSECONDS_IN_HOUR;
		long weekRemainder = microseconds % MICROSECONDS_IN_WEEK + isDSTInt * MICROSECONDS_IN_HOUR;
		bool inClose = dayRemainder >= CME_CLOSE && dayRemainder < CME_OPEN;
		bool onWeekend = weekRemainder >= CME_CLOSE_FRIDAY && weekRemainder < CME_OPEN_SUNDAY;

		if (e.price != 0.0) {
			double profitMargin;
			if (e.isLong) {
				profitMargin = price / e.price;
				if (inClose || profitMargin >= precomputedTarget || profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					e = (entry) {0.0, false};
					lw += profitMargin >= 1.0;
					ll += profitMargin < 1.0;
					tradesInInterval++;
				}
			} else {
				profitMargin = 2 - price / e.price;
				if (inClose || profitMargin >= precomputedTarget || profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					e = (entry) {0.0, false};
					sw += profitMargin >= 1.0;
					sl += profitMargin < 1.0;
					tradesInInterval++;
				}
			}
		}

		if (e.price == 0.0) {
			if (buyVol >= c.buyVolPercentile && !inClose && !onWeekend) {
				e = (entry) {price, true};
				ls += 1;
			} else if (sellVol >= c.sellVolPercentile && !inClose && !onWeekend) {
				e = (entry) {price, false};
				ss += 1;
			}
		}
	}

	entries[index] = e;
	tradeRecords[index] = (tradeRecord) {capital, ss, sw, sl, ls, lw, ll};
	numTradesInInterval[index] = tradesInInterval;
}

__kernel void volTrader(__global int* numTrades, __global tradeWithoutDate* trades, __global combo* combos, __global entry* entries, __global tradeRecord* tradeRecords, __global positionData* positionDatas, __global twMetadata* twBetweenRuns, __global int* numTradesInInterval) {
	int index = get_global_id(0);
	double capital = tradeRecords[index].capital;
	combo c = combos[index];
	// printf("%d %d %d %d %d\n", sizeof(int), sizeof(double), sizeof(long), sizeof(bool), sizeof(long long));
	// printf("%d %f %f %f %f\n", c.window, c.buyVolPercentile, c.sellVolPercentile, c.stopLoss, c.target);
	// printf("%d\n", numTrades);
	int ls = tradeRecords[index].longs;
	int lw = tradeRecords[index].longWins;
	int ll = tradeRecords[index].longLosses;
	int ss = tradeRecords[index].shorts;
	int sw = tradeRecords[index].shortWins;
	int sl = tradeRecords[index].shortLosses;
	timeWindow tw = {positionDatas[index].tradeIdx - twBetweenRuns->twTranslation, positionDatas[index].timestamp};
	double buyVol = positionDatas[index].buyVol;
	double sellVol = positionDatas[index].sellVol;
	entry e = entries[index];

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

	int tradesInInterval = 0;

	for (int i = twBetweenRuns->twStart; i < *numTrades; i++) {
		double vol = trades[i].qty;
		double price = trades[i].price;
		// printf("%e %f %lld %d %d\n", vol, price, trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		// printf("%ld %d %d\n", trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		long microseconds = trades[i].timestamp;
		int isDSTInt = isDST(microseconds);
		long dayRemainder = microseconds % MICROSECONDS_IN_DAY + isDSTInt * MICROSECONDS_IN_HOUR;
		long weekRemainder = microseconds % MICROSECONDS_IN_WEEK + isDSTInt * MICROSECONDS_IN_HOUR;
		bool inClose = dayRemainder >= CME_CLOSE && dayRemainder < CME_OPEN;
		bool onWeekend = weekRemainder >= CME_CLOSE_FRIDAY && weekRemainder < CME_OPEN_SUNDAY;

		if (e.price != 0.0) {
			double profitMargin;
			if (e.isLong) {
				profitMargin = price / e.price;
				if (inClose || profitMargin >= precomputedTarget || profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					e = (entry) {0.0, false};
					lw += profitMargin >= 1.0;
					ll += profitMargin < 1.0;
					tradesInInterval++;
				}
			} else {
				profitMargin = 2 - price / e.price;
				if (inClose || profitMargin >= precomputedTarget || profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					e = (entry) {0.0, false};
					sw += profitMargin >= 1.0;
					sl += profitMargin < 1.0;
					tradesInInterval++;
				}
			}
		}

		if (microseconds - tw.timestamp > c.window) {
			int k = tw.tradeIdx;
			long newMicroseconds;
			while (k < i) {
				newMicroseconds = trades[k].timestamp;
				if (microseconds - newMicroseconds > c.window) {
					double newVol = trades[k].qty * trades[k].price;
					if (trades[k].isBuyerMaker) sellVol -= newVol;
					else buyVol -= newVol;
				} else {
					break;
				}
				k++;
			}
			tw = (timeWindow) {k, newMicroseconds};
		}
		if (trades[i].isBuyerMaker) sellVol += vol * price;
		else buyVol += vol * price;

		if (e.price == 0.0) {
			if (buyVol >= c.buyVolPercentile && !inClose && !onWeekend) {
				e = (entry) {price, true};
				ls += 1;
			} else if (sellVol >= c.sellVolPercentile && !inClose && !onWeekend) {
				e = (entry) {price, false};
				ss += 1;
			}
		}
	}

	entries[index] = e;
	tradeRecords[index] = (tradeRecord) {capital, ss, sw, sl, ls, lw, ll};
	positionDatas[index] = (positionData) {tw.timestamp, buyVol, sellVol, tw.tradeIdx};
	numTradesInInterval[index] = tradesInInterval;
}

__kernel void volTraderWithTrades(__global int* numTrades, __global tradeWithoutDate* trades, __global combo* combos, __global entry* entries, __global tradeRecord* tradeRecords, __global positionData* positionDatas, __global twMetadata* twBetweenRuns, __global entryAndExit* entriesAndExits) {
	int index = get_global_id(0);
	double capital = tradeRecords[index].capital;
	combo c = combos[index];
	// printf("%d %d %d %d %d\n", sizeof(int), sizeof(double), sizeof(long), sizeof(bool), sizeof(long long));
	// printf("%d %f %f %f %f\n", c.window, c.buyVolPercentile, c.sellVolPercentile, c.stopLoss, c.target);
	// printf("%d\n", numTrades);
	int ls = tradeRecords[index].longs;
	int lw = tradeRecords[index].longWins;
	int ll = tradeRecords[index].longLosses;
	int ss = tradeRecords[index].shorts;
	int sw = tradeRecords[index].shortWins;
	int sl = tradeRecords[index].shortLosses;
	timeWindow tw = {positionDatas[index].tradeIdx - twBetweenRuns->twTranslation, positionDatas[index].timestamp};
	double buyVol = positionDatas[index].buyVol;
	double sellVol = positionDatas[index].sellVol;
	entry e = entries[index];

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

	int tradesInInterval = 0;

	entryAndExit currentTrade = {0, 0, 0, {0, 0}, false};

	for (int i = twBetweenRuns->twStart; i < *numTrades; i++) {
		double vol = trades[i].qty;
		double price = trades[i].price;
		// printf("%e %f %lld %d %d\n", vol, price, trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		// printf("%ld %d %d\n", trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		long microseconds = trades[i].timestamp;
		int isDSTInt = isDST(microseconds);
		long dayRemainder = microseconds % MICROSECONDS_IN_DAY + isDSTInt * MICROSECONDS_IN_HOUR;
		long weekRemainder = microseconds % MICROSECONDS_IN_WEEK + isDSTInt * MICROSECONDS_IN_HOUR;
		bool inClose = dayRemainder >= CME_CLOSE && dayRemainder < CME_OPEN;
		bool onWeekend = weekRemainder >= CME_CLOSE_FRIDAY && weekRemainder < CME_OPEN_SUNDAY;

		if (e.price != 0.0) {
			double profitMargin;
			if (e.isLong) {
				profitMargin = price / e.price;
				if (inClose || profitMargin >= precomputedTarget || profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					currentTrade.profitMargin = profitMargin;
					e = (entry) {0.0, false};
					lw += profitMargin >= 1.0;
					ll += profitMargin < 1.0;
					currentTrade.exitIndex = i;
					entriesAndExits[index * MAX_TOTAL_TRADES + tradesInInterval++] = currentTrade;
					currentTrade = (entryAndExit) {0, 0, 0, {0, 0}, false};
				}
			} else {
				profitMargin = 2 - price / e.price;
				if (inClose || profitMargin >= precomputedTarget || profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					currentTrade.profitMargin = profitMargin;
					e = (entry) {0.0, false};
					sw += profitMargin >= 1.0;
					sl += profitMargin < 1.0;
					currentTrade.exitIndex = i;
					entriesAndExits[index * MAX_TOTAL_TRADES + tradesInInterval++] = currentTrade;
					currentTrade = (entryAndExit) {0, 0, 0, {0, 0}, false};
				}
			}
		}

		if (microseconds - tw.timestamp > c.window) {
			int k = tw.tradeIdx;
			long newMicroseconds;
			while (k < i) {
				newMicroseconds = trades[k].timestamp;
				if (microseconds - newMicroseconds > c.window) {
					double newVol = trades[k].qty * trades[k].price;
					if (trades[k].isBuyerMaker) sellVol -= newVol;
					else buyVol -= newVol;
				} else {
					break;
				}
				k++;
			}
			tw = (timeWindow) {k, newMicroseconds};
		}
		if (trades[i].isBuyerMaker) sellVol += vol * price;
		else buyVol += vol * price;

		if (e.price == 0.0) {
			if (buyVol >= c.buyVolPercentile && !inClose && !onWeekend) {
				e = (entry) {price, true};
				ls += 1;
				currentTrade.entryIndex = i;
				currentTrade.isLong = true;
				currentTrade.e.buyVol = buyVol;
				currentTrade.e.sellVol = sellVol;
				entriesAndExits[index * MAX_TOTAL_TRADES + tradesInInterval] = currentTrade;
			} else if (sellVol >= c.sellVolPercentile && !inClose && !onWeekend) {
				e = (entry) {price, false};
				ss += 1;
				currentTrade.entryIndex = i;
				currentTrade.e.buyVol = buyVol;
				currentTrade.e.sellVol = sellVol;
				entriesAndExits[index * MAX_TOTAL_TRADES + tradesInInterval] = currentTrade;
			}
		}
	}

	entries[index] = e;
	tradeRecords[index] = (tradeRecord) {capital, ss, sw, sl, ls, lw, ll};
	positionDatas[index] = (positionData) {tw.timestamp, buyVol, sellVol, tw.tradeIdx};
}

__kernel void volTraderWithTradesAndIndicators(__global int* numTrades, __global tradeWithoutDate* trades, __global indicators* inds, __global combo* combos, __global entry* entries, __global tradeRecord* tradeRecords, __global entryAndExit* entriesAndExits) {
	int index = get_global_id(0);
	double capital = tradeRecords[index].capital;
	combo c = combos[index];
	// printf("%d %d %d %d %d\n", sizeof(int), sizeof(double), sizeof(long), sizeof(bool), sizeof(long long));
	// printf("%d %f %f %f %f\n", c.window, c.buyVolPercentile, c.sellVolPercentile, c.stopLoss, c.target);
	// printf("%d\n", numTrades);
	int ls = tradeRecords[index].longs;
	int lw = tradeRecords[index].longWins;
	int ll = tradeRecords[index].longLosses;
	int ss = tradeRecords[index].shorts;
	int sw = tradeRecords[index].shortWins;
	int sl = tradeRecords[index].shortLosses;
	entry e = entries[index];

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

	int indicatorIdx = computeIndicatorIdx(c.window);

	int tradesInInterval = 0;

	entryAndExit currentTrade = {0, 0, 0, {0, 0}, false};

	for (int i = 0; i < *numTrades; i++) {
		double vol = trades[i].qty;
		double price = trades[i].price;
		double sellVol = inds[i].vols[2 * indicatorIdx];
		double buyVol = inds[i].vols[2 * indicatorIdx + 1];
		// printf("%e %f %lld %d %d\n", vol, price, trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		// printf("%ld %d %d\n", trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		long microseconds = trades[i].timestamp;
		int isDSTInt = isDST(microseconds);
		long dayRemainder = microseconds % MICROSECONDS_IN_DAY + isDSTInt * MICROSECONDS_IN_HOUR;
		long weekRemainder = microseconds % MICROSECONDS_IN_WEEK + isDSTInt * MICROSECONDS_IN_HOUR;
		bool inClose = dayRemainder >= CME_CLOSE && dayRemainder < CME_OPEN;
		bool onWeekend = weekRemainder >= CME_CLOSE_FRIDAY && weekRemainder < CME_OPEN_SUNDAY;

		if (e.price != 0.0) {
			// consider consolidating these code paths
			double profitMargin;
			if (e.isLong) {
				profitMargin = price / e.price;
				if (inClose || profitMargin >= precomputedTarget || profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					e = (entry) {0.0, false};
					lw += profitMargin >= 1.0;
					ll += profitMargin < 1.0;
					currentTrade.exitIndex = i;
					entriesAndExits[index * MAX_TOTAL_TRADES + tradesInInterval++] = currentTrade;
					currentTrade = (entryAndExit) {0, 0, 0, {0, 0}, false};
				}
			} else {
				profitMargin = 2 - price / e.price;
				if (inClose || profitMargin >= precomputedTarget || profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					e = (entry) {0.0, false};
					sw += profitMargin >= 1.0;
					sl += profitMargin < 1.0;
					currentTrade.exitIndex = i;
					entriesAndExits[index * MAX_TOTAL_TRADES + tradesInInterval++] = currentTrade;
					currentTrade = (entryAndExit) {0, 0, 0, {0, 0}, false};
				}
			}
		}

		if (e.price == 0.0) {
			if (buyVol >= c.buyVolPercentile && !inClose && !onWeekend) {
				e = (entry) {price, true};
				ls += 1;
				currentTrade.entryIndex = i;
				currentTrade.isLong = true;
				currentTrade.e.buyVol = buyVol;
				currentTrade.e.sellVol = sellVol;
				entriesAndExits[index * MAX_TOTAL_TRADES + tradesInInterval] = currentTrade;
			} else if (sellVol >= c.sellVolPercentile && !inClose && !onWeekend) {
				e = (entry) {price, false};
				ss += 1;
				currentTrade.entryIndex = i;
				currentTrade.e.buyVol = buyVol;
				currentTrade.e.sellVol = sellVol;
				entriesAndExits[index * MAX_TOTAL_TRADES + tradesInInterval] = currentTrade;
			}
		}
	}

	entries[index] = e;
	tradeRecords[index] = (tradeRecord) {capital, ss, sw, sl, ls, lw, ll};
}

__kernel void volTraderWithOnlineAlgs(__global int* numTrades, __global tradeWithoutDate* trades, __global indicators* inds, __global combo* combos, __global entry* entries, __global tradeRecord* tradeRecords, __global drawdowns* drawdowns, __global drawdownLengths* drawdownLengths, __global lossStreaks* lossStreaks, __global tradeDurations* tradeDurations, __global monthlyReturns* monthlyReturns) {
	int index = get_global_id(0);
	double capital = tradeRecords[index].capital;
	combo c = combos[index];
	// printf("%d %d %d %d %d\n", sizeof(int), sizeof(double), sizeof(long), sizeof(bool), sizeof(long long));
	// printf("%d %f %f %f %f\n", c.window, c.buyVolPercentile, c.sellVolPercentile, c.stopLoss, c.target);
	// printf("%d\n", numTrades);
	int ls = tradeRecords[index].longs;
	int lw = tradeRecords[index].longWins;
	int ll = tradeRecords[index].longLosses;
	int ss = tradeRecords[index].shorts;
	int sw = tradeRecords[index].shortWins;
	int sl = tradeRecords[index].shortLosses;
	entry e = entries[index];

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

	int tradesInInterval = 0;

	int indicatorIdx = computeIndicatorIdx(c.window);

	for (int i = 0; i < *numTrades; i++) {
		double vol = trades[i].qty;
		double price = trades[i].price;
		double sellVol = inds[i].vols[2 * indicatorIdx];
		double buyVol = inds[i].vols[2 * indicatorIdx + 1];
		// printf("%e %f %lld %d %d\n", vol, price, trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		// printf("%ld %d %d\n", trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		long microseconds = trades[i].timestamp;
		int isDSTInt = isDST(microseconds);
		long dayRemainder = microseconds % MICROSECONDS_IN_DAY + isDSTInt * MICROSECONDS_IN_HOUR;
		long weekRemainder = microseconds % MICROSECONDS_IN_WEEK + isDSTInt * MICROSECONDS_IN_HOUR;
		bool inClose = dayRemainder >= CME_CLOSE && dayRemainder < CME_OPEN;
		bool onWeekend = weekRemainder >= CME_CLOSE_FRIDAY && weekRemainder < CME_OPEN_SUNDAY;

		if (e.price != 0.0) {
			double profitMargin;
			if (e.isLong) {
				profitMargin = price / e.price;
				if (inClose || profitMargin >= precomputedTarget || profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					e = (entry) {0.0, false};
					bool win = profitMargin >= 1.0;
					lw += win;
					ll += !win;
					tradesInInterval++;

					long newTradeDuration = microseconds - tradeDurations[index].entryTimestamp;
					tradeDurations[index].max = max(tradeDurations[index].max, newTradeDuration);
					tradeDurations[index].mean += ((double) newTradeDuration - tradeDurations[index].mean) / (double) ++tradeDurations[index].n;

					if (monthlyReturns[index].nextMonth == 0)
						monthlyReturns[index].nextMonth = getTsOfNextMonth(microseconds);

					if (microseconds < monthlyReturns[index].nextMonth) {
						monthlyReturns[index].current *= profitMargin;
					} else {
						monthlyReturns[index].nextMonth = getTsOfNextMonth(microseconds);
						double oldMean = monthlyReturns[index].mean;
						monthlyReturns[index].mean += (monthlyReturns[index].current - monthlyReturns[index].mean) / (double) ++monthlyReturns[index].n;
						monthlyReturns[index].m2 += (monthlyReturns[index].current - oldMean) * (monthlyReturns[index].current - monthlyReturns[index].mean);
						monthlyReturns[index].current = profitMargin;
					}

					if (!win) {
						drawdowns[index].current *= profitMargin;

						// consider moving start to the trade entry instead of exit
						if (drawdownLengths[index].drawdownStart == 0)
							drawdownLengths[index].drawdownStart = microseconds;

						lossStreaks[index].current++;
					} else {
						if (lossStreaks[index].current != 0) {
							drawdowns[index].max = min(drawdowns[index].max, drawdowns[index].current);
							drawdowns[index].mean += (drawdowns[index].current - drawdowns[index].mean) / (double) ++lossStreaks[index].n;
							drawdowns[index].current = 1;

							long newDrawdownLength = microseconds - drawdownLengths[index].drawdownStart;
							drawdownLengths[index].max = max(drawdownLengths[index].max, newDrawdownLength);
							drawdownLengths[index].mean += ((double) newDrawdownLength - drawdownLengths[index].mean) / (double) lossStreaks[index].n;
							drawdownLengths[index].drawdownStart = 0;

							lossStreaks[index].max = max(lossStreaks[index].max, lossStreaks[index].current);
							lossStreaks[index].mean += ((double) lossStreaks[index].current - lossStreaks[index].mean) / (double) lossStreaks[index].n;
							lossStreaks[index].current = 0;
						}
					}
				}
			} else {
				profitMargin = 2 - price / e.price;
				if (inClose || profitMargin >= precomputedTarget || profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					e = (entry) {0.0, false};
					bool win = profitMargin >= 1.0;
					sw += win;
					sl += !win;
					tradesInInterval++;

					long newTradeDuration = microseconds - tradeDurations[index].entryTimestamp;
					tradeDurations[index].max = max(tradeDurations[index].max, newTradeDuration);
					tradeDurations[index].mean += ((double) newTradeDuration - tradeDurations[index].mean) / (double) ++tradeDurations[index].n;

					if (monthlyReturns[index].nextMonth == 0)
						monthlyReturns[index].nextMonth = getTsOfNextMonth(microseconds);

					if (microseconds < monthlyReturns[index].nextMonth) {
						monthlyReturns[index].current *= profitMargin;
					} else {
						monthlyReturns[index].nextMonth = getTsOfNextMonth(microseconds);
						double oldMean = monthlyReturns[index].mean;
						monthlyReturns[index].mean += (monthlyReturns[index].current - monthlyReturns[index].mean) / (double) ++monthlyReturns[index].n;
						monthlyReturns[index].m2 += (monthlyReturns[index].current - oldMean) * (monthlyReturns[index].current - monthlyReturns[index].mean);
						monthlyReturns[index].current = profitMargin;
					}

					if (!win) {
						drawdowns[index].current *= profitMargin;

						// consider moving start to the trade entry instead of exit
						if (drawdownLengths[index].drawdownStart == 0)
							drawdownLengths[index].drawdownStart = microseconds;

						lossStreaks[index].current++;
					} else {
						if (lossStreaks[index].current != 0) {
							drawdowns[index].max = min(drawdowns[index].max, drawdowns[index].current);
							drawdowns[index].mean += (drawdowns[index].current - drawdowns[index].mean) / (double) ++lossStreaks[index].n;
							drawdowns[index].current = 1;

							long newDrawdownLength = microseconds - drawdownLengths[index].drawdownStart;
							drawdownLengths[index].max = max(drawdownLengths[index].max, newDrawdownLength);
							drawdownLengths[index].mean += ((double) newDrawdownLength - drawdownLengths[index].mean) / (double) lossStreaks[index].n;
							drawdownLengths[index].drawdownStart = 0;

							lossStreaks[index].max = max(lossStreaks[index].max, lossStreaks[index].current);
							lossStreaks[index].mean += ((double) lossStreaks[index].current - lossStreaks[index].mean) / (double) lossStreaks[index].n;
							lossStreaks[index].current = 0;
						}
					}
				}
			}
		}

		if (e.price == 0.0) {
			if (buyVol >= c.buyVolPercentile && !inClose && !onWeekend) {
				e = (entry) {price, true};
				ls += 1;
				tradeDurations[index].entryTimestamp = microseconds;
			} else if (sellVol >= c.sellVolPercentile && !inClose && !onWeekend) {
				e = (entry) {price, false};
				ss += 1;
				tradeDurations[index].entryTimestamp = microseconds;
			}
		}
	}

	entries[index] = e;
	tradeRecords[index] = (tradeRecord) {capital, ss, sw, sl, ls, lw, ll};
}
