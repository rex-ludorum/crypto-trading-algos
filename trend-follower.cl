typedef struct __attribute__ ((packed)) tradeWithoutDate {
	long timestamp;
	double price;
	double qty;
	int tradeId;
	uchar isBuyerMaker;
} tradeWithoutDate;

typedef struct __attribute__ ((packed)) combo {
	double entryThreshold;
	double stopLoss;
	double target;
} combo;

typedef struct __attribute__ ((packed)) entry {
	double price;
	bool isLong;
} entry;

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
	double maxPrice;
	double minPrice;
	uchar bools;
} positionData;

#define INCREASING_BIT 0

#define MICROSECONDS_IN_HOUR 3600000000
#define MICROSECONDS_IN_DAY 86400000000
#define MICROSECONDS_IN_WEEK 604800000000
#define CME_CLOSE 79200000000
#define CME_OPEN 82800000000
#define CME_CLOSE_FRIDAY 165600000000
#define CME_OPEN_SUNDAY 342000000000

#define MARCH_1_1972_IN_SECONDS 68256000

__kernel void test(const int g, __global float* ds) {
	int index = get_global_id(0);
	printf("%d\n", index);
	ds[index] = 5;
	printf("%f\n", ds[index]);
}

inline int isDST(long ts) {
	int timestamp = ts / 1000000;
	int leapYearCycles = (timestamp - MARCH_1_1972_IN_SECONDS) / ((365 * 4 + 1) * 86400);
	int days = (timestamp - MARCH_1_1972_IN_SECONDS) / 86400;
	int daysInCurrentCycle = days % (365 * 4 + 1);
	int yearsInCurrentCycle = daysInCurrentCycle / 365;
	int daysInCurrentYear = daysInCurrentCycle % 365;

	int timeInCurrentDay = timestamp % 86400;

	int marchFirstDayOfWeekInCurrentCycle = leapYearCycles * (365 * 4 + 1) % 7;
	int marchFirstDayOfWeekInCurrentYear = (marchFirstDayOfWeekInCurrentCycle + yearsInCurrentCycle * 365) % 7;

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

__kernel void trendFollower(__global int* numTrades, __global tradeWithoutDate* trades, __global combo* combos, __global entry* entries, __global tradeRecord* tradeRecords, __global positionData* positionDatas) {
	int index = get_global_id(0);
	double capital = tradeRecords[index].capital;
	combo c = combos[index];
	// printf("%d %d %d %d\n", sizeof(int), sizeof(double), sizeof(long), sizeof(bool));
	// printf("%d %f %f %f %f\n", c.window, c.buyVolPercentile, c.sellVolPercentile, c.stopLoss, c.target);
	int ls = tradeRecords[index].longs;
	int lw = tradeRecords[index].longWins;
	int ll = tradeRecords[index].longLosses;
	int ss = tradeRecords[index].shorts;
	int sw = tradeRecords[index].shortWins;
	int sl = tradeRecords[index].shortLosses;
	entry e = entries[index];
	double maxPrice = positionDatas[index].maxPrice;
	double minPrice = positionDatas[index].minPrice;
	bool increasing = positionDatas[index].bools & 1 << INCREASING_BIT;

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

	double precomputedLongEntryThreshold = 1 + c.entryThreshold * 0.01;
	double precomputedShortEntryThreshold = 1 - c.entryThreshold * 0.01;

	for (int i = 0; i < *numTrades; i++) {
		double price = trades[i].price;
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
				}
			} else {
				profitMargin = 2 - price / e.price;
				if (inClose || profitMargin >= precomputedTarget || profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					e = (entry) {0.0, false};
					sw += profitMargin >= 1.0;
					sl += profitMargin < 1.0;
				}
			}
		}

		// printf("%e %f %lld %d %d\n", vol, price, trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		// printf("%ld %d %d\n", trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		if (increasing) {
			maxPrice = max(maxPrice, price);
			if (price / maxPrice <= precomputedShortEntryThreshold) {
				minPrice = 1000000;
				increasing = false;
				if (e.price == 0.0 && !inClose && !onWeekend) {
					e = (entry) {price, false};
					ss += 1;
				}
			} else if (price / minPrice >= precomputedLongEntryThreshold) {
				if (e.price == 0.0 && !inClose && !onWeekend) {
					e = (entry) {price, true};
					ls += 1;
				}
			}
		} else {
			minPrice = min(minPrice, price);
			if (price / minPrice >= precomputedLongEntryThreshold) {
				maxPrice = 1;
				increasing = true;
				if (e.price == 0.0 && !inClose && !onWeekend) {
					e = (entry) {price, true};
					ls += 1;
				}
			} else if (price / maxPrice <= precomputedShortEntryThreshold) {
				if (e.price == 0.0 && !inClose && !onWeekend) {
					e = (entry) {price, false};
					ss += 1;
				}
			}
		}
	}

	entries[index] = e;
	tradeRecords[index] = (tradeRecord) {capital, ss, sw, sl, ls, lw, ll};
	uchar bools = 0;
	bools |= (uchar) increasing << INCREASING_BIT;
	positionDatas[index] = (positionData) {maxPrice, minPrice, bools};
}
