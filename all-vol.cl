typedef struct __attribute__ ((packed)) tradeWithoutDate {
	long timestamp;
	double price;
	double qty;
	int tradeId;
	uchar isBuyerMaker;
} tradeWithoutDate;

typedef struct __attribute__ ((packed)) combo {
	long window;
	double buyVolPercentile;
	double sellVolPercentile;
	double buyDeltaPercentile;
	double sellDeltaPercentile;
	double stopLoss;
	double target;
} combo;

typedef struct __attribute__ ((packed)) entry {
	double price;
	bool isLong;
} entry;

typedef struct __attribute__ ((packed)) timeWindow {
	int tradeId;
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
	int tradeId;
} positionData;

typedef struct __attribute__ ((packed)) twMetadata {
	int twTranslation;
	int twStart;
} twMetadata;

__kernel void test(const int g, __global float* ds) {
	int index = get_global_id(0);
	printf("%d\n", index);
	ds[index] = 5;
	printf("%f\n", ds[index]);
}

#define MICROSECONDS_IN_HOUR 3600000000
#define MICROSECONDS_IN_DAY 86400000000
#define MICROSECONDS_IN_WEEK 604800000000
#define CME_CLOSE 79200000000
#define CME_OPEN 82800000000
#define CME_CLOSE_FRIDAY 165600000000
#define CME_OPEN_SUNDAY 342000000000

#define MARCH_1_1972_IN_SECONDS 68256000

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

__kernel void volTrader(__global int* numTrades, __global tradeWithoutDate* trades, __global combo* combos, __global entry* entries, __global tradeRecord* tradeRecords, __global positionData* positionDatas, __global twMetadata* twBetweenRuns) {
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
	timeWindow tw = {positionDatas[index].tradeId - twBetweenRuns->twTranslation, positionDatas[index].timestamp};
	double buyVol = positionDatas[index].buyVol;
	double sellVol = positionDatas[index].sellVol;
	entry e = entries[index];

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

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

		if (microseconds - tw.timestamp > c.window) {
			for (int k = tw.tradeId; k < i; k++) {
				long newMicroseconds = trades[k].timestamp;
				if (microseconds - newMicroseconds > c.window) {
					double newVol = trades[k].qty;
					if (trades[k].isBuyerMaker) sellVol -= newVol;
					else buyVol -= newVol;
				} else {
					tw = (timeWindow) {k, newMicroseconds};
					break;
				}
			}
		}
		if (trades[i].isBuyerMaker) sellVol += vol;
		else buyVol += vol;

		if (e.price == 0.0) {
			double volDelta = buyVol - sellVol;
			if (buyVol >= c.buyVolPercentile && volDelta >= c.buyDeltaPercentile && !inClose && !onWeekend) {
				e = (entry) {price, true};
				ls += 1;
			} else if (sellVol >= c.sellVolPercentile && volDelta <= c.sellDeltaPercentile !inClose && !onWeekend) {
				e = (entry) {price, false};
				ss += 1;
			}
		}
	}

	entries[index] = e;
	tradeRecords[index] = (tradeRecord) {capital, ss, sw, sl, ls, lw, ll};
	positionDatas[index] = (positionData) {tw.timestamp, buyVol, sellVol, tw.tradeId};
}
