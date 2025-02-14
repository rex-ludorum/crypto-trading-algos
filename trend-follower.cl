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
	int totalTrades;
	int wins;
	int losses;
} tradeRecord;

typedef struct __attribute__ ((packed)) positionData {
	double maxPrice;
	double minPrice;
	uchar bools;
} positionData;

#define STARTED_BIT 0
#define INCREASING_BIT 1

__kernel void test(const int g, __global float* ds) {
	int index = get_global_id(0);
	printf("%d\n", index);
	ds[index] = 5;
	printf("%f\n", ds[index]);
}

__kernel void trendFollower(__global int* numTrades, __global tradeWithoutDate* trades, __global combo* combos, __global entry* entries, __global tradeRecord* tradeRecords, __global positionData* positionDatas) {
	int index = get_global_id(0);
	double capital = tradeRecords[index].capital;
	combo c = combos[index];
	// printf("%d %d %d %d\n", sizeof(int), sizeof(double), sizeof(long), sizeof(bool));
	// printf("%d %f %f %f %f\n", c.window, c.buyVolPercentile, c.sellVolPercentile, c.stopLoss, c.target);
	int t = tradeRecords[index].totalTrades;
	int l = tradeRecords[index].losses;
	int w = tradeRecords[index].wins;
	entry e = entries[index];
	double maxPrice = positionDatas[index].maxPrice;
	double minPrice = positionDatas[index].minPrice;
	bool started = positionDatas[index].bools & 1 << STARTED_BIT;
	bool increasing = positionDatas[index].bools & 1 << INCREASING_BIT;

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

	double precomputedLongEntryThreshold = 1 + c.entryThreshold * 0.01;
	double precomputedShortEntryThreshold = 1 - c.entryThreshold * 0.01;

	for (int i = 0; i < *numTrades; i++) {
		double price = trades[i].price;

		if (e.price != 0.0) {
			double profitMargin;
			if (e.isLong) {
				profitMargin = price / e.price;
			} else {
				profitMargin = 2 - price / e.price;
			}

			if (profitMargin >= precomputedTarget) {
				capital *= profitMargin;
				e = (entry) {0.0, false};
				w += 1;
			} else if (profitMargin <= precomputedStopLoss) {
				capital *= profitMargin;
				e = (entry) {0.0, false};
				l += 1;
			}
		}

		// printf("%e %f %lld %d %d\n", vol, price, trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		// printf("%ld %d %d\n", trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		if (started) {
			if (increasing) {
				maxPrice = max(maxPrice, price);
			} else {
				minPrice = min(minPrice, price);
			}
		}

		if (price / minPrice >= precomputedLongEntryThreshold) {
			if (started && !increasing) {
				maxPrice = 1;
				increasing = true;
				if (e.price == 0.0) {
					e = (entry) {price, true};
					t += 1;
				}
			} else if (!started) {
				started = true;
				increasing = true;
			}
		} else if (price / maxPrice <= precomputedShortEntryThreshold) {
			if (started && increasing) {
				minPrice = 1000000;
				increasing = false;
				if (e.price == 0.0) {
					e = (entry) {price, false};
					t += 1;
				}
			} else if (!started) {
				started = true;
				increasing = false;
			}
		}
	}

	entries[index] = e;
	tradeRecords[index] = (tradeRecord) {capital, t, w, l};
	uchar bools = 0;
	bools |= (uchar) started << STARTED_BIT;
	bools |= (uchar) increasing << INCREASING_BIT;
	positionDatas[index] = (positionData) {maxPrice, minPrice, bools};
}
