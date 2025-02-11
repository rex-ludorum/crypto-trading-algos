typedef struct __attribute__ ((packed)) tradeWithoutDate {
	long timestamp;
	double price;
	double qty;
	int tradeId;
	int isBuyerMaker;
} tradeWithoutDate;

typedef struct __attribute__ ((packed)) combo {
	long window;
	double buyVolPercentile;
	double sellVolPercentile;
	double entryThreshold;
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

__kernel void test(const int g, __global float* ds) {
	int index = get_global_id(0);
	printf("%d\n", index);
	ds[index] = 5;
	printf("%f\n", ds[index]);
}

__kernel void volTrendTrader(const int numTrades, __global tradeWithoutDate* trades, __global combo* combos, __global double* capitals, __global int* totalTrades, __global int* wins, __global int* losses) {
	int index = get_global_id(0);
	double capital = 1.0;
	combo c = combos[index];
	// printf("%d %d %d %d %d\n", sizeof(int), sizeof(double), sizeof(long), sizeof(bool), sizeof(long long));
	// printf("%d %f %f %f %f\n", c.window, c.buyVolPercentile, c.sellVolPercentile, c.stopLoss, c.target);
	// printf("%d\n", numTrades);
	int t = 0, l = 0, w = 0;
	timeWindow tw = {0, 0};
	double buyVol = 0;
	double sellVol = 0;
	entry e = {0, false};

	double maxPrice = trades[0].price;
	double minPrice = trades[0].price;
	bool started = false;
	bool increasing = false;

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

	double precomputedLongEntryThreshold = 1 + c.entryThreshold * 0.01;
	double precomputedShortEntryThreshold = 1 - c.entryThreshold * 0.01;

	for (int i = 0; i < numTrades; i++) {
		/*
		if (index == 0) {
			if (i & 524287 == 0) {
				printf("%d\n", i);
			} else if (i == numTrades - 1) {
				printf("last\n");
			}
		}
		*/
		double vol = trades[i].qty;
		double price = trades[i].price;
		long microseconds = trades[i].timestamp;
		// printf("%e %f %lld %d %d\n", vol, price, trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		// printf("%ld %d %d\n", trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);

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
				if (e.price == 0.0 && buyVol >= c.buyVolPercentile) {
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
				if (e.price == 0.0 && sellVol >= c.sellVolPercentile) {
					e = (entry) {price, false};
					t += 1;
				}
			} else if (!started) {
				started = true;
				increasing = false;
			}
		}
	}

	totalTrades[index] = t;
	wins[index] = w;
	losses[index] = l;
	capitals[index] = capital;
}
