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
	double maxProfit = 0;
	entry e = {0, false};

	double pivotPrice = trades[0].price;
	double maxPrice = trades[0].price;
	double minPrice = trades[0].price;
	bool started = false;
	bool increasing = false;

	for (int i = 0; i < numTrades; i++) {
		double vol = trades[i].qty;
		double price = trades[i].price;
		long microseconds = trades[i].timestamp;
		// printf("%e %f %lld %d %d\n", vol, price, trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		// printf("%ld %d %d\n", trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);

		if (e.price != 0.0) {
			if (e.isLong) {
				double profitMargin = (price - e.price) / e.price;
				if (profitMargin >= c.target / 100) {
					capital *= 1 + profitMargin;
					e = (entry) {0.0, false};
					w += 1;
				} else if (price <= (1 - c.stopLoss / 100) * e.price) {
					capital *= 1 - c.stopLoss / 100;
					e = (entry) {0.0, false};
					l += 1;
				}
			} else {
				double profitMargin = (e.price - price) / e.price;
				if (profitMargin >= c.target / 100) {
					capital *= 1 + profitMargin;
					e = (entry) {0.0, false};
					w += 1;
				} else if (price >= (1 + c.stopLoss / 100) * e.price) {
					capital *= 1 - c.stopLoss / 100;
					e = (entry) {0.0, false};
					l += 1;
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

		if (started) {
			if (increasing) {
				maxPrice = max(maxPrice, price);
			} else {
				minPrice = min(minPrice, price);
			}
		}

		if ((price - minPrice) / minPrice * 100 >= c.entryThreshold) {
			if (started && !increasing) {
				pivotPrice = minPrice;
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
		} else if ((maxPrice - price) / maxPrice * 100 >= c.entryThreshold) {
			if (started && increasing) {
				pivotPrice = maxPrice;
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

		if (e.price == 0.0) {
			if (buyVol >= c.buyVolPercentile) {
				maxProfit = price;
				e = (entry) {price, true};
				t += 1;
			} else if (sellVol >= c.sellVolPercentile) {
				maxProfit = price;
				e = (entry) {price, false};
				t += 1;
			}
		}
	}

	totalTrades[index] = t;
	wins[index] = w;
	losses[index] = l;
	capitals[index] = capital;
}
