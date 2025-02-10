#define LOSS_BIT 0
#define WIN_BIT 1
#define LONG_BIT 2
#define SHORT_BIT 3

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

typedef struct __attribute__ ((packed)) entryAndExit {
	int entryIndex;
	int exitIndex;
	int longShortWinLoss;
} entryAndExit;

__kernel void test(const int g, __global float* ds) {
	int index = get_global_id(0);
	printf("%d\n", index);
	ds[index] = 5;
	printf("%f\n", ds[index]);
}

__kernel void volTrader(const int numTrades, __global tradeWithoutDate* trades, __global combo* combos, __global double* capitals, __global int* totalTrades, __global int* wins, __global int* losses) {
	int index = get_global_id(0);
	double capital = 1.0;
	combo c = combos[index];
	// printf("%d %d %d %d\n", sizeof(int), sizeof(double), sizeof(long), sizeof(bool));
	// printf("%d %f %f %f %f\n", c.window, c.buyVolPercentile, c.sellVolPercentile, c.stopLoss, c.target);
	int t = 0, l = 0, w = 0;
	timeWindow tw = {0, 0};
	double buyVol = 0;
	double sellVol = 0;
	double maxProfit = 0;
	entry e = {0, false};
	int currentTradeIdx = 0;
	entryAndExit currentTrade = {0, 0, 0};

	for (int i = 0; i < numTrades; i++) {
		double vol = trades[i].qty;
		double price = trades[i].price;
		// printf("%e %f %lld %d %d\n", vol, price, trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		// printf("%ld %d %d\n", trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		long microseconds = trades[i].timestamp;

		if (microseconds - tw.timestamp > c.window) {
			for (int k = tw.tradeId; k < i; k++) {
				long long newMicroseconds = trades[k].timestamp;
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
			if (buyVol >= c.buyVolPercentile) {
				maxProfit = price;
				e = (entry) {price, true};
				currentTrade.entryIndex = i;
				currentTrade.longShortWinLoss |= 1 << LONG_BIT;
				// tradeLogs[j][k].emplace_back(joinStrings(entries[j][k]));
				t += 1;
			} else if (sellVol >= c.sellVolPercentile) {
				maxProfit = price;
				e = (entry) {price, false};
				currentTrade.entryIndex = i;
				currentTrade.longShortWinLoss |= 1 << SHORT_BIT;
				// tradeLogs[j][k].emplace_back(joinStrings(entries[j][k]));
				t += 1;
			}
		} else {
			if (e.isLong) {
				maxProfit = max(maxProfit, price);
				double profitMargin = (maxProfit - e.price) / e.price;
				if (profitMargin >= c.target / 100) {
					capital *= 1 + profitMargin;
					e = (entry) {0.0, false};
					// tradeLogs[j][k].emplace_back("Profit: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
					// tradeLogs[j][k].emplace_back("Capital: " + to_string(capitals[j][k]));
					w += 1;
					currentTrade.exitIndex = i;
					currentTrade.longShortWinLoss |= 1 << WIN_BIT;
					currentTrade = (entryAndExit) {0, 0, 0};
				} else if (price <= (1 - c.stopLoss / 100) * e.price) {
					capital *= 1 - c.stopLoss / 100;
					e = (entry) {0.0, false};
					// tradeLogs[j][k].emplace_back("Loss: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
					// tradeLogs[j][k].emplace_back("Capital: " + to_string(capitals[j][k]));
					l += 1;
					currentTrade.exitIndex = i;
					currentTrade.longShortWinLoss |= 1 << LOSS_BIT;
					currentTrade = (entryAndExit) {0, 0, 0};
				}
			} else {
				maxProfit = min(maxProfit, price);
				double profitMargin = (e.price - maxProfit) / e.price;
				if (profitMargin >= c.target / 100) {
					capital *= 1 + profitMargin;
					e = (entry) {0.0, false};
					// tradeLogs[j][k].emplace_back("Profit: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
					// tradeLogs[j][k].emplace_back("Capital: " + to_string(capitals[j][k]));
					w += 1;
					currentTrade.exitIndex = i;
					currentTrade.longShortWinLoss |= 1 << WIN_BIT;
					currentTrade = (entryAndExit) {0, 0, 0};
				} else if (price >= (1 + c.stopLoss / 100) * e.price) {
					capital *= 1 - c.stopLoss / 100;
					e = (entry) {0.0, false};
					// tradeLogs[j][k].emplace_back("Loss: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
					// tradeLogs[j][k].emplace_back("Capital: " + to_string(capitals[j][k]));
					l += 1;
					currentTrade.exitIndex = i;
					currentTrade.longShortWinLoss |= 1 << LOSS_BIT;
					currentTrade = (entryAndExit) {0, 0, 0};
				}
			}
		}
	}

	totalTrades[index] = t;
	wins[index] = w;
	losses[index] = l;
	capitals[index] = capital;
}
