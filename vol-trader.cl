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

typedef struct __attribute__ ((packed)) tradeRecord {
	double capital;
	int totalTrades;
	int wins;
	int losses;
} tradeRecord;

typedef struct __attribute__ ((packed)) positionData {
	long timestamp;
	double buyVol;
	double sellVol;
	int tradeId;
} positionData;

__kernel void test(const int g, __global float* ds) {
	int index = get_global_id(0);
	printf("%d\n", index);
	ds[index] = 5;
	printf("%f\n", ds[index]);
}

__kernel void volTrader(const int numTrades, __global tradeWithoutDate* trades, __global combo* combos, __global entry* entries, __global tradeRecord* tradeRecords, __global positionData* positionDatas) {
	int index = get_global_id(0);
	double capital = tradeRecords[index].capital;
	combo c = combos[index];
	// printf("%d %d %d %d %d\n", sizeof(int), sizeof(double), sizeof(long), sizeof(bool), sizeof(long long));
	// printf("%d %f %f %f %f\n", c.window, c.buyVolPercentile, c.sellVolPercentile, c.stopLoss, c.target);
	// printf("%d\n", numTrades);
	int t = tradeRecords[index].totalTrades;
	int l = tradeRecords[index].losses;
	int w = tradeRecords[index].wins;
	timeWindow tw = {positionDatas[index].tradeId, positionDatas[index].timestamp};
	double buyVol = positionDatas[index].buyVol;
	double sellVol = positionDatas[index].sellVol;
	entry e = entries[index];

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

	for (int i = 0; i < numTrades; i++) {
		double vol = trades[i].qty;
		double price = trades[i].price;
		// printf("%e %f %lld %d %d\n", vol, price, trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		// printf("%ld %d %d\n", trades[i].timestamp, trades[i].tradeId, trades[i].isBuyerMaker);
		long microseconds = trades[i].timestamp;

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

		if (e.price == 0.0) {
			if (buyVol >= c.buyVolPercentile) {
				e = (entry) {price, true};
				t += 1;
			} else if (sellVol >= c.sellVolPercentile) {
				e = (entry) {price, false};
				t += 1;
			}
		}
	}

	entries[index] = e;
	tradeRecords[index] = (tradeRecord) {capital, t, w, l};
	positionDatas[index] = (positionData) {tw.timestamp, buyVol, sellVol, tw.tradeId};
}
