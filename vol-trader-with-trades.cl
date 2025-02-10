#define MAX_TOTAL_TRADES 42

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

__kernel void volTrader(const int numTrades, __global tradeWithoutDate* trades, __global combo* combos, __global double* capitals, __global int* totalTrades, __global int* wins, __global int* losses, __global entryAndExit* entriesAndExits) {
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
				entriesAndExits[index * MAX_TOTAL_TRADES + currentTradeIdx] = currentTrade;
				// tradeLogs[j][k].emplace_back(joinStrings(entries[j][k]));
				t += 1;
			} else if (sellVol >= c.sellVolPercentile) {
				maxProfit = price;
				e = (entry) {price, false};
				currentTrade.entryIndex = i;
				currentTrade.longShortWinLoss |= 1 << SHORT_BIT;
				entriesAndExits[index * MAX_TOTAL_TRADES + currentTradeIdx] = currentTrade;
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
					entriesAndExits[index * MAX_TOTAL_TRADES + currentTradeIdx++] = currentTrade;
					currentTrade = (entryAndExit) {0, 0, 0};
				} else if (price <= (1 - c.stopLoss / 100) * e.price) {
					capital *= 1 - c.stopLoss / 100;
					e = (entry) {0.0, false};
					// tradeLogs[j][k].emplace_back("Loss: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
					// tradeLogs[j][k].emplace_back("Capital: " + to_string(capitals[j][k]));
					l += 1;
					currentTrade.exitIndex = i;
					currentTrade.longShortWinLoss |= 1 << LOSS_BIT;
					entriesAndExits[index * MAX_TOTAL_TRADES + currentTradeIdx++] = currentTrade;
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
					entriesAndExits[index * MAX_TOTAL_TRADES + currentTradeIdx++] = currentTrade;
					currentTrade = (entryAndExit) {0, 0, 0};
				} else if (price >= (1 + c.stopLoss / 100) * e.price) {
					capital *= 1 - c.stopLoss / 100;
					e = (entry) {0.0, false};
					// tradeLogs[j][k].emplace_back("Loss: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
					// tradeLogs[j][k].emplace_back("Capital: " + to_string(capitals[j][k]));
					l += 1;
					currentTrade.exitIndex = i;
					currentTrade.longShortWinLoss |= 1 << LOSS_BIT;
					entriesAndExits[index * MAX_TOTAL_TRADES + currentTradeIdx++] = currentTrade;
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

	/*
	// vector<entry> entries;
	vector<combo> comboVect;
	int numEntries = 0;

	for (int i = 0; i < NUM_WINDOWS; i++) {
		for (int j = 0; j < combos[i].size(); j++) {
			combo c = {get<0>(combos[i][j]), get<1>(combos[i][j]), get<2>(combos[i][j]), get<3>(combos[i][j]), get<4>(combos[i][j])};
			comboVect.emplace_back(c);
		}
		// entries[i] = vector< tuple<bool, double, string, int> >(combos[i].size(), {false, 0.0, "", 0});
		numEntries += combos[i].size();
		// tradeLogs[i] = vector< vector<string> >(combos[i].size());
		// startingCapitals[i] = vector<double>(combos[i].size(), 1.0);
		// maxProfits[i] = vector<double>(combos[i].size(), 0.0);
		// totalTrades[i] = vector<int>(combos[i].size(), 0);
		// wins[i] = vector<int>(combos[i].size(), 0);
		// losses[i] = vector<int>(combos[i].size(), 0);
	}
	cl::Buffer inputCombos(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, comboVect.size() * sizeof(combo), &comboVect[0]);
	// vector<double> entries(numEntries, 0);
	cl::Buffer entries(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS, numEntries * sizeof(int));
	// vector<double> startingCapitals(numStartingCapitals, 0);
	cl::Buffer startingCapitals(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS, numEntries * sizeof(int));
	// vector<double> maxProfits(numMaxProfits, 0);
	cl::Buffer maxProfits(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS, numEntries * sizeof(int));
	// vector<int> totalTrades(numTotalTrades, 0);
	cl::Buffer totalTrades(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS, numEntries * sizeof(int));
	// vector<int> wins(numWins, 0);
	cl::Buffer wins(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS, numEntries * sizeof(int));
	// vector<int> losses(numLosses, 0);
	cl::Buffer losses(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS, numEntries * sizeof(int));

	for (int i = 0; i < trades.size(); i++) {
		double vol = trades[i].qty;
		double price = trades[i].price;
		long long microseconds = trades[i].timestamp;

		for (int j = 0; j < windows.size(); j++) {
			if (microseconds - get<1>(timeWindows[j]) > windows[j]) {
				for (int k = get<0>(timeWindows[j]); k < i; k++) {
					long long newMicroseconds = trades[k].timestamp;
					if (microseconds - newMicroseconds > windows[j]) {
						double newVol = trades[k].qty;
						if (trades[k].isBuyerMaker) sellVols[j] -= newVol;
						else buyVols[j] -= newVol;
					} else {
						timeWindows[j] = {k, newMicroseconds};
						break;
					}
				}
			}
			if (trades[i].isBuyerMaker) sellVols[j] += vol;
			else buyVols[j] += vol;

			for (int k = 0; k < combos[j].size(); k++) {
				double buyVolPercentile = get<0>(combos[j][k]);
				double sellVolPercentile = get<1>(combos[j][k]);
				double stopLoss = get<2>(combos[j][k]);
				double target = get<3>(combos[j][k]);

				if (get<3>(entries[j][k]) == 0) {
					if (buyVols[j] >= buyVolPercentile) {
						maxProfits[j][k] = price;
						entries[j][k] = {true, price, trades[i].date, trades[i].tradeId};
						tradeLogs[j][k].emplace_back(joinStrings(entries[j][k]));
						totalTrades[j][k] += 1;
					} else if (sellVols[j] >= sellVolPercentile) {
						maxProfits[j][k] = price;
						entries[j][k] = {false, price, trades[i].date, trades[i].tradeId};
						tradeLogs[j][k].emplace_back(joinStrings(entries[j][k]));
						totalTrades[j][k] += 1;
					}
				} else {
					if (get<0>(entries[j][k])) {
						maxProfits[j][k] = max(maxProfits[j][k], price);
						double profitMargin = (maxProfits[j][k] - get<1>(entries[j][k])) / get<1>(entries[j][k]);
						if (profitMargin >= target / 100) {
							startingCapitals[j][k] *= 1 + profitMargin;
							entries[j][k] = {false, 0.0, "", 0};
							tradeLogs[j][k].emplace_back("Profit: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
							tradeLogs[j][k].emplace_back("Capital: " + to_string(startingCapitals[j][k]));
							wins[j][k] += 1;
						} else if (price <= (1 - stopLoss / 100) * get<1>(entries[j][k])) {
							startingCapitals[j][k] *= 1 - stopLoss / 100;
							entries[j][k] = {false, 0.0, "", 0};
							tradeLogs[j][k].emplace_back("Loss: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
							tradeLogs[j][k].emplace_back("Capital: " + to_string(startingCapitals[j][k]));
							losses[j][k] += 1;
						}
					} else {
						maxProfits[j][k] = min(maxProfits[j][k], price);
						double profitMargin = (get<1>(entries[j][k]) - maxProfits[j][k]) / get<1>(entries[j][k]);
						if (profitMargin >= target / 100) {
							startingCapitals[j][k] *= 1 + profitMargin;
							entries[j][k] = {false, 0.0, "", 0};
							tradeLogs[j][k].emplace_back("Profit: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
							tradeLogs[j][k].emplace_back("Capital: " + to_string(startingCapitals[j][k]));
							wins[j][k] += 1;
						} else if (price >= (1 + stopLoss / 100) * get<1>(entries[j][k])) {
							startingCapitals[j][k] *= 1 - stopLoss / 100;
							entries[j][k] = {false, 0.0, "", 0};
							tradeLogs[j][k].emplace_back("Loss: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
							tradeLogs[j][k].emplace_back("Capital: " + to_string(startingCapitals[j][k]));
							losses[j][k] += 1;
						}
					}
				}
			}
		}
	}

	std::ofstream outFile;
	outFile.open("resultsCpp");
	if (outFile.is_open()) {
		for (int i = 0; i < NUM_WINDOWS; i++) {
			for (int j = 0; j < combos[i].size(); j++) {
				outFile << "Time window: " << to_string(windows[i]) << endl;
				outFile << "Buy vol threshold: " << to_string(get<0>(combos[i][j])) << endl;
				outFile << "Sell vol threshold: " << to_string(get<1>(combos[i][j])) << endl;
				outFile << "Stop loss: " << to_string(get<2>(combos[i][j])) << endl;
				outFile << "Target: " << to_string(get<3>(combos[i][j])) << endl;
				outFile << "Total trades: " << to_string(totalTrades[i][j]) << endl;
				outFile << "Wins: " << to_string(wins[i][j]) << endl;
				outFile << "Losses: " << to_string(losses[i][j]) << endl;
				outFile << "Final capital: " << to_string(startingCapitals[i][j]) << endl;
				for (string s : tradeLogs[i][j]) {
					outFile << s << endl;
				}
				outFile << endl;
			}
		}
		outFile.close();
	}
	*/
