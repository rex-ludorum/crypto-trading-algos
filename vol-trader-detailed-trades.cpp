#include "ThreadPool.h"
#include "helper.h"
#include <CL/opencl.hpp>
#include <array>
#include <cassert>
#include <chrono>
#include <fstream>
#include <ios>
#include <iostream>
#include <ranges>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using std::array;
using std::boolalpha;
using std::defaultfloat;
using std::fixed;
using std::format;
using std::generate;
using std::get_time;
using std::ifstream;
using std::ios;
using std::istringstream;
using std::lock_guard;
using std::max;
using std::max_element;
using std::min;
using std::min_element;
using std::mktime;
using std::mutex;
using std::ofstream;
using std::ostream;
using std::stod;
using std::stoi;
using std::string;
using std::thread;
using std::tm;
using std::transform;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::system_clock;
using std::ranges::views::cartesian_product;

using std::cout;
using std::endl;

#define NUM_WINDOWS 4

#define INCREMENT 1000000
#define TRADE_CHUNK 50000000

#define PERCENTILE_CEILING 30
#define PERCENTILE_FLOOR 12

static size_t newStart = 0;
static size_t globalIdx = 0;

mutex coutMutex;

const vector<size_t> indicesToRun = {62134, 53504, 62983};

cl::Program program; // The program that will run on the device.
cl::Context context; // The context which holds the device.
cl::Device device;	 // The device where the kernel will run.

struct __attribute__((packed)) indicators {
	array<cl_double, 2 * NUM_WINDOWS> vols;
};

struct timeWindow {
	size_t tradeIdx;
	long long timestamp;
};

struct __attribute__((packed)) combo {
	cl_long window;
	cl_double buyVolPercentile;
	cl_double sellVolPercentile;
	cl_double stopLoss;
	cl_double target;
};

struct __attribute__((packed)) entryData {
	cl_double buyVol;
	cl_double sellVol;
};

struct __attribute__((packed)) entryAndExit {
	cl_double profitMargin;
	cl_int entryIndex;
	cl_int exitIndex;
	entryData e;
	cl_uchar isLong;
};

struct __attribute__((packed)) detailedTrade {
	double profitMargin;
	long long entryTimestamp;
	long long exitTimestamp;
	entryData e;
	bool isLong;
};

struct __attribute__((packed)) positionData {
	cl_long timestamp;
	cl_double buyVol;
	cl_double sellVol;
	cl_int tradeIdx;
};

struct __attribute__((packed)) twMetadata {
	cl_int twTranslation;
	cl_int twStart;
};

void computeIndicators(const vector<tradeWithoutDate> &tradesWithoutDates,
											 vector<indicators> &inds, vector<timeWindow> &tws,
											 const vector<long long> &windows, bool firstRun) {
#ifdef DEBUG
	static size_t testIdx;
	if (firstRun)
		assert(inds.size() == 0);
	else
		assert(inds.size() == newStart);
#endif
	indicators ind{};
	if (!inds.empty())
		ind = inds.back();
	// should be equal to newStart
	for (size_t i = inds.size(); i < tradesWithoutDates.size(); i++) {
		tradeWithoutDate t = tradesWithoutDates[i];
		if (t.isBuyerMaker) {
			for (size_t j = 0; j < NUM_WINDOWS; j++)
				ind.vols[2 * j] += t.qty * t.price;
		} else {
			for (size_t j = 0; j < NUM_WINDOWS; j++)
				ind.vols[2 * j + 1] += t.qty * t.price;
		}
		for (size_t j = 0; j < NUM_WINDOWS; j++) {
			if (t.timestamp - tws[j].timestamp > windows[j]) {
				size_t k;
				for (k = tws[j].tradeIdx; k < i; k++) {
					tradeWithoutDate oldT = tradesWithoutDates[k];
					if (t.timestamp - oldT.timestamp > windows[j]) {
						if (oldT.isBuyerMaker)
							ind.vols[2 * j] -= oldT.qty * oldT.price;
						else
							ind.vols[2 * j + 1] -= oldT.qty * oldT.price;
					} else {
						break;
					}
				}
				tws[j] = {k, tradesWithoutDates[k].timestamp};
			}
		}
#ifdef DEBUG
		testIdx++;
		if (testIdx % 500000 == 0) {
			cout << fixed;
			cout << "Timestamp: " << t.timestamp << endl;
			cout << "15 minute buy volume: " << ind.vols[1] << endl;
			cout << "15 minute sell volume: " << ind.vols[0] << endl;
			cout << "30 minute buy volume: " << ind.vols[3] << endl;
			cout << "30 minute sell volume: " << ind.vols[2] << endl;
			cout << "45 minute buy volume: " << ind.vols[5] << endl;
			cout << "45 minute sell volume: " << ind.vols[4] << endl;
			cout << "60 minute buy volume: " << ind.vols[7] << endl;
			cout << "60 minute sell volume: " << ind.vols[6] << endl;
			cout << defaultfloat;
		}
#endif
		inds.push_back(ind);
	}
}

void performWork(size_t index, size_t currIdx, size_t currSize,
								 const vector<tradeWithoutDate> &trades,
								 const vector<indicators> &inds, const vector<combo> &combos,
								 vector<entry> &entries, vector<tradeRecord> &tradeRecords,
								 vector<int> &numTradesInIntervalVec) {
	double capital = tradeRecords[index].capital;
	combo c = combos[index];
	entry e = entries[index];
	int ls = tradeRecords[index].longs;
	int lw = tradeRecords[index].longWins;
	int ll = tradeRecords[index].longLosses;
	int ss = tradeRecords[index].shorts;
	int sw = tradeRecords[index].shortWins;
	int sl = tradeRecords[index].shortLosses;

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

	int tradesInInterval = 0;

	int indicatorIdx = c.window / FIFTEEN_MINUTES_MICROSECONDS - 1;

	for (size_t i = currIdx; i < currIdx + currSize; i++) {
		double price = trades[i].price;
		double sellVol = inds[i].vols[2 * indicatorIdx];
		double buyVol = inds[i].vols[2 * indicatorIdx + 1];

		long long microseconds = trades[i].timestamp;
		int isDSTInt = isDst(microseconds);
		long long dayRemainder =
				microseconds % MICROSECONDS_IN_DAY + isDSTInt * MICROSECONDS_IN_HOUR;
		long long weekRemainder =
				microseconds % MICROSECONDS_IN_WEEK + isDSTInt * MICROSECONDS_IN_HOUR;
		bool inClose = dayRemainder >= CME_CLOSE && dayRemainder < CME_OPEN;
		bool onWeekend =
				weekRemainder >= CME_CLOSE_FRIDAY && weekRemainder < CME_OPEN_SUNDAY;

		if (e.price != 0.0) {
			double profitMargin;
			if (e.isLong) {
				profitMargin = price / e.price;
				if (inClose || profitMargin >= precomputedTarget ||
						profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					e = (entry){0.0, false};
					lw += profitMargin >= 1.0;
					ll += profitMargin < 1.0;
					tradesInInterval++;
				}
			} else {
				profitMargin = 2 - price / e.price;
				if (inClose || profitMargin >= precomputedTarget ||
						profitMargin <= precomputedStopLoss) {
					capital *= profitMargin;
					e = (entry){0.0, false};
					sw += profitMargin >= 1.0;
					sl += profitMargin < 1.0;
					tradesInInterval++;
				}
			}
		}

		if (e.price == 0.0) {
			if (buyVol >= c.buyVolPercentile && !inClose && !onWeekend) {
				e = (entry){price, true};
				ls += 1;
			} else if (sellVol >= c.sellVolPercentile && !inClose && !onWeekend) {
				e = (entry){price, false};
				ss += 1;
			}
		}
	}

	entries[index] = e;
	tradeRecords[index] = (tradeRecord){capital, ss, sw, sl, ls, lw, ll};
	numTradesInIntervalVec[index] = tradesInInterval;
}

void performWork(size_t index, size_t currIdx, size_t currSize,
								 const vector<tradeWithoutDate> &trades,
								 const vector<indicators> &inds, const vector<combo> &combos,
								 vector<entry> &entries, vector<tradeRecord> &tradeRecords,
								 vector<drawdowns> &drawdownsVec,
								 vector<drawdownLengths> &drawdownLengthsVec,
								 vector<lossStreaks> &lossStreaksVec,
								 vector<tradeDurations> &tradeDurationsVec,
								 vector<monthlyReturns> &monthlyReturnsVec,
								 vector<vector<detailedTrade>> &allTrades) {
	double capital = tradeRecords[index].capital;
	combo c = combos[index];
	entry e = entries[index];
	int ls = tradeRecords[index].longs;
	int lw = tradeRecords[index].longWins;
	int ll = tradeRecords[index].longLosses;
	int ss = tradeRecords[index].shorts;
	int sw = tradeRecords[index].shortWins;
	int sl = tradeRecords[index].shortLosses;

	double precomputedTarget = 1 + c.target * 0.01;
	double precomputedStopLoss = 1 - c.stopLoss * 0.01;

	int indicatorIdx = c.window / FIFTEEN_MINUTES_MICROSECONDS - 1;

	for (size_t i = currIdx; i < currIdx + currSize; i++) {
		double price = trades[i].price;
		double sellVol = inds[i].vols[2 * indicatorIdx];
		double buyVol = inds[i].vols[2 * indicatorIdx + 1];

		long long microseconds = trades[i].timestamp;
		int isDSTInt = isDst(microseconds);
		long long dayRemainder =
				microseconds % MICROSECONDS_IN_DAY + isDSTInt * MICROSECONDS_IN_HOUR;
		long long weekRemainder =
				microseconds % MICROSECONDS_IN_WEEK + isDSTInt * MICROSECONDS_IN_HOUR;
		bool inClose = dayRemainder >= CME_CLOSE && dayRemainder < CME_OPEN;
		bool onWeekend =
				weekRemainder >= CME_CLOSE_FRIDAY && weekRemainder < CME_OPEN_SUNDAY;

		if (e.price != 0.0) {
			double profitMargin;
			if (e.isLong) {
				profitMargin = price / e.price;
				{
					lock_guard<mutex> lock(coutMutex);
					cout << profitMargin << endl;
				}
			} else {
				profitMargin = 2 - price / e.price;
			}
			if (inClose || profitMargin >= precomputedTarget ||
					profitMargin <= precomputedStopLoss) {
				capital *= profitMargin;
				bool win = profitMargin >= 1.0;
				if (e.isLong) {
					lw += win;
					ll += !win;
				} else {
					sw += win;
					sl += !win;
				}

#ifdef DEBUG
				assert(!allTrades[index].empty());

				detailedTrade currentTrade = allTrades[index].back();
				assert(currentTrade.profitMargin == 0);
				assert(currentTrade.exitTimestamp == 0);
				assert(currentTrade.entryTimestamp > 0);
				assert(microseconds > currentTrade.entryTimestamp);
				assert(currentTrade.e.buyVol > 0);
				assert(currentTrade.e.sellVol > 0);
#endif
				allTrades[index].back().profitMargin = profitMargin;
				allTrades[index].back().exitTimestamp = microseconds;

				e = (entry){0.0, false};

				long newTradeDuration =
						microseconds - tradeDurationsVec[index].entryTimestamp;
				tradeDurationsVec[index].max =
						max((long long)tradeDurationsVec[index].max,
								(long long)newTradeDuration);
				tradeDurationsVec[index].mean +=
						((double)newTradeDuration - tradeDurationsVec[index].mean) /
						(double)++tradeDurationsVec[index].n;

				if (monthlyReturnsVec[index].nextMonth == 0)
					monthlyReturnsVec[index].nextMonth = getTsOfNextMonth(microseconds);

				if (microseconds < monthlyReturnsVec[index].nextMonth) {
					monthlyReturnsVec[index].current *= profitMargin;
				} else {
					monthlyReturnsVec[index].nextMonth = getTsOfNextMonth(microseconds);
					double oldMean = monthlyReturnsVec[index].mean;
					monthlyReturnsVec[index].mean += (monthlyReturnsVec[index].current -
																						monthlyReturnsVec[index].mean) /
																					 (double)++monthlyReturnsVec[index].n;
					monthlyReturnsVec[index].m2 +=
							(monthlyReturnsVec[index].current - oldMean) *
							(monthlyReturnsVec[index].current -
							 monthlyReturnsVec[index].mean);
					monthlyReturnsVec[index].current = profitMargin;
				}

				if (!win) {
					drawdownsVec[index].current *= profitMargin;

					// consider moving start to the trade entry instead of exit
					if (drawdownLengthsVec[index].drawdownStart == 0)
						drawdownLengthsVec[index].drawdownStart = microseconds;

					lossStreaksVec[index].current++;
				} else {
					if (lossStreaksVec[index].current != 0) {
						drawdownsVec[index].max =
								min(drawdownsVec[index].max, drawdownsVec[index].current);
						drawdownsVec[index].mean +=
								(drawdownsVec[index].current - drawdownsVec[index].mean) /
								(double)++lossStreaksVec[index].n;
						drawdownsVec[index].current = 1;

						long newDrawdownLength =
								microseconds - drawdownLengthsVec[index].drawdownStart;
						drawdownLengthsVec[index].max =
								max((long long)drawdownLengthsVec[index].max,
										(long long)newDrawdownLength);
						drawdownLengthsVec[index].mean +=
								((double)newDrawdownLength - drawdownLengthsVec[index].mean) /
								(double)lossStreaksVec[index].n;
						drawdownLengthsVec[index].drawdownStart = 0;

						lossStreaksVec[index].max =
								max(lossStreaksVec[index].max, lossStreaksVec[index].current);
						lossStreaksVec[index].mean +=
								((double)lossStreaksVec[index].current -
								 lossStreaksVec[index].mean) /
								(double)lossStreaksVec[index].n;
						lossStreaksVec[index].current = 0;
					}
				}
			}
		}

		if (e.price == 0.0) {
			if (buyVol >= c.buyVolPercentile && !inClose && !onWeekend) {
				e = (entry){price, true};
				ls += 1;
				tradeDurationsVec[index].entryTimestamp = microseconds;
				detailedTrade currentTrade = {0, 0, 0, {0, 0}, false};
				currentTrade.entryTimestamp = microseconds;
				currentTrade.isLong = true;
				currentTrade.e.buyVol = buyVol;
				currentTrade.e.sellVol = sellVol;
				allTrades[index].push_back(currentTrade);
			} else if (sellVol >= c.sellVolPercentile && !inClose && !onWeekend) {
				e = (entry){price, false};
				ss += 1;
				tradeDurationsVec[index].entryTimestamp = microseconds;
				detailedTrade currentTrade = {0, 0, 0, {0, 0}, false};
				currentTrade.entryTimestamp = microseconds;
				currentTrade.isLong = false;
				currentTrade.e.buyVol = buyVol;
				currentTrade.e.sellVol = sellVol;
				allTrades[index].push_back(currentTrade);
			}
		}
	}

	entries[index] = e;
	tradeRecords[index] = (tradeRecord){capital, ss, sw, sl, ls, lw, ll};
}

int processTradesWithIndicators(vector<tradeWithoutDate> &tradesWithoutDates,
																vector<indicators> &inds,
																vector<timeWindow> &tws,
																const vector<combo> &comboVec,
																vector<entry> &entriesVec,
																vector<tradeRecord> &tradeRecordsVec,
																vector<cl_int> &numTradesInIntervalVec) {
	size_t currIdx = newStart;

	int maxTradesPerInterval = 0;

	while (true) {
		size_t currSize =
				min((size_t)INCREMENT, tradesWithoutDates.size() - currIdx);

		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << " (" << globalIdx << "-" << globalIdx + currSize - 1 << ")" << endl;
		ThreadPool pool(thread::hardware_concurrency());
		for (size_t i = 0; i < comboVec.size(); ++i) {
			pool.enqueue([=, &tradesWithoutDates, &inds, &comboVec, &entriesVec,
										&tradeRecordsVec, &numTradesInIntervalVec]() {
				/*
				{
					lock_guard<mutex> lock(coutMutex);
					cout << "Task " << i << " running in thread " <<
				std::this_thread::get_id() << endl;
				}
				*/
				performWork(i, currIdx, currSize, tradesWithoutDates, inds, comboVec,
										entriesVec, tradeRecordsVec, numTradesInIntervalVec);
			});
		}
		pool.wait();

		// cout << *max_element(numTradesInIntervalVec.begin(),
		// numTradesInIntervalVec.end())
		// 		 << endl;
		maxTradesPerInterval =
				max(maxTradesPerInterval, *max_element(numTradesInIntervalVec.begin(),
																							 numTradesInIntervalVec.end()));

		globalIdx += currSize;
		currIdx += currSize;

		if (currIdx >= tradesWithoutDates.size()) {
			break;
		}
	}

#ifdef DEBUG
	assert(currIdx == tradesWithoutDates.size());
#endif

	int minElementIdx = min_element(tws.begin(), tws.end(),
																	[](timeWindow t1, timeWindow t2) {
																		return t1.tradeIdx < t2.tradeIdx;
																	}) -
											tws.begin();
	int minTradeIdx = tws[minElementIdx].tradeIdx;
	for (auto &tw : tws) {
		tw.tradeIdx -= minTradeIdx;
	}

	newStart = currIdx - minTradeIdx;
	tradesWithoutDates.erase(tradesWithoutDates.begin(),
													 tradesWithoutDates.begin() + minTradeIdx);
	inds.erase(inds.begin(), inds.begin() + minTradeIdx);

	return maxTradesPerInterval;
}

void processTradesWithOnlineAlgs(vector<tradeWithoutDate> &tradesWithoutDates,
																 vector<indicators> &inds,
																 vector<timeWindow> &tws,
																 const vector<combo> &comboVec,
																 vector<entry> &entriesVec,
																 vector<tradeRecord> &tradeRecordsVec,
																 vector<drawdowns> &drawdownsVec,
																 vector<drawdownLengths> &drawdownLengthsVec,
																 vector<lossStreaks> &lossStreaksVec,
																 vector<tradeDurations> &tradeDurationsVec,
																 vector<monthlyReturns> &monthlyReturnsVec,
																 vector<vector<detailedTrade>> &allTrades) {
	size_t currIdx = newStart;

	while (true) {
		size_t currSize =
				min((size_t)INCREMENT, tradesWithoutDates.size() - currIdx);

		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << " (" << globalIdx << "-" << globalIdx + currSize - 1 << ")" << endl;
		ThreadPool pool(thread::hardware_concurrency());
		for (size_t i = 0; i < comboVec.size(); ++i) {
			pool.enqueue([=, &tradesWithoutDates, &inds, &comboVec, &entriesVec,
										&tradeRecordsVec, &drawdownsVec, &drawdownLengthsVec,
										&lossStreaksVec, &tradeDurationsVec, &monthlyReturnsVec,
										&allTrades]() {
				/*
				{
					lock_guard<mutex> lock(coutMutex);
					cout << "Task " << i << " running in thread " <<
				std::this_thread::get_id() << endl;
				}
				*/
				performWork(i, currIdx, currSize, tradesWithoutDates, inds, comboVec,
										entriesVec, tradeRecordsVec, drawdownsVec,
										drawdownLengthsVec, lossStreaksVec, tradeDurationsVec,
										monthlyReturnsVec, allTrades);
			});
		}
		pool.wait();

		globalIdx += currSize;
		currIdx += currSize;

		if (currIdx >= tradesWithoutDates.size()) {
			break;
		}
	}

#ifdef DEBUG
	assert(currIdx == tradesWithoutDates.size());
#endif

	int minElementIdx = min_element(tws.begin(), tws.end(),
																	[](timeWindow t1, timeWindow t2) {
																		return t1.tradeIdx < t2.tradeIdx;
																	}) -
											tws.begin();
	int minTradeIdx = tws[minElementIdx].tradeIdx;
	for (auto &tw : tws) {
		tw.tradeIdx -= minTradeIdx;
	}

	newStart = currIdx - minTradeIdx;
	tradesWithoutDates.erase(tradesWithoutDates.begin(),
													 tradesWithoutDates.begin() + minTradeIdx);
	inds.erase(inds.begin(), inds.begin() + minTradeIdx);

	return;
}

void outputMetrics(ostream &os, size_t idx, const vector<combo> &comboVec,
									 const vector<tradeRecord> &tradeRecordsVec,
									 const vector<perfMetrics> &allPerfMetrics, bool listTrades) {
	os << fixed;
	os << "Combo index: " << indicesToRun[idx] << endl;
	os << "Annualized return: " << tradeRecordsVec[idx].capital << endl;
	os << "Target: " << format("{:.2f}", (double)comboVec[idx].target) << endl;
	os << "Stop loss: " << format("{:.2f}", (double)comboVec[idx].stopLoss)
		 << endl;
	os << "Window: " << comboVec[idx].window / ONE_MINUTE_MICROSECONDS
		 << " minutes" << endl;
	os << "Buy volume threshold: " << comboVec[idx].buyVolPercentile << endl;
	os << "Sell volume threshold: " << comboVec[idx].sellVolPercentile << endl;
	os << "Total trades: "
		 << tradeRecordsVec[idx].shorts + tradeRecordsVec[idx].longs << endl;
	os << "Wins: "
		 << tradeRecordsVec[idx].shortWins + tradeRecordsVec[idx].longWins << endl;
	os << "Losses: "
		 << tradeRecordsVec[idx].shortLosses + tradeRecordsVec[idx].longLosses
		 << endl;
	os << "Longs: " << tradeRecordsVec[idx].longs << endl;
	os << "Long wins: " << tradeRecordsVec[idx].longWins << endl;
	os << "Long losses: " << tradeRecordsVec[idx].longLosses << endl;
	os << "Shorts: " << tradeRecordsVec[idx].shorts << endl;
	os << "Short wins: " << tradeRecordsVec[idx].shortWins << endl;
	os << "Short losses: " << tradeRecordsVec[idx].shortLosses << endl;

	if (listTrades) {
		os << "Sharpe ratio: " << allPerfMetrics[idx].sharpe << endl;
		os << "Average drawdown: " << allPerfMetrics[idx].avgDrawdown << endl;
		os << "Max drawdown: " << allPerfMetrics[idx].maxDrawdown << endl;
		os << "Average loss streak: " << allPerfMetrics[idx].avgLossStreak << endl;
		os << "Max loss streak: " << allPerfMetrics[idx].maxLossStreak << endl;
		os << "Average drawdown length: " << allPerfMetrics[idx].avgDrawdownLength
			 << endl;
		os << "Max drawdown length: " << allPerfMetrics[idx].maxDrawdownLength
			 << endl;
		os << "Average trade duration: " << allPerfMetrics[idx].avgTradeDuration
			 << endl;
		os << "Max trade duration: " << allPerfMetrics[idx].maxTradeDuration
			 << endl;
	}
	os << defaultfloat;
}

void outputMetrics(ostream &os, size_t idx, const vector<combo> &comboVec,
									 const vector<tradeRecord> &tradeRecordsVec,
									 const vector<perfMetrics> &allPerfMetrics,
									 const vector<drawdowns> &drawdownsVec,
									 const vector<drawdownLengths> &drawdownLengthsVec,
									 const vector<lossStreaks> &lossStreaksVec,
									 const vector<tradeDurations> &tradeDurationsVec,
									 const vector<vector<detailedTrade>> &allTrades,
									 bool listTrades) {
	os << fixed;
	os << "Combo index: " << indicesToRun[idx] << endl;
	os << "Annualized return: " << tradeRecordsVec[idx].capital << endl;
	os << "Target: " << format("{:.2f}", (double)comboVec[idx].target) << endl;
	os << "Stop loss: " << format("{:.2f}", (double)comboVec[idx].stopLoss)
		 << endl;
	os << "Window: " << comboVec[idx].window / ONE_MINUTE_MICROSECONDS
		 << " minutes" << endl;
	os << "Buy volume threshold: " << comboVec[idx].buyVolPercentile << endl;
	os << "Sell volume threshold: " << comboVec[idx].sellVolPercentile << endl;
	os << "Total trades: "
		 << tradeRecordsVec[idx].shorts + tradeRecordsVec[idx].longs << endl;
	os << "Wins: "
		 << tradeRecordsVec[idx].shortWins + tradeRecordsVec[idx].longWins << endl;
	os << "Losses: "
		 << tradeRecordsVec[idx].shortLosses + tradeRecordsVec[idx].longLosses
		 << endl;
	os << "Longs: " << tradeRecordsVec[idx].longs << endl;
	os << "Long wins: " << tradeRecordsVec[idx].longWins << endl;
	os << "Long losses: " << tradeRecordsVec[idx].longLosses << endl;
	os << "Shorts: " << tradeRecordsVec[idx].shorts << endl;
	os << "Short wins: " << tradeRecordsVec[idx].shortWins << endl;
	os << "Short losses: " << tradeRecordsVec[idx].shortLosses << endl;

	if (listTrades) {
		os << "Sharpe ratio: " << allPerfMetrics[idx].sharpe << endl;
		os << "Average drawdown: " << drawdownsVec[idx].mean << endl;
		os << "Max drawdown: " << drawdownsVec[idx].max << endl;
		os << "Average loss streak: " << lossStreaksVec[idx].mean << endl;
		os << "Max loss streak: " << lossStreaksVec[idx].max << endl;
		os << "Average drawdown length: "
			 << drawdownLengthsVec[idx].mean / (double)ONE_HOUR_MICROSECONDS
			 << " hours" << endl;
		os << "Max drawdown length: "
			 << drawdownLengthsVec[idx].max / (double)ONE_HOUR_MICROSECONDS
			 << " hours" << endl;
		os << "Average trade duration: "
			 << tradeDurationsVec[idx].mean / (double)ONE_MINUTE_MICROSECONDS
			 << " minutes" << endl;
		os << "Max trade duration: "
			 << (double)tradeDurationsVec[idx].max / (double)ONE_MINUTE_MICROSECONDS
			 << " minutes" << endl;
		if (&os != &cout) {
			for (size_t j = 0; j < allTrades[idx].size(); j++) {
				detailedTrade e = allTrades[idx][j];
				if (e.isLong) {
					os << "Long ";
				} else {
					os << "Short ";
				}
				os << e.profitMargin << endl;
				os << e.entryTimestamp << " ";
				auto tp = system_clock::time_point(microseconds(e.entryTimestamp));
				os << format("{:%Y-%m-%d %H:%M:%S}", tp) << endl;
				os << e.exitTimestamp << " ";
				tp = system_clock::time_point(microseconds(e.exitTimestamp));
				os << format("{:%Y-%m-%d %H:%M:%S}", tp) << endl;
				os << "Buy volume: " << e.e.buyVol << endl;
				os << "Sell volume: " << e.e.sellVol << endl;
				os << endl;
			}
		}
	}
	os << defaultfloat;
}

int main(int argc, char *argv[]) {
	bool writeResults = false, listTrades = false;

	bool isBTC;

	int opt;
	while ((opt = getopt(argc, argv, "wl")) != -1) {
		switch (opt) {
		case 'w':
			writeResults = true;
			cout << "Enabled writing results" << endl;
			break;
		case 'l':
			listTrades = true;
			cout << "Enabled listing trades" << endl;
			break;
		case '?':
			cout << "Got unknown option: " << (char)optopt << endl;
			exit(EXIT_FAILURE);
		default:
			cout << "Got unknown parse returns: " << opt << endl;
			exit(EXIT_FAILURE);
		}
	}

	if (optind == argc) {
		cout << "No file names specified!" << endl;
		exit(EXIT_FAILURE);
	}

	string dateSuffix;
	string firstFile = argv[optind];
	size_t firstDateFirstUnderscore = firstFile.find("_");
	size_t firstDateSecondUnderscore =
			firstFile.find("_", firstDateFirstUnderscore + 1);
	dateSuffix += firstFile.substr(firstDateFirstUnderscore + 1,
																 firstDateSecondUnderscore -
																		 firstDateFirstUnderscore - 1);
	dateSuffix += "_";
	string lastFile = argv[argc - 1];
	size_t lastDateFirstUnderscore = lastFile.find("_");
	size_t lastDateSecondUnderscore =
			lastFile.find("_", lastDateFirstUnderscore + 1);
	size_t lastDateLastPeriod = lastFile.rfind(".");
	dateSuffix +=
			lastFile.substr(lastDateSecondUnderscore + 1,
											lastDateLastPeriod - lastDateSecondUnderscore - 1);

	string symbol(argv[optind]);
	if (symbol.find("BTC") != string::npos) {
		isBTC = true;
	} else {
		isBTC = false;
	}

#if defined(__WIN64)
	_putenv("TZ=/usr/share/zoneinfo/UTC");
#elif defined(__linux)
	char c[] = "TZ=/usr/share/zoneinfo/UTC";
	putenv(c);
#endif

	auto startTime = high_resolution_clock::now();

	vector<long long> windows(NUM_WINDOWS);
	long long n = 0;
	generate(windows.begin(), windows.end(),
					 [n]() mutable { return n += FIFTEEN_MINUTES_MICROSECONDS; });

	vector<double> stopLosses(12);
	double x = 0;
	generate(stopLosses.begin(), stopLosses.end(),
					 [x]() mutable { return x += 0.25; });

	vector<double> targets(20);
	x = 0;
	generate(targets.begin(), targets.end(), [x]() mutable { return x += 0.25; });

	vector<vector<double>> buyVolPercentiles;
	vector<vector<double>> sellVolPercentiles;

	ifstream myFile;
	if (isBTC)
		myFile.open("buyPercentilesDollarsBTC");
	else
		myFile.open("buyPercentilesDollarsETH");
	if (myFile.is_open()) {
		string line;
		while (getline(myFile, line)) {
			vector<string> splits = split(line);
			vector<double> rowPercentiles;
			for (string s : splits) {
				rowPercentiles.emplace_back(stod(s));
			}
			rowPercentiles.erase(rowPercentiles.begin() + PERCENTILE_CEILING,
													 rowPercentiles.end());
			rowPercentiles.erase(rowPercentiles.begin(),
													 rowPercentiles.begin() + PERCENTILE_FLOOR);
			buyVolPercentiles.emplace_back(rowPercentiles);
		}
		myFile.close();
	}

	if (isBTC)
		myFile.open("sellPercentilesDollarsBTC");
	else
		myFile.open("sellPercentilesDollarsETH");
	if (myFile.is_open()) {
		string line;
		while (getline(myFile, line)) {
			vector<string> splits = split(line);
			vector<double> rowPercentiles;
			for (string s : splits) {
				rowPercentiles.emplace_back(stod(s));
			}
			rowPercentiles.erase(rowPercentiles.begin() + PERCENTILE_CEILING,
													 rowPercentiles.end());
			rowPercentiles.erase(rowPercentiles.begin(),
													 rowPercentiles.begin() + PERCENTILE_FLOOR);
			sellVolPercentiles.emplace_back(rowPercentiles);
		}
		myFile.close();
	}

	vector<vector<long long>> indWindows(NUM_WINDOWS);
	for (int i = 0; i < NUM_WINDOWS; i++) {
		indWindows[i] = vector<long long>(1, windows[i]);
	}
	auto firstCombo =
			cartesian_product(indWindows[0], buyVolPercentiles[0],
												sellVolPercentiles[0], stopLosses, targets);
	vector<decltype(firstCombo)> combos;
	combos.emplace_back(firstCombo);

	for (int i = 1; i < NUM_WINDOWS; i++) {
		combos.emplace_back(cartesian_product(indWindows[i], buyVolPercentiles[i],
																					sellVolPercentiles[i], stopLosses,
																					targets));
	}

	vector<combo> comboVec, tempCombos;

	for (int i = 0; i < NUM_WINDOWS; i++) {
		for (unsigned int j = 0; j < combos[i].size(); j++) {
			combo c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
								 get<2>(combos[i][j]), get<3>(combos[i][j]),
								 get<4>(combos[i][j])};
			// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
			// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
			if (c.stopLoss < c.target + 1.1)
				tempCombos.emplace_back(c);
		}
	}

	for (size_t i : indicesToRun) {
		comboVec.push_back(tempCombos[i]);
	}
	tempCombos.clear();

	size_t comboVecSize = comboVec.size() * sizeof(combo);

	vector<positionData> positionDatasVec(comboVec.size(), {0, 0.0, 0.0, 0});
	bool initializedPositions = false;

	vector<entry> entriesVec(comboVec.size(), {0.0, 0});
	size_t entriesVecSize = entriesVec.size() * sizeof(entry);
	vector<tradeRecord> tradeRecordsVec(comboVec.size(), {1.0, 0, 0, 0, 0, 0, 0});
	size_t tradeRecordsVecSize = tradeRecordsVec.size() * sizeof(tradeRecord);

	int maxTradesPerInterval = 0;

	vector<vector<entryAndExit>> entriesAndExitsVec;

	vector<drawdowns> drawdownsVec;
	vector<drawdownLengths> drawdownLengthsVec;
	vector<lossStreaks> lossStreaksVec;
	vector<tradeDurations> tradeDurationsVec;
	vector<monthlyReturns> monthlyReturnsVec;

	vector<cl_int> numTradesInIntervalVec;

	cout << "Number of combinations: " << comboVec.size() << endl;
	cout << "Size of combinations: " << comboVecSize << endl;
	// cout << "Size of trades: " << inputTradesSize << endl;
	// cout << "Size of indicators: " << indicatorsSize << endl;
	cout << "Size of entries: " << entriesVecSize << endl;
	cout << "Size of trade records: " << tradeRecordsVecSize << endl;
	// size_t totalSize = comboVecSize + inputTradesSize + indicatorsSize +
	// 									 entriesVecSize + tradeRecordsVecSize;
	size_t totalSize = comboVecSize + entriesVecSize + tradeRecordsVecSize;
	if (listTrades) {
		drawdownsVec = vector<drawdowns>(comboVec.size(), {1.0, 0.0, 1.0});
		size_t drawdownsVecSize = drawdownsVec.size() * sizeof(drawdowns);
		cout << "Size of drawdowns: " << drawdownsVecSize << endl;

		drawdownLengthsVec =
				vector<drawdownLengths>(comboVec.size(), drawdownLengths{});
		size_t drawdownLengthsVecSize =
				drawdownLengthsVec.size() * sizeof(drawdownLengths);
		cout << "Size of drawdown lengths: " << drawdownLengthsVecSize << endl;

		lossStreaksVec = vector<lossStreaks>(comboVec.size(), lossStreaks{});
		size_t lossStreaksVecSize = lossStreaksVec.size() * sizeof(lossStreaks);
		cout << "Size of loss streaks: " << lossStreaksVecSize << endl;

		tradeDurationsVec =
				vector<tradeDurations>(comboVec.size(), tradeDurations{});
		size_t tradeDurationsVecSize =
				tradeDurationsVec.size() * sizeof(tradeDurations);
		cout << "Size of trade durations: " << tradeDurationsVecSize << endl;

		monthlyReturnsVec =
				vector<monthlyReturns>(comboVec.size(), monthlyReturns{});
		size_t monthlyReturnsVecSize =
				monthlyReturnsVec.size() * sizeof(monthlyReturns);
		cout << "Size of monthly returns: " << monthlyReturnsVecSize << endl;

		totalSize += drawdownsVecSize + drawdownLengthsVecSize +
								 lossStreaksVecSize + tradeDurationsVecSize +
								 monthlyReturnsVecSize;
		cout << "Total size: " << totalSize << endl;
		cout << "Total size: " << (double)totalSize / (double)(1024 * 1024 * 1024)
				 << " GiB" << endl;
	} else {
		numTradesInIntervalVec = vector<cl_int>(comboVec.size(), 0);
		size_t numTradesInIntervalVecSize =
				numTradesInIntervalVec.size() * sizeof(cl_int);
		cout << "Size of number of trades in interval: "
				 << numTradesInIntervalVecSize << endl;

		totalSize += numTradesInIntervalVecSize;
		cout << "Total size: " << totalSize << endl;
		cout << "Total size: " << (double)totalSize / (double)(1024 * 1024 * 1024)
				 << " GiB" << endl;
	}

	vector<trade> trades;
	vector<tradeWithoutDate> tradesWithoutDates;
	vector<indicators> indicators;
	vector<timeWindow> tws(NUM_WINDOWS, {0, 0});

	bool justProcessed = false;

	vector<vector<detailedTrade>> allTrades;
	vector<perfMetrics> allPerfMetrics;
	if (listTrades) {
		allTrades.resize(comboVec.size());
		allPerfMetrics.resize(comboVec.size());
	}

	auto setupTime = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(setupTime - startTime);
	cout << "Time taken for setup: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	long long firstTimestamp = 0, lastTimestamp = 0;
	bool firstRun = true;
#ifdef DEBUG
	int lastId = 0;
#endif

	auto beforeFileReadTime = high_resolution_clock::now();
	auto afterFileReadTime = high_resolution_clock::now();
	size_t currIdx = 0;
	for (int i = optind; i < argc; i++) {
		myFile.open(argv[i]);
		if (myFile.is_open()) {
			cout << "Reading " << argv[i] << endl;
			string line;
			getline(myFile, line);
			while (getline(myFile, line)) {
				if (justProcessed) {
					beforeFileReadTime = high_resolution_clock::now();
					cout << "Going back to reading" << endl;
				}

				vector<string> splits = splitByComma(line);
				trade t;
				// t.tradeId = stoi(splits[3]);
				t.price = stod(splits[4]);
				t.qty = stod(splits[5]);
				string isBuyerMakerString = splits[6];
				transform(isBuyerMakerString.begin(), isBuyerMakerString.end(),
									isBuyerMakerString.begin(),
									[](unsigned char c) { return tolower(c); });
				istringstream(isBuyerMakerString) >> boolalpha >> t.isBuyerMaker;

				/*
				string date = splits[1] + "T" + splits[2];
				t.date = date;
				tm localTimeTm;
				int micros;
				istringstream(date) >> get_time(&localTimeTm, "%Y-%m-%dT%H:%M:%S.") >>
						micros;
				micros /= 1000;
				localTimeTm.tm_isdst = 0;
				auto tpLocal = system_clock::from_time_t(mktime(&localTimeTm));
				t.timestamp =
						duration_cast<microseconds>(tpLocal.time_since_epoch()).count() +
						micros;
				*/
				t.timestamp = stoll(splits[0]);
				if (firstTimestamp == 0)
					firstTimestamp = t.timestamp;
#ifdef DEBUG
				if (lastTimestamp != 0)
					assert(t.timestamp > lastTimestamp);
				if (lastId != 0)
					assert(stoi(splits[3]) > lastId);
				lastId = stoi(splits[3]);
#endif
				lastTimestamp = t.timestamp;
				// trades.emplace_back(t);

				tradesWithoutDates.emplace_back(convertTrade(t));
				if (!initializedPositions) {
					/*
					for (auto &pos : positionDatasVec) {
						pos.timestamp = t.timestamp;
					}
					*/
					for (auto &tw : tws) {
						tw.timestamp = t.timestamp;
					}
					initializedPositions = true;
				}
				justProcessed = false;
				if (tradesWithoutDates.size() - newStart == TRADE_CHUNK) {
					afterFileReadTime = high_resolution_clock::now();
					duration = duration_cast<microseconds>(afterFileReadTime -
																								 beforeFileReadTime);
					cout << "Time taken to read trades: "
							 << (double)duration.count() / 1000000 << " seconds" << endl;

					auto beforeIndicatorTime = high_resolution_clock::now();
					computeIndicators(tradesWithoutDates, indicators, tws, windows,
														firstRun);
					if (firstRun)
						firstRun = false;
					auto afterIndicatorTime = high_resolution_clock::now();
					duration = duration_cast<microseconds>(afterIndicatorTime -
																								 beforeIndicatorTime);
					cout << "Time taken to compute indicators: "
							 << (double)duration.count() / 1000000 << " seconds" << endl;
					if (listTrades)
						// processTradesWithListing(queue, volKernel, tradesWithoutDates,
						// 												 comboVec, inputTrades, inputSize,
						// 												 twBetweenRunData, positionDatas,
						// 												 numTradesInIntervalBuf, allTrades);
						// processTradesWithListingAndIndicators(
						//		queue, volKernel, tradesWithoutDates, indicators, tws,
						//		comboVec, inputTrades, indicatorBuffer, inputSize,
						//		entriesAndExitsBuf, allTrades);
						processTradesWithOnlineAlgs(tradesWithoutDates, indicators, tws,
																				comboVec, entriesVec, tradeRecordsVec,
																				drawdownsVec, drawdownLengthsVec,
																				lossStreaksVec, tradeDurationsVec,
																				monthlyReturnsVec, allTrades);
					else
						maxTradesPerInterval =
								max(maxTradesPerInterval,
										// processTrades(queue, volKernel, tradesWithoutDates,
										// comboVec, 							inputTrades, inputSize,
										// twBetweenRunData, 							positionDatas,
										// numTradesInIntervalBuf));
										processTradesWithIndicators(
												tradesWithoutDates, indicators, tws, comboVec,
												entriesVec, tradeRecordsVec, numTradesInIntervalVec));
					justProcessed = true;
				}
				currIdx++;
			}
			myFile.close();
		}
	}

	if (!justProcessed) {
		afterFileReadTime = high_resolution_clock::now();
		duration =
				duration_cast<microseconds>(afterFileReadTime - beforeFileReadTime);
		cout << "Time taken to read trades: " << (double)duration.count() / 1000000
				 << " seconds" << endl;

		if (tradesWithoutDates.size() > 0) {
			auto beforeIndicatorTime = high_resolution_clock::now();
			computeIndicators(tradesWithoutDates, indicators, tws, windows, firstRun);
			if (firstRun)
				firstRun = false;
			auto afterIndicatorTime = high_resolution_clock::now();
			duration =
					duration_cast<microseconds>(afterIndicatorTime - beforeIndicatorTime);
			cout << "Time taken to compute indicators: "
					 << (double)duration.count() / 1000000 << " seconds" << endl;
			if (listTrades)
				// processTradesWithListing(queue, volKernel, tradesWithoutDates,
				// comboVec, 												 inputTrades, inputSize,
				// twBetweenRunData, 												 positionDatas,
				// numTradesInIntervalBuf,
				//												 allTrades);
				// processTradesWithListingAndIndicators(
				// 		queue, volKernel, tradesWithoutDates, indicators, tws, comboVec,
				// 		inputTrades, indicatorBuffer, inputSize, entriesAndExitsBuf,
				// 		allTrades);
				processTradesWithOnlineAlgs(
						tradesWithoutDates, indicators, tws, comboVec, entriesVec,
						tradeRecordsVec, drawdownsVec, drawdownLengthsVec, lossStreaksVec,
						tradeDurationsVec, monthlyReturnsVec, allTrades);
			else
				maxTradesPerInterval =
						max(maxTradesPerInterval,
								// processTrades(queue, volKernel, tradesWithoutDates, comboVec,
								// 							inputTrades, inputSize, twBetweenRunData,
								// 							positionDatas, numTradesInIntervalBuf));
								processTradesWithIndicators(
										tradesWithoutDates, indicators, tws, comboVec, entriesVec,
										tradeRecordsVec, numTradesInIntervalVec));
		} else
			cout << "Run was already finished in the snapshot" << endl;
	}

	for (size_t i = 0; i < tradeRecordsVec.size(); i++) {
		tradeRecordsVec[i].capital = capitalToAnnualizedReturn(
				tradeRecordsVec[i].capital, firstTimestamp, lastTimestamp);
	}

	auto afterKernelTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(afterKernelTime - setupTime);
	cout << endl;
	cout << "Time taken to run trades: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	string resultsFilename = "resultsVol";
	if (isBTC)
		resultsFilename += "BTC";
	else
		resultsFilename += "ETH";

	if (listTrades) {
		resultsFilename += "WithTrades";
		// analyzePerf(allPerfMetrics, allTrades);
		analyzePerf(allPerfMetrics, monthlyReturnsVec, drawdownsVec);

		auto afterAnalysisTime = high_resolution_clock::now();
		duration = duration_cast<microseconds>(afterAnalysisTime - afterKernelTime);
		cout << "Time taken to analyze models: "
				 << (double)duration.count() / 1000000 << " seconds" << endl;
	}
	resultsFilename += "_";
	resultsFilename += dateSuffix;

	if (writeResults) {
		ofstream outFile;
		outFile.open(resultsFilename);
		if (outFile.is_open()) {
			for (size_t i = 0; i < comboVec.size(); i++) {
				outputMetrics(outFile, i, comboVec, tradeRecordsVec, allPerfMetrics,
											drawdownsVec, drawdownLengthsVec, lossStreaksVec,
											tradeDurationsVec, allTrades, listTrades);
				outFile << endl;
			}
			outFile.close();

			auto outputTime = high_resolution_clock::now();
			duration = duration_cast<microseconds>(outputTime - afterKernelTime);
			cout << "Time taken to write output: "
					 << (double)duration.count() / 1000000 << " seconds" << endl;
			cout << endl;
		}
	}

	for (size_t i = 0; i < comboVec.size(); i++) {
		outputMetrics(cout, i, comboVec, tradeRecordsVec, allPerfMetrics,
									drawdownsVec, drawdownLengthsVec, lossStreaksVec,
									tradeDurationsVec, allTrades, listTrades);
		cout << endl;
	}

	if (!listTrades)
		cout << "Max trades per interval: " << maxTradesPerInterval << endl;

	auto endTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(endTime - startTime);
	cout << "Total time taken: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	return 0;
}
