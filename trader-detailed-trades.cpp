#include "ThreadPool.h"
#include "helper.h"
#include <CL/opencl.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <ios>
#include <iostream>
#include <ranges>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using std::boolalpha;
using std::decay_t;
using std::defaultfloat;
using std::fixed;
using std::format;
using std::generate;
using std::get_time;
using std::ifstream;
using std::ios;
using std::is_same_v;
using std::istringstream;
using std::lock_guard;
using std::max;
using std::max_element;
using std::min;
using std::min_element;
using std::mktime;
using std::mutex;
using std::ofstream;
using std::stod;
using std::stoi;
using std::string;
using std::thread;
using std::tm;
using std::transform;
using std::vector;
using std::visit;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::system_clock;
using std::ranges::views::cartesian_product;

using std::cout;
using std::endl;

#define INCREMENT 1000000
#define TRADE_CHUNK 50000000

#define PERCENTILE_CEILING 30
#define PERCENTILE_FLOOR 12

static size_t newStart = 0;
static size_t globalIdx = 0;

mutex coutMutex;

// const vector<size_t> indicesToRun = {62134, 53504, 62983};
// const vector<size_t> indicesToRun = {962240, 473688};
const vector<size_t> indicesToRun = {22596959, 14971560, 2588586};

Mode mode;

cl::Program program; // The program that will run on the device.
cl::Context context; // The context which holds the device.
cl::Device device;	 // The device where the kernel will run.

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

void processOnlineAlgs(int index, double capital, long long microseconds,
											 double profitMargin, bool win,
											 vector<drawdowns> &drawdowns,
											 vector<drawdownLengths> &drawdownLengths,
											 vector<lossStreaks> &lossStreaks,
											 vector<tradeDurations> &tradeDurations,
											 vector<monthlyReturns> &monthlyReturns,
											 vector<wins> &wins, vector<losses> &losses,
											 vector<twap> &twaps) {
	long long newTradeDuration =
			microseconds - tradeDurations[index].entryTimestamp;
	tradeDurations[index].max = max(tradeDurations[index].max, newTradeDuration);
	double oldMean = tradeDurations[index].mean;
	tradeDurations[index].mean +=
			((double)newTradeDuration - tradeDurations[index].mean) /
			(double)++tradeDurations[index].n;
	tradeDurations[index].m2 +=
			((double)newTradeDuration - oldMean) *
			((double)newTradeDuration - tradeDurations[index].mean);

	double newTwap =
			twaps[index].currentNumer / (double)twaps[index].currentDenom;
	twaps[index].max = max(newTwap, twaps[index].max);
	twaps[index].min = min(newTwap, twaps[index].min);
	oldMean = twaps[index].mean;
	twaps[index].mean += (newTwap - twaps[index].mean) / (double)++twaps[index].n;

	if (monthlyReturns[index].nextMonth == 0)
		monthlyReturns[index].nextMonth = getTsOfNextMonth(microseconds);

	if (microseconds < monthlyReturns[index].nextMonth) {
		monthlyReturns[index].current *= profitMargin;
	} else {
		monthlyReturns[index].nextMonth = getTsOfNextMonth(microseconds);
		oldMean = monthlyReturns[index].mean;
		monthlyReturns[index].mean +=
				(monthlyReturns[index].current - monthlyReturns[index].mean) /
				(double)++monthlyReturns[index].n;
		monthlyReturns[index].m2 +=
				(monthlyReturns[index].current - oldMean) *
				(monthlyReturns[index].current - monthlyReturns[index].mean);
		monthlyReturns[index].current = profitMargin;
	}

	if (!win) {
		// consider moving start to the trade entry instead of exit
		if (!drawdowns[index].enabled)
			drawdownLengths[index].drawdownStart = microseconds;

		drawdowns[index].enabled = true;
		drawdowns[index].currentMin = min(capital, drawdowns[index].currentMin);

		lossStreaks[index].current++;

		losses[index].max = min(profitMargin, losses[index].max);
		losses[index].min = max(profitMargin, losses[index].min);
		oldMean = losses[index].mean;
		losses[index].mean +=
				(profitMargin - losses[index].mean) / (double)++losses[index].n;
		losses[index].m2 +=
				(profitMargin - oldMean) * (profitMargin - losses[index].mean);
	} else {
		if (capital > drawdowns[index].currentMax && drawdowns[index].enabled) {
			double newDrawdown =
					drawdowns[index].currentMin / drawdowns[index].currentMax;
			oldMean = drawdowns[index].mean;
			drawdowns[index].max = min(drawdowns[index].max, newDrawdown);
			drawdowns[index].mean +=
					(newDrawdown - drawdowns[index].mean) / (double)++drawdowns[index].n;
			drawdowns[index].m2 +=
					(newDrawdown - oldMean) * (newDrawdown - drawdowns[index].mean);
			drawdowns[index].currentMin = 1000000000;
			drawdowns[index].enabled = false;

			long long newDrawdownLength =
					microseconds - drawdownLengths[index].drawdownStart;
			drawdownLengths[index].max =
					max(drawdownLengths[index].max, newDrawdownLength);
			oldMean = drawdownLengths[index].mean;
			drawdownLengths[index].mean +=
					((double)newDrawdownLength - drawdownLengths[index].mean) /
					(double)drawdowns[index].n;
			drawdownLengths[index].m2 +=
					((double)newDrawdownLength - oldMean) *
					((double)newDrawdownLength - drawdownLengths[index].mean);
		}
		drawdowns[index].currentMax = max(capital, drawdowns[index].currentMax);

		if (lossStreaks[index].current != 0) {
			lossStreaks[index].max =
					max(lossStreaks[index].max, lossStreaks[index].current);
			oldMean = lossStreaks[index].mean;
			lossStreaks[index].mean +=
					((double)lossStreaks[index].current - lossStreaks[index].mean) /
					(double)++lossStreaks[index].n;
			lossStreaks[index].m2 +=
					((double)lossStreaks[index].current - oldMean) *
					((double)lossStreaks[index].current - lossStreaks[index].mean);
			lossStreaks[index].current = 0;
		}

		wins[index].max = max(profitMargin, wins[index].max);
		wins[index].min = min(profitMargin, wins[index].min);
		oldMean = wins[index].mean;
		wins[index].mean +=
				(profitMargin - wins[index].mean) / (double)++wins[index].n;
		wins[index].m2 +=
				(profitMargin - oldMean) * (profitMargin - wins[index].mean);
	}
}

void processTWAPs(int index, long long microseconds, double price, entry e,
									vector<twap> &twaps) {
	long long timeDiff = microseconds - twaps[index].currentTimestamp;
	double increment = timeDiff * (twaps[index].prevPrice - e.price);
	if (e.isLong)
		twaps[index].currentNumer += increment;
	else
		twaps[index].currentNumer -= increment;
	twaps[index].currentDenom += timeDiff;
	twaps[index].currentTimestamp = microseconds;
	twaps[index].prevPrice = price;
}

void performWork(size_t index, size_t currIdx, size_t currSize,
								 const vector<tradeWithoutDate> &trades,
								 const vector<indicators> &inds, const ComboVec &combos,
								 vector<entry> &entriesVec,
								 vector<tradeRecord> &tradeRecordsVec,
								 vector<positionData> &positionDatasVec,
								 vector<int> &numTradesInIntervalVec) {
	double precomputedTarget, precomputedStopLoss, precomputedLongEntryThreshold,
			precomputedShortEntryThreshold, maxPrice, minPrice;
	int periodsAfterOpen, indicatorIdx;
	bool increasing;

	auto visitor = [index, &precomputedTarget, &precomputedStopLoss,
									&precomputedLongEntryThreshold,
									&precomputedShortEntryThreshold, &maxPrice, &minPrice,
									&increasing, &positionDatasVec, &periodsAfterOpen,
									&indicatorIdx](auto &combos) {
		using T = typename decay_t<decltype(combos)>::value_type;
		if constexpr (is_same_v<T, comboVolPerps> ||
									is_same_v<T, comboVolDeltaPerps> ||
									is_same_v<T, comboAllVolPerps>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		} else if constexpr (is_same_v<T, comboVolFutures> ||
												 is_same_v<T, comboVolDeltaFutures> ||
												 is_same_v<T, comboAllVolFutures>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		} else if constexpr (is_same_v<T, comboVolExitPerps>) {
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		} else if constexpr (is_same_v<T, comboVolExitFutures>) {
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		} else if constexpr (is_same_v<T, comboTrendPerps>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			precomputedLongEntryThreshold = 1 + combos[index].entryThreshold * 0.01;
			precomputedShortEntryThreshold = 1 - combos[index].entryThreshold * 0.01;
			maxPrice = positionDatasVec[index].maxPrice;
			minPrice = positionDatasVec[index].minPrice;
			increasing = positionDatasVec[index].bools;
		} else if constexpr (is_same_v<T, comboTrendFutures>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			precomputedLongEntryThreshold = 1 + combos[index].entryThreshold * 0.01;
			precomputedShortEntryThreshold = 1 - combos[index].entryThreshold * 0.01;
			maxPrice = positionDatasVec[index].maxPrice;
			minPrice = positionDatasVec[index].minPrice;
			increasing = positionDatasVec[index].bools;
		} else if constexpr (is_same_v<T, comboVolTrendPerps>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			precomputedLongEntryThreshold = 1 + combos[index].entryThreshold * 0.01;
			precomputedShortEntryThreshold = 1 - combos[index].entryThreshold * 0.01;
			maxPrice = positionDatasVec[index].maxPrice;
			minPrice = positionDatasVec[index].minPrice;
			increasing = positionDatasVec[index].bools;
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		} else if constexpr (is_same_v<T, comboVolTrendFutures>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			precomputedLongEntryThreshold = 1 + combos[index].entryThreshold * 0.01;
			precomputedShortEntryThreshold = 1 - combos[index].entryThreshold * 0.01;
			maxPrice = positionDatasVec[index].maxPrice;
			minPrice = positionDatasVec[index].minPrice;
			increasing = positionDatasVec[index].bools;
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		}
		periodsAfterOpen = combos[index].periodsAfterOpen;
	};

	visit(visitor, combos);

	double capital = tradeRecordsVec[index].capital;
	entry e = entriesVec[index];
	int ls = tradeRecordsVec[index].longs;
	int lw = tradeRecordsVec[index].longWins;
	int ll = tradeRecordsVec[index].longLosses;
	int ss = tradeRecordsVec[index].shorts;
	int sw = tradeRecordsVec[index].shortWins;
	int sl = tradeRecordsVec[index].shortLosses;

	int tradesInInterval = 0;

	double lastPrice = trades[currIdx].price;
	for (size_t i = currIdx; i < currIdx + currSize; i++) {
		double price = trades[i].price;
		long long microseconds = trades[i].timestamp;

		int isValidTradingHours = isInTradingHours(microseconds, periodsAfterOpen);

		double sellVol = inds[i].vols[2 * indicatorIdx];
		double buyVol = inds[i].vols[2 * indicatorIdx + 1];
		double volDelta = buyVol - sellVol;

		if (e.price != 0.0) {
			double profitMargin;

			auto visitor = [index, &e, &profitMargin, &capital, &lw, &ll, &sw, &sl,
											&tradesInInterval, precomputedTarget, precomputedStopLoss,
											price, lastPrice, isValidTradingHours, buyVol,
											sellVol](auto &combos) {
				using T = typename decay_t<decltype(combos)>::value_type;
				if constexpr (is_same_v<T, comboVolPerps> ||
											is_same_v<T, comboVolDeltaPerps> ||
											is_same_v<T, comboAllVolPerps> ||
											is_same_v<T, comboTrendPerps> ||
											is_same_v<T, comboVolTrendPerps>) {
					if (e.isLong)
						profitMargin = price / e.price - 1;
					else
						profitMargin = 1 - price / e.price;
					profitMargin *= combos[index].leverageMultiplier;
					profitMargin += 1;
					if (!isValidTradingHours || profitMargin >= precomputedTarget ||
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
						e = (entry){0.0, false};
						tradesInInterval++;
					}
				} else if constexpr (is_same_v<T, comboVolFutures> ||
														 is_same_v<T, comboVolDeltaFutures> ||
														 is_same_v<T, comboAllVolFutures> ||
														 is_same_v<T, comboTrendFutures> ||
														 is_same_v<T, comboVolTrendFutures>) {
					int contracts = capital / (e.price * BTC_MARGIN_RATE) *
													combos[index].contractMultiplier;
					double profit;
					if (e.isLong)
						profit = contracts * (price - e.price) * 0.1;
					else
						profit = contracts * (e.price - price) * 0.1;
					profitMargin = (capital + profit) / capital;
					if (!isValidTradingHours || profitMargin >= precomputedTarget ||
							profitMargin <= precomputedStopLoss) {
						if (abs(price - lastPrice) >= FUTURES_PRICE_CHANGE_LIMIT)
							profitMargin = max(min(profitMargin, precomputedTarget),
																 precomputedStopLoss);
						capital *= profitMargin;
						capital -= 2 * contracts * BTC_COMMISSION_PER_CONTRACT;
						bool win = profitMargin >= 1.0;
						if (e.isLong) {
							lw += win;
							ll += !win;
						} else {
							sw += win;
							sl += !win;
						}
						e = (entry){0.0, false};
						tradesInInterval++;
					}
				} else if constexpr (is_same_v<T, comboVolExitPerps>) {
					if (e.isLong)
						profitMargin = price / e.price - 1;
					else
						profitMargin = 1 - price / e.price;
					profitMargin *= combos[index].leverageMultiplier;
					profitMargin += 1;
					if (!isValidTradingHours ||
							(e.isLong && buyVol <= combos[index].buyVolExitPercentile) ||
							(!e.isLong && sellVol <= combos[index].sellVolExitPercentile)) {
						capital *= profitMargin;
						bool win = profitMargin >= 1.0;
						if (e.isLong) {
							lw += win;
							ll += !win;
						} else {
							sw += win;
							sl += !win;
						}
						e = (entry){0.0, false};
						tradesInInterval++;
					}
				} else if constexpr (is_same_v<T, comboVolExitFutures>) {
					int contracts = capital / (e.price * BTC_MARGIN_RATE) *
													combos[index].contractMultiplier;
					double profit;
					if (e.isLong)
						profit = contracts * (price - e.price) * 0.1;
					else
						profit = contracts * (e.price - price) * 0.1;
					profitMargin = (capital + profit) / capital;
					if (!isValidTradingHours ||
							(e.isLong && buyVol <= combos[index].buyVolExitPercentile) ||
							(!e.isLong && sellVol <= combos[index].sellVolExitPercentile)) {
						if (abs(price - lastPrice) >= FUTURES_PRICE_CHANGE_LIMIT)
							profitMargin = max(min(profitMargin, precomputedTarget),
																 precomputedStopLoss);
						capital *= profitMargin;
						capital -= 2 * contracts * BTC_COMMISSION_PER_CONTRACT;
						bool win = profitMargin >= 1.0;
						if (e.isLong) {
							lw += win;
							ll += !win;
						} else {
							sw += win;
							sl += !win;
						}
						e = (entry){0.0, false};
						tradesInInterval++;
					}
				}
			};

			visit(visitor, combos);
		}

		auto visitor =
				[index, &e, &ls, &ss, &maxPrice, &minPrice, &increasing, price, buyVol,
				 sellVol, volDelta, precomputedLongEntryThreshold,
				 precomputedShortEntryThreshold, isValidTradingHours](auto &combos) {
					using T = typename decay_t<decltype(combos)>::value_type;
					if constexpr (is_same_v<T, comboVolPerps> ||
												is_same_v<T, comboVolFutures> ||
												is_same_v<T, comboVolExitPerps> ||
												is_same_v<T, comboVolExitFutures>) {
						if (e.price == 0.0 && isValidTradingHours) {
							if (buyVol >= combos[index].buyVolPercentile) {
								e = (entry){price, true};
								ls += 1;
							} else if (sellVol >= combos[index].sellVolPercentile) {
								e = (entry){price, false};
								ss += 1;
							}
						}
					} else if constexpr (is_same_v<T, comboVolDeltaPerps> ||
															 is_same_v<T, comboVolDeltaFutures>) {
						if (e.price == 0.0 && isValidTradingHours) {
							if (volDelta >= combos[index].buyVolDeltaPercentile) {
								e = (entry){price, true};
								ls += 1;
							} else if (volDelta <= combos[index].sellVolDeltaPercentile) {
								e = (entry){price, false};
								ss += 1;
							}
						}
					} else if constexpr (is_same_v<T, comboAllVolPerps> ||
															 is_same_v<T, comboAllVolFutures>) {
						if (e.price == 0.0 && isValidTradingHours) {
							if (buyVol >= combos[index].buyVolPercentile &&
									volDelta >= combos[index].buyVolDeltaPercentile) {
								e = (entry){price, true};
								ls += 1;
							} else if (sellVol >= combos[index].sellVolPercentile &&
												 volDelta <= combos[index].sellVolDeltaPercentile) {
								e = (entry){price, false};
								ss += 1;
							}
						}
					} else if constexpr (is_same_v<T, comboTrendPerps> ||
															 is_same_v<T, comboTrendFutures>) {
						if (increasing) {
							maxPrice = max(maxPrice, price);
							if (price / maxPrice <= precomputedShortEntryThreshold) {
								minPrice = 1000000;
								increasing = false;
								if (e.price == 0.0 && isValidTradingHours) {
									e = (entry){price, false};
									ss += 1;
								}
							} else if (price / minPrice >= precomputedLongEntryThreshold) {
								if (e.price == 0.0 && isValidTradingHours) {
									e = (entry){price, true};
									ls += 1;
								}
							}
						} else {
							minPrice = min(minPrice, price);
							if (price / minPrice >= precomputedLongEntryThreshold) {
								maxPrice = 1;
								increasing = true;
								if (e.price == 0.0 && isValidTradingHours) {
									e = (entry){price, true};
									ls += 1;
								}
							} else if (price / maxPrice <= precomputedShortEntryThreshold) {
								if (e.price == 0.0 && isValidTradingHours) {
									e = (entry){price, false};
									ss += 1;
								}
							}
						}
					} else if constexpr (is_same_v<T, comboVolTrendPerps> ||
															 is_same_v<T, comboVolTrendFutures>) {
						if (increasing) {
							maxPrice = max(maxPrice, price);
							if (price / maxPrice <= precomputedShortEntryThreshold) {
								minPrice = 1000000;
								increasing = false;
								if (e.price == 0.0 && isValidTradingHours &&
										sellVol >= combos[index].sellVolPercentile) {
									e = (entry){price, false};
									ss += 1;
								}
							} else if (price / minPrice >= precomputedLongEntryThreshold) {
								if (e.price == 0.0 && isValidTradingHours &&
										buyVol >= combos[index].buyVolPercentile) {
									e = (entry){price, true};
									ls += 1;
								}
							}
						} else {
							minPrice = min(minPrice, price);
							if (price / minPrice >= precomputedLongEntryThreshold) {
								maxPrice = 1;
								increasing = true;
								if (e.price == 0.0 && isValidTradingHours &&
										buyVol >= combos[index].buyVolPercentile) {
									e = (entry){price, true};
									ls += 1;
								}
							} else if (price / maxPrice <= precomputedShortEntryThreshold) {
								if (e.price == 0.0 && isValidTradingHours &&
										sellVol >= combos[index].sellVolPercentile) {
									e = (entry){price, false};
									ss += 1;
								}
							}
						}
					}
				};

		visit(visitor, combos);

		lastPrice = price;
	}

	entriesVec[index] = e;
	tradeRecordsVec[index] = (tradeRecord){capital, ss, sw, sl, ls, lw, ll};
	positionDatasVec[index] = (positionData){maxPrice, minPrice, increasing};
	numTradesInIntervalVec[index] = tradesInInterval;
}

void performWork(size_t index, size_t currIdx, size_t currSize,
								 const vector<tradeWithoutDate> &trades,
								 const vector<indicators> &inds, const ComboVec &combos,
								 vector<entry> &entriesVec,
								 vector<tradeRecord> &tradeRecordsVec,
								 vector<positionData> &positionDatasVec,
								 vector<drawdowns> &drawdownsVec,
								 vector<drawdownLengths> &drawdownLengthsVec,
								 vector<lossStreaks> &lossStreaksVec,
								 vector<tradeDurations> &tradeDurationsVec,
								 vector<monthlyReturns> &monthlyReturnsVec,
								 vector<wins> &winsVec, vector<losses> &lossesVec,
								 vector<twap> &twapsVec, vector<DetailedTradeVec> &allTrades) {
	double precomputedTarget, precomputedStopLoss, precomputedLongEntryThreshold,
			precomputedShortEntryThreshold, maxPrice, minPrice;
	int periodsAfterOpen, indicatorIdx;
	bool increasing;

	auto visitor = [index, &precomputedTarget, &precomputedStopLoss,
									&precomputedLongEntryThreshold,
									&precomputedShortEntryThreshold, &maxPrice, &minPrice,
									&increasing, &positionDatasVec, &periodsAfterOpen,
									&indicatorIdx](auto &combos) {
		using T = typename decay_t<decltype(combos)>::value_type;
		if constexpr (is_same_v<T, comboVolPerps> ||
									is_same_v<T, comboVolDeltaPerps> ||
									is_same_v<T, comboAllVolPerps>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		} else if constexpr (is_same_v<T, comboVolFutures> ||
												 is_same_v<T, comboVolDeltaFutures> ||
												 is_same_v<T, comboAllVolFutures>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		} else if constexpr (is_same_v<T, comboVolExitPerps>) {
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		} else if constexpr (is_same_v<T, comboVolExitFutures>) {
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		} else if constexpr (is_same_v<T, comboTrendPerps>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			precomputedLongEntryThreshold = 1 + combos[index].entryThreshold * 0.01;
			precomputedShortEntryThreshold = 1 - combos[index].entryThreshold * 0.01;
			maxPrice = positionDatasVec[index].maxPrice;
			minPrice = positionDatasVec[index].minPrice;
			increasing = positionDatasVec[index].bools;
		} else if constexpr (is_same_v<T, comboTrendFutures>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			precomputedLongEntryThreshold = 1 + combos[index].entryThreshold * 0.01;
			precomputedShortEntryThreshold = 1 - combos[index].entryThreshold * 0.01;
			maxPrice = positionDatasVec[index].maxPrice;
			minPrice = positionDatasVec[index].minPrice;
			increasing = positionDatasVec[index].bools;
		} else if constexpr (is_same_v<T, comboVolTrendPerps>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			precomputedLongEntryThreshold = 1 + combos[index].entryThreshold * 0.01;
			precomputedShortEntryThreshold = 1 - combos[index].entryThreshold * 0.01;
			maxPrice = positionDatasVec[index].maxPrice;
			minPrice = positionDatasVec[index].minPrice;
			increasing = positionDatasVec[index].bools;
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		} else if constexpr (is_same_v<T, comboVolTrendFutures>) {
			precomputedTarget = 1 + combos[index].target * 0.01;
			precomputedStopLoss = 1 - combos[index].stopLoss * 0.01;
			precomputedLongEntryThreshold = 1 + combos[index].entryThreshold * 0.01;
			precomputedShortEntryThreshold = 1 - combos[index].entryThreshold * 0.01;
			maxPrice = positionDatasVec[index].maxPrice;
			minPrice = positionDatasVec[index].minPrice;
			increasing = positionDatasVec[index].bools;
			indicatorIdx = combos[index].window / FIFTEEN_MINUTES_MICROSECONDS - 1;
		}
		periodsAfterOpen = combos[index].periodsAfterOpen;
	};

	visit(visitor, combos);

	double capital = tradeRecordsVec[index].capital;
	entry e = entriesVec[index];
	int ls = tradeRecordsVec[index].longs;
	int lw = tradeRecordsVec[index].longWins;
	int ll = tradeRecordsVec[index].longLosses;
	int ss = tradeRecordsVec[index].shorts;
	int sw = tradeRecordsVec[index].shortWins;
	int sl = tradeRecordsVec[index].shortLosses;

	int tradesInInterval = 0;

	double lastPrice = trades[currIdx].price;
	for (size_t i = currIdx; i < currIdx + currSize; i++) {
		double price = trades[i].price;
		long long microseconds = trades[i].timestamp;

		int isValidTradingHours = isInTradingHours(microseconds, periodsAfterOpen);

		double sellVol = inds[i].vols[2 * indicatorIdx];
		double buyVol = inds[i].vols[2 * indicatorIdx + 1];
		double volDelta = buyVol - sellVol;

		if (e.price != 0.0) {
			processTWAPs(index, microseconds, price, e, twapsVec);
			double profitMargin;

			auto visitor = [index, &e, &profitMargin, &capital, &lw, &ll, &sw, &sl,
											&tradesInInterval, precomputedTarget, precomputedStopLoss,
											price, lastPrice, isValidTradingHours, buyVol, sellVol,
											microseconds, &drawdownsVec, &drawdownLengthsVec,
											&lossStreaksVec, &tradeDurationsVec, &monthlyReturnsVec,
											&winsVec, &lossesVec,
											&twapsVec](auto &combos, auto &trades) {
				using T = typename decay_t<decltype(combos)>::value_type;
				using U = typename decay_t<decltype(trades)>::value_type;
				if constexpr (is_same_v<T, comboVolPerps> ||
											is_same_v<T, comboVolDeltaPerps> ||
											is_same_v<T, comboAllVolPerps> ||
											is_same_v<T, comboTrendPerps> ||
											is_same_v<T, comboVolTrendPerps>) {
					if (e.isLong)
						profitMargin = price / e.price - 1;
					else
						profitMargin = 1 - price / e.price;
					profitMargin *= combos[index].leverageMultiplier;
					profitMargin += 1;
					if (!isValidTradingHours || profitMargin >= precomputedTarget ||
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
						e = (entry){0.0, false};
						tradesInInterval++;

						processOnlineAlgs(index, capital, microseconds, profitMargin, win,
															drawdownsVec, drawdownLengthsVec, lossStreaksVec,
															tradeDurationsVec, monthlyReturnsVec, winsVec,
															lossesVec, twapsVec);

#ifdef DEBUG
						assert(!trades.empty());

						auto &currentTrade = trades.back();
						assert(currentTrade.profitMargin == 0);
						assert(currentTrade.exitTimestamp == 0);
						assert(currentTrade.entryTimestamp > 0);
						assert(currentTrade.endingCapital == 0.0);
						assert(microseconds > currentTrade.entryTimestamp);
#endif
						trades.back().profitMargin = profitMargin;
						trades.back().exitTimestamp = microseconds;
						trades.back().endingCapital = capital;
					}
				} else if constexpr (is_same_v<T, comboVolFutures> ||
														 is_same_v<T, comboVolDeltaFutures> ||
														 is_same_v<T, comboAllVolFutures> ||
														 is_same_v<T, comboTrendFutures> ||
														 is_same_v<T, comboVolTrendFutures>) {
					int contracts = capital / (e.price * BTC_MARGIN_RATE) *
													combos[index].contractMultiplier;
					double profit;
					if (e.isLong)
						profit = contracts * (price - e.price) * 0.1;
					else
						profit = contracts * (e.price - price) * 0.1;
					profitMargin = (capital + profit) / capital;
					if (!isValidTradingHours || profitMargin >= precomputedTarget ||
							profitMargin <= precomputedStopLoss) {
						if (abs(price - lastPrice) >= FUTURES_PRICE_CHANGE_LIMIT)
							profitMargin = max(min(profitMargin, precomputedTarget),
																 precomputedStopLoss);
						capital *= profitMargin;
						capital -= 2 * contracts * BTC_COMMISSION_PER_CONTRACT;
						bool win = profitMargin >= 1.0;
						if (e.isLong) {
							lw += win;
							ll += !win;
						} else {
							sw += win;
							sl += !win;
						}
						e = (entry){0.0, false};
						tradesInInterval++;

						processOnlineAlgs(index, capital, microseconds, profitMargin, win,
															drawdownsVec, drawdownLengthsVec, lossStreaksVec,
															tradeDurationsVec, monthlyReturnsVec, winsVec,
															lossesVec, twapsVec);

#ifdef DEBUG
						assert(!trades.empty());

						auto &currentTrade = trades.back();
						assert(currentTrade.profitMargin == 0);
						assert(currentTrade.exitTimestamp == 0);
						assert(currentTrade.entryTimestamp > 0);
						assert(currentTrade.endingCapital == 0.0);
						assert(microseconds > currentTrade.entryTimestamp);
#endif
						trades.back().profitMargin = profitMargin;
						trades.back().exitTimestamp = microseconds;
						trades.back().endingCapital = capital;
					}
				} else if constexpr (is_same_v<T, comboVolExitPerps>) {
					if (e.isLong)
						profitMargin = price / e.price - 1;
					else
						profitMargin = 1 - price / e.price;
					profitMargin *= combos[index].leverageMultiplier;
					profitMargin += 1;
					if (!isValidTradingHours ||
							(e.isLong && buyVol <= combos[index].buyVolExitPercentile) ||
							(!e.isLong && sellVol <= combos[index].sellVolExitPercentile)) {
						capital *= profitMargin;
						bool win = profitMargin >= 1.0;
						if (e.isLong) {
							lw += win;
							ll += !win;
						} else {
							sw += win;
							sl += !win;
						}
						e = (entry){0.0, false};
						tradesInInterval++;

						processOnlineAlgs(index, capital, microseconds, profitMargin, win,
															drawdownsVec, drawdownLengthsVec, lossStreaksVec,
															tradeDurationsVec, monthlyReturnsVec, winsVec,
															lossesVec, twapsVec);

#ifdef DEBUG
						assert(!trades.empty());

						auto &currentTrade = trades.back();
						assert(currentTrade.profitMargin == 0);
						assert(currentTrade.exitTimestamp == 0);
						assert(currentTrade.entryTimestamp > 0);
						assert(currentTrade.endingCapital == 0.0);
						assert(microseconds > currentTrade.entryTimestamp);
#endif
						if constexpr (is_same_v<U, volExitDetailedTrade>) {
							trades.back().profitMargin = profitMargin;
							trades.back().exitTimestamp = microseconds;
							trades.back().exitData = {buyVol, sellVol};
							trades.back().endingCapital = capital;
						} else {
							cout << "Incorrect trade type" << endl;
						}
					}
				} else if constexpr (is_same_v<T, comboVolExitFutures>) {
					int contracts = capital / (e.price * BTC_MARGIN_RATE) *
													combos[index].contractMultiplier;
					double profit;
					if (e.isLong)
						profit = contracts * (price - e.price) * 0.1;
					else
						profit = contracts * (e.price - price) * 0.1;
					profitMargin = (capital + profit) / capital;
					if (!isValidTradingHours ||
							(e.isLong && buyVol <= combos[index].buyVolExitPercentile) ||
							(!e.isLong && sellVol <= combos[index].sellVolExitPercentile)) {
						if (abs(price - lastPrice) >= FUTURES_PRICE_CHANGE_LIMIT)
							profitMargin = max(min(profitMargin, precomputedTarget),
																 precomputedStopLoss);
						capital *= profitMargin;
						capital -= 2 * contracts * BTC_COMMISSION_PER_CONTRACT;
						bool win = profitMargin >= 1.0;
						if (e.isLong) {
							lw += win;
							ll += !win;
						} else {
							sw += win;
							sl += !win;
						}
						e = (entry){0.0, false};
						tradesInInterval++;

						processOnlineAlgs(index, capital, microseconds, profitMargin, win,
															drawdownsVec, drawdownLengthsVec, lossStreaksVec,
															tradeDurationsVec, monthlyReturnsVec, winsVec,
															lossesVec, twapsVec);

#ifdef DEBUG
						assert(!trades.empty());

						auto &currentTrade = trades.back();
						assert(currentTrade.profitMargin == 0);
						assert(currentTrade.exitTimestamp == 0);
						assert(currentTrade.entryTimestamp > 0);
						assert(currentTrade.endingCapital == 0.0);
						assert(microseconds > currentTrade.entryTimestamp);
#endif
						if constexpr (is_same_v<U, volExitDetailedTrade>) {
							trades.back().profitMargin = profitMargin;
							trades.back().exitTimestamp = microseconds;
							trades.back().exitData = {buyVol, sellVol};
							trades.back().endingCapital = capital;
						} else {
							cout << "Incorrect trade type" << endl;
						}
					}
				}
			};

			visit(visitor, combos, allTrades[index]);
		}

		auto visitor = [index, &e, &ls, &ss, &maxPrice, &minPrice, &increasing,
										microseconds, price, buyVol, sellVol, volDelta,
										precomputedLongEntryThreshold,
										precomputedShortEntryThreshold, isValidTradingHours,
										&tradeDurationsVec, &twapsVec, &allTrades](auto &combos) {
			using T = typename decay_t<decltype(combos)>::value_type;
			if constexpr (is_same_v<T, comboVolPerps> ||
										is_same_v<T, comboVolFutures>) {
				if (e.price == 0.0 && isValidTradingHours) {
					if (buyVol >= combos[index].buyVolPercentile) {
						e = (entry){price, true};
						ls += 1;
						tradeDurationsVec[index].entryTimestamp = microseconds;
						twapsVec[index].prevPrice = price;
						twapsVec[index].currentTimestamp = microseconds;
						twapsVec[index].currentNumer = 0;
						twapsVec[index].currentDenom = 0;
						volDetailedTrade currentTrade = {
								0, microseconds, 0, {buyVol, sellVol}, true, 0};
						vector<volDetailedTrade> &trades =
								get<vector<volDetailedTrade>>(allTrades[index]);
						trades.push_back(currentTrade);
					} else if (sellVol >= combos[index].sellVolPercentile) {
						e = (entry){price, false};
						ss += 1;
						tradeDurationsVec[index].entryTimestamp = microseconds;
						twapsVec[index].prevPrice = price;
						twapsVec[index].currentTimestamp = microseconds;
						twapsVec[index].currentNumer = 0;
						twapsVec[index].currentDenom = 0;
						volDetailedTrade currentTrade = {
								0, microseconds, 0, {buyVol, sellVol}, false, 0};
						vector<volDetailedTrade> &trades =
								get<vector<volDetailedTrade>>(allTrades[index]);
						trades.push_back(currentTrade);
					}
				}
			} else if constexpr (is_same_v<T, comboVolDeltaPerps> ||
													 is_same_v<T, comboVolDeltaFutures>) {
				if (e.price == 0.0 && isValidTradingHours) {
					if (volDelta >= combos[index].buyVolDeltaPercentile) {
						e = (entry){price, true};
						ls += 1;
						tradeDurationsVec[index].entryTimestamp = microseconds;
						twapsVec[index].prevPrice = price;
						twapsVec[index].currentTimestamp = microseconds;
						twapsVec[index].currentNumer = 0;
						twapsVec[index].currentDenom = 0;
						volDetailedTrade currentTrade = {
								0, microseconds, 0, {buyVol, sellVol}, true, 0};
						vector<volDetailedTrade> &trades =
								get<vector<volDetailedTrade>>(allTrades[index]);
						trades.push_back(currentTrade);
					} else if (volDelta <= combos[index].sellVolDeltaPercentile) {
						e = (entry){price, false};
						ss += 1;
						tradeDurationsVec[index].entryTimestamp = microseconds;
						twapsVec[index].prevPrice = price;
						twapsVec[index].currentTimestamp = microseconds;
						twapsVec[index].currentNumer = 0;
						twapsVec[index].currentDenom = 0;
						volDetailedTrade currentTrade = {
								0, microseconds, 0, {buyVol, sellVol}, false, 0};
						vector<volDetailedTrade> &trades =
								get<vector<volDetailedTrade>>(allTrades[index]);
						trades.push_back(currentTrade);
					}
				}
			} else if constexpr (is_same_v<T, comboVolExitPerps> ||
													 is_same_v<T, comboVolExitFutures>) {
				if (e.price == 0.0 && isValidTradingHours) {
					if (buyVol >= combos[index].buyVolPercentile) {
						e = (entry){price, true};
						ls += 1;
						tradeDurationsVec[index].entryTimestamp = microseconds;
						twapsVec[index].prevPrice = price;
						twapsVec[index].currentTimestamp = microseconds;
						twapsVec[index].currentNumer = 0;
						twapsVec[index].currentDenom = 0;
						volExitDetailedTrade currentTrade = {
								0, microseconds, 0, {buyVol, sellVol}, {}, true, 0};
						vector<volExitDetailedTrade> &trades =
								get<vector<volExitDetailedTrade>>(allTrades[index]);
						trades.push_back(currentTrade);
					} else if (sellVol >= combos[index].sellVolPercentile) {
						e = (entry){price, false};
						ss += 1;
						tradeDurationsVec[index].entryTimestamp = microseconds;
						twapsVec[index].prevPrice = price;
						twapsVec[index].currentTimestamp = microseconds;
						twapsVec[index].currentNumer = 0;
						twapsVec[index].currentDenom = 0;
						volExitDetailedTrade currentTrade = {
								0, microseconds, 0, {buyVol, sellVol}, {}, false, 0};
						vector<volExitDetailedTrade> &trades =
								get<vector<volExitDetailedTrade>>(allTrades[index]);
						trades.push_back(currentTrade);
					}
				}
			} else if constexpr (is_same_v<T, comboAllVolPerps> ||
													 is_same_v<T, comboAllVolFutures>) {
				if (e.price == 0.0 && isValidTradingHours) {
					if (buyVol >= combos[index].buyVolPercentile &&
							volDelta >= combos[index].buyVolDeltaPercentile) {
						e = (entry){price, true};
						ls += 1;
						tradeDurationsVec[index].entryTimestamp = microseconds;
						twapsVec[index].prevPrice = price;
						twapsVec[index].currentTimestamp = microseconds;
						twapsVec[index].currentNumer = 0;
						twapsVec[index].currentDenom = 0;
						volDetailedTrade currentTrade = {
								0, microseconds, 0, {buyVol, sellVol}, true, 0};
						vector<volDetailedTrade> &trades =
								get<vector<volDetailedTrade>>(allTrades[index]);
						trades.push_back(currentTrade);
					} else if (sellVol >= combos[index].sellVolPercentile &&
										 volDelta <= combos[index].sellVolDeltaPercentile) {
						e = (entry){price, false};
						ss += 1;
						tradeDurationsVec[index].entryTimestamp = microseconds;
						twapsVec[index].prevPrice = price;
						twapsVec[index].currentTimestamp = microseconds;
						twapsVec[index].currentNumer = 0;
						twapsVec[index].currentDenom = 0;
						volDetailedTrade currentTrade = {
								0, microseconds, 0, {buyVol, sellVol}, false, 0};
						vector<volDetailedTrade> &trades =
								get<vector<volDetailedTrade>>(allTrades[index]);
						trades.push_back(currentTrade);
					}
				}
			} else if constexpr (is_same_v<T, comboTrendPerps> ||
													 is_same_v<T, comboTrendFutures>) {
				if (increasing) {
					maxPrice = max(maxPrice, price);
					if (price / maxPrice <= precomputedShortEntryThreshold) {
						minPrice = 1000000;
						increasing = false;
						if (e.price == 0.0 && isValidTradingHours) {
							e = (entry){price, false};
							ss += 1;
							tradeDurationsVec[index].entryTimestamp = microseconds;
							twapsVec[index].prevPrice = price;
							twapsVec[index].currentTimestamp = microseconds;
							twapsVec[index].currentNumer = 0;
							twapsVec[index].currentDenom = 0;
							trendDetailedTrade currentTrade = {
									0,		 microseconds,
									0,		 {price, maxPrice, minPrice, increasing},
									false, 0};
							vector<trendDetailedTrade> &trades =
									get<vector<trendDetailedTrade>>(allTrades[index]);
							trades.push_back(currentTrade);
						}
					} else if (price / minPrice >= precomputedLongEntryThreshold) {
						if (e.price == 0.0 && isValidTradingHours) {
							e = (entry){price, true};
							ls += 1;
							tradeDurationsVec[index].entryTimestamp = microseconds;
							twapsVec[index].prevPrice = price;
							twapsVec[index].currentTimestamp = microseconds;
							twapsVec[index].currentNumer = 0;
							twapsVec[index].currentDenom = 0;
							trendDetailedTrade currentTrade = {
									0,		microseconds,
									0,		{price, maxPrice, minPrice, increasing},
									true, 0};
							vector<trendDetailedTrade> &trades =
									get<vector<trendDetailedTrade>>(allTrades[index]);
							trades.push_back(currentTrade);
						}
					}
				} else {
					minPrice = min(minPrice, price);
					if (price / minPrice >= precomputedLongEntryThreshold) {
						maxPrice = 1;
						increasing = true;
						if (e.price == 0.0 && isValidTradingHours) {
							e = (entry){price, true};
							ls += 1;
							tradeDurationsVec[index].entryTimestamp = microseconds;
							twapsVec[index].prevPrice = price;
							twapsVec[index].currentTimestamp = microseconds;
							twapsVec[index].currentNumer = 0;
							twapsVec[index].currentDenom = 0;
							trendDetailedTrade currentTrade = {
									0,		microseconds,
									0,		{price, maxPrice, minPrice, increasing},
									true, 0};
							vector<trendDetailedTrade> &trades =
									get<vector<trendDetailedTrade>>(allTrades[index]);
							trades.push_back(currentTrade);
						}
					} else if (price / maxPrice <= precomputedShortEntryThreshold) {
						if (e.price == 0.0 && isValidTradingHours) {
							e = (entry){price, false};
							ss += 1;
							tradeDurationsVec[index].entryTimestamp = microseconds;
							twapsVec[index].prevPrice = price;
							twapsVec[index].currentTimestamp = microseconds;
							twapsVec[index].currentNumer = 0;
							twapsVec[index].currentDenom = 0;
							trendDetailedTrade currentTrade = {
									0,		 microseconds,
									0,		 {price, maxPrice, minPrice, increasing},
									false, 0};
							vector<trendDetailedTrade> &trades =
									get<vector<trendDetailedTrade>>(allTrades[index]);
							trades.push_back(currentTrade);
						}
					}
				}
			} else if constexpr (is_same_v<T, comboVolTrendPerps> ||
													 is_same_v<T, comboVolTrendFutures>) {
				if (increasing) {
					maxPrice = max(maxPrice, price);
					if (price / maxPrice <= precomputedShortEntryThreshold) {
						minPrice = 1000000;
						increasing = false;
						if (e.price == 0.0 && isValidTradingHours &&
								sellVol >= combos[index].sellVolPercentile) {
							e = (entry){price, false};
							ss += 1;
							tradeDurationsVec[index].entryTimestamp = microseconds;
							twapsVec[index].prevPrice = price;
							twapsVec[index].currentTimestamp = microseconds;
							twapsVec[index].currentNumer = 0;
							twapsVec[index].currentDenom = 0;
							volTrendDetailedTrade currentTrade = {
									0,
									microseconds,
									0,
									{price, maxPrice, minPrice, increasing, buyVol, sellVol},
									false,
									0};
							vector<volTrendDetailedTrade> &trades =
									get<vector<volTrendDetailedTrade>>(allTrades[index]);
							trades.push_back(currentTrade);
						}
					} else if (price / minPrice >= precomputedLongEntryThreshold) {
						if (e.price == 0.0 && isValidTradingHours &&
								buyVol >= combos[index].buyVolPercentile) {
							e = (entry){price, true};
							ls += 1;
							tradeDurationsVec[index].entryTimestamp = microseconds;
							twapsVec[index].prevPrice = price;
							twapsVec[index].currentTimestamp = microseconds;
							twapsVec[index].currentNumer = 0;
							twapsVec[index].currentDenom = 0;
							volTrendDetailedTrade currentTrade = {
									0,
									microseconds,
									0,
									{price, maxPrice, minPrice, increasing, buyVol, sellVol},
									true,
									0};
							vector<volTrendDetailedTrade> &trades =
									get<vector<volTrendDetailedTrade>>(allTrades[index]);
							trades.push_back(currentTrade);
						}
					}
				} else {
					minPrice = min(minPrice, price);
					if (price / minPrice >= precomputedLongEntryThreshold) {
						maxPrice = 1;
						increasing = true;
						if (e.price == 0.0 && isValidTradingHours &&
								buyVol >= combos[index].buyVolPercentile) {
							e = (entry){price, true};
							ls += 1;
							tradeDurationsVec[index].entryTimestamp = microseconds;
							twapsVec[index].prevPrice = price;
							twapsVec[index].currentTimestamp = microseconds;
							twapsVec[index].currentNumer = 0;
							twapsVec[index].currentDenom = 0;
							volTrendDetailedTrade currentTrade = {
									0,
									microseconds,
									0,
									{price, maxPrice, minPrice, increasing, buyVol, sellVol},
									true,
									0};
							vector<volTrendDetailedTrade> &trades =
									get<vector<volTrendDetailedTrade>>(allTrades[index]);
							trades.push_back(currentTrade);
						}
					} else if (price / maxPrice <= precomputedShortEntryThreshold) {
						if (e.price == 0.0 && isValidTradingHours &&
								sellVol >= combos[index].sellVolPercentile) {
							e = (entry){price, false};
							ss += 1;
							tradeDurationsVec[index].entryTimestamp = microseconds;
							twapsVec[index].prevPrice = price;
							twapsVec[index].currentTimestamp = microseconds;
							twapsVec[index].currentNumer = 0;
							twapsVec[index].currentDenom = 0;
							volTrendDetailedTrade currentTrade = {
									0,
									microseconds,
									0,
									{price, maxPrice, minPrice, increasing, buyVol, sellVol},
									false,
									0};
							vector<volTrendDetailedTrade> &trades =
									get<vector<volTrendDetailedTrade>>(allTrades[index]);
							trades.push_back(currentTrade);
						}
					}
				}
			}
		};

		visit(visitor, combos);

		lastPrice = price;
	}

	entriesVec[index] = e;
	tradeRecordsVec[index] = (tradeRecord){capital, ss, sw, sl, ls, lw, ll};
	positionDatasVec[index] = (positionData){maxPrice, minPrice, increasing};

	/*
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
					allTrades[index].back().endingCapital = capital;
					*/
}

int processTrades(vector<tradeWithoutDate> &tradesWithoutDates,
									vector<indicators> &inds, vector<timeWindow> &tws,
									const size_t numCombos, const ComboVec &comboVec,
									vector<entry> &entriesVec,
									vector<tradeRecord> &tradeRecordsVec,
									vector<positionData> &positionDatasVec,
									vector<cl_int> &numTradesInIntervalVec) {
	size_t currIdx = newStart;

	int maxTradesPerInterval = 0;

	while (true) {
		size_t currSize =
				min((size_t)INCREMENT, tradesWithoutDates.size() - currIdx);

		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << " (" << globalIdx << "-" << globalIdx + currSize - 1 << ")" << endl;
		ThreadPool pool(thread::hardware_concurrency());
		for (size_t i = 0; i < numCombos; ++i) {
			pool.enqueue([=, &tradesWithoutDates, &inds, &comboVec, &entriesVec,
										&tradeRecordsVec, &positionDatasVec,
										&numTradesInIntervalVec]() {
				/*
				{
					lock_guard<mutex> lock(coutMutex);
					cout << "Task " << i << " running in thread " <<
				std::this_thread::get_id() << endl;
				}
				*/
				performWork(i, currIdx, currSize, tradesWithoutDates, inds, comboVec,
										entriesVec, tradeRecordsVec, positionDatasVec,
										numTradesInIntervalVec);
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

void processTradesWithOnlineAlgs(
		vector<tradeWithoutDate> &tradesWithoutDates, vector<indicators> &inds,
		vector<timeWindow> &tws, const size_t numCombos, const ComboVec &comboVec,
		vector<entry> &entriesVec, vector<tradeRecord> &tradeRecordsVec,
		vector<positionData> &positionDatasVec, vector<drawdowns> &drawdownsVec,
		vector<drawdownLengths> &drawdownLengthsVec,
		vector<lossStreaks> &lossStreaksVec,
		vector<tradeDurations> &tradeDurationsVec,
		vector<monthlyReturns> &monthlyReturnsVec, vector<wins> &winsVec,
		vector<losses> &lossesVec, vector<twap> &twapsVec,
		vector<DetailedTradeVec> &allTrades) {
	size_t currIdx = newStart;

	while (true) {
		size_t currSize =
				min((size_t)INCREMENT, tradesWithoutDates.size() - currIdx);

		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << " (" << globalIdx << "-" << globalIdx + currSize - 1 << ")" << endl;
		ThreadPool pool(thread::hardware_concurrency());
		for (size_t i = 0; i < numCombos; ++i) {
			pool.enqueue([=, &tradesWithoutDates, &inds, &comboVec, &entriesVec,
										&tradeRecordsVec, &positionDatasVec, &drawdownsVec,
										&drawdownLengthsVec, &lossStreaksVec, &tradeDurationsVec,
										&monthlyReturnsVec, &winsVec, &lossesVec, &twapsVec,
										&allTrades]() {
				/*
				{
					lock_guard<mutex> lock(coutMutex);
					cout << "Task " << i << " running in thread " <<
				std::this_thread::get_id() << endl;
				}
				*/
				performWork(i, currIdx, currSize, tradesWithoutDates, inds, comboVec,
										entriesVec, tradeRecordsVec, positionDatasVec, drawdownsVec,
										drawdownLengthsVec, lossStreaksVec, tradeDurationsVec,
										monthlyReturnsVec, winsVec, lossesVec, twapsVec, allTrades);
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

int main(int argc, char *argv[]) {
	bool writeResults = false, listTrades = false, futures = false;

	bool isBTC;

	Mode mode;
	string modeString;

	int opt;
	while ((opt = getopt(argc, argv, ":wlft:")) != -1) {
		switch (opt) {
		case 'w':
			writeResults = true;
			cout << "Enabled writing results" << endl;
			break;
		case 'l':
			listTrades = true;
			cout << "Enabled listing trades" << endl;
			break;
		case 'f':
			futures = true;
			cout << "Enabled futures" << endl;
			break;
		case 't': {
			auto it = modeMap.find(optarg);
			if (it == modeMap.end()) {
				cout << "Invalid mode: " << optarg << endl;
				exit(EXIT_FAILURE);
			}
			mode = it->second;
			modeString = optarg;
			modeString[0] =
					static_cast<char>(toupper(static_cast<unsigned char>(modeString[0])));
			cout << "Using mode: " << optarg << endl;
			break;
		}
		case ':':
			cout << "Option -" << static_cast<char>(optopt) << " requires an argument"
					 << endl;
			exit(EXIT_FAILURE);
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

	vector<double> entryThresholds(20);
	x = 0;
	generate(entryThresholds.begin(), entryThresholds.end(),
					 [x]() mutable { return x += 0.05; });

	vector<double> contractMultipliers(10);
	x = 0;
	generate(contractMultipliers.begin(), contractMultipliers.end(),
					 [x]() mutable { return x += 0.1; });

	vector<double> leverageMultipliers(6);
	x = 0;
	generate(leverageMultipliers.begin(), leverageMultipliers.end(),
					 [x]() mutable { return x += 1; });

	int y = 0;
	vector<int> periodsAfterOpen(11);
	generate(periodsAfterOpen.begin(), periodsAfterOpen.end(),
					 [y]() mutable { return y += 2; });

	vector<vector<double>> buyVolPercentiles;
	vector<vector<double>> sellVolPercentiles;
	vector<vector<double>> deltaVolPercentiles;

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

	if (isBTC)
		myFile.open("deltaPercentilesDollarsBTC");
	else
		myFile.open("deltaPercentilesDollarsETH");
	if (myFile.is_open()) {
		string line;
		while (getline(myFile, line)) {
			vector<string> splits = split(line);
			vector<double> rowPercentiles;
			for (string s : splits) {
				rowPercentiles.emplace_back(stod(s));
			}
			deltaVolPercentiles.emplace_back(rowPercentiles);
		}
		myFile.close();
	}

	vector<vector<long long>> indWindows(NUM_WINDOWS);
	for (int i = 0; i < NUM_WINDOWS; i++) {
		indWindows[i] = vector<long long>(1, windows[i]);
	}

	ComboVec comboVec, tempCombos;
	size_t comboVecSize;
	size_t numCombos = indicesToRun.size();

	switch (mode) {
	case Mode::Volume:
		if (futures) {
			comboVec = vector<comboVolFutures>{};
			tempCombos = vector<comboVolFutures>{};
		} else {
			comboVec = vector<comboVolPerps>{};
			tempCombos = vector<comboVolPerps>{};
		}
		break;
	case Mode::VolumeDelta:
		if (futures) {
			comboVec = vector<comboVolDeltaFutures>{};
			tempCombos = vector<comboVolDeltaFutures>{};
		} else {
			comboVec = vector<comboVolDeltaPerps>{};
			tempCombos = vector<comboVolDeltaPerps>{};
		}
		break;
	case Mode::VolumeExit:
		if (futures) {
			comboVec = vector<comboVolExitFutures>{};
			tempCombos = vector<comboVolExitFutures>{};
		} else {
			comboVec = vector<comboVolExitPerps>{};
			tempCombos = vector<comboVolExitPerps>{};
		}
		break;
	case Mode::AllVolume:
		if (futures) {
			comboVec = vector<comboAllVolFutures>{};
			tempCombos = vector<comboAllVolFutures>{};
		} else {
			comboVec = vector<comboAllVolPerps>{};
			tempCombos = vector<comboAllVolPerps>{};
		}
		break;
	case Mode::Trend:
		if (futures) {
			comboVec = vector<comboTrendFutures>{};
			tempCombos = vector<comboTrendFutures>{};
		} else {
			comboVec = vector<comboTrendPerps>{};
			tempCombos = vector<comboTrendPerps>{};
		}
		break;
	case Mode::VolumeTrend:
		if (futures) {
			comboVec = vector<comboVolTrendFutures>{};
			tempCombos = vector<comboVolTrendFutures>{};
		} else {
			comboVec = vector<comboVolTrendPerps>{};
			tempCombos = vector<comboVolTrendPerps>{};
		}
		break;
	}

	auto visitor = [&indWindows, &buyVolPercentiles, &sellVolPercentiles,
									&stopLosses, &targets, &deltaVolPercentiles, &entryThresholds,
									&comboVecSize, &leverageMultipliers, &contractMultipliers,
									&periodsAfterOpen](auto &tempCombos) {
		using T = typename decay_t<decltype(tempCombos)>::value_type;
		if constexpr (is_same_v<T, comboVolPerps>) {
			auto firstCombo = cartesian_product(
					indWindows[0], buyVolPercentiles[0], sellVolPercentiles[0],
					stopLosses, targets, leverageMultipliers, periodsAfterOpen);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i],
						stopLosses, targets, leverageMultipliers, periodsAfterOpen));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboVolPerps c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
														 get<2>(combos[i][j]), get<3>(combos[i][j]),
														 get<4>(combos[i][j]), get<5>(combos[i][j]),
														 get<6>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1)
						tempCombos.emplace_back(c);
				}
			}
			comboVecSize = indicesToRun.size() * sizeof(comboVolPerps);
		} else if constexpr (is_same_v<T, comboVolFutures>) {
			auto firstCombo = cartesian_product(
					indWindows[0], buyVolPercentiles[0], sellVolPercentiles[0],
					stopLosses, targets, contractMultipliers, periodsAfterOpen);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i],
						stopLosses, targets, contractMultipliers, periodsAfterOpen));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboVolFutures c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
															 get<2>(combos[i][j]), get<3>(combos[i][j]),
															 get<4>(combos[i][j]), get<5>(combos[i][j]),
															 get<6>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1)
						tempCombos.emplace_back(c);
				}
			}
			comboVecSize = indicesToRun.size() * sizeof(comboVolFutures);
		} else if constexpr (is_same_v<T, comboVolDeltaPerps>) {
			auto firstCombo = cartesian_product(
					indWindows[0], deltaVolPercentiles[0], deltaVolPercentiles[0],
					stopLosses, targets, leverageMultipliers, periodsAfterOpen);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], deltaVolPercentiles[i], deltaVolPercentiles[i],
						stopLosses, targets, leverageMultipliers, periodsAfterOpen));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboVolDeltaPerps c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
																	get<2>(combos[i][j]), get<3>(combos[i][j]),
																	get<4>(combos[i][j]), get<5>(combos[i][j]),
																	get<6>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1 &&
							c.buyVolDeltaPercentile >= c.sellVolDeltaPercentile)
						tempCombos.emplace_back(c);
				}
			}
			comboVecSize = indicesToRun.size() * sizeof(comboVolDeltaPerps);
		} else if constexpr (is_same_v<T, comboVolDeltaFutures>) {
			auto firstCombo = cartesian_product(
					indWindows[0], deltaVolPercentiles[0], deltaVolPercentiles[0],
					stopLosses, targets, contractMultipliers, periodsAfterOpen);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], deltaVolPercentiles[i], deltaVolPercentiles[i],
						stopLosses, targets, contractMultipliers, periodsAfterOpen));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboVolDeltaFutures c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
																		get<2>(combos[i][j]), get<3>(combos[i][j]),
																		get<4>(combos[i][j]), get<5>(combos[i][j]),
																		get<6>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1 &&
							c.buyVolDeltaPercentile >= c.sellVolDeltaPercentile)
						tempCombos.emplace_back(c);
				}
			}
			comboVecSize = indicesToRun.size() * sizeof(comboVolDeltaFutures);
		} else if constexpr (is_same_v<T, comboVolExitPerps>) {
			auto firstCombo = cartesian_product(
					indWindows[0], buyVolPercentiles[0], sellVolPercentiles[0],
					buyVolPercentiles[0], sellVolPercentiles[0], leverageMultipliers,
					periodsAfterOpen);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i],
						buyVolPercentiles[i], sellVolPercentiles[i], leverageMultipliers,
						periodsAfterOpen));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboVolExitPerps c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
																 get<2>(combos[i][j]), get<3>(combos[i][j]),
																 get<4>(combos[i][j]), get<5>(combos[i][j]),
																 get<6>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.buyVolPercentile > c.buyVolExitPercentile &&
							c.sellVolPercentile > c.sellVolExitPercentile)
						tempCombos.emplace_back(c);
				}
			}
			comboVecSize = indicesToRun.size() * sizeof(comboVolExitPerps);
		} else if constexpr (is_same_v<T, comboVolExitFutures>) {
			auto firstCombo = cartesian_product(
					indWindows[0], buyVolPercentiles[0], sellVolPercentiles[0],
					buyVolPercentiles[0], sellVolPercentiles[0], contractMultipliers,
					periodsAfterOpen);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i],
						buyVolPercentiles[i], sellVolPercentiles[i], contractMultipliers,
						periodsAfterOpen));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboVolExitFutures c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
																	 get<2>(combos[i][j]), get<3>(combos[i][j]),
																	 get<4>(combos[i][j]), get<5>(combos[i][j]),
																	 get<6>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.buyVolPercentile > c.buyVolExitPercentile &&
							c.sellVolPercentile > c.sellVolExitPercentile)
						tempCombos.emplace_back(c);
				}
			}
			comboVecSize = indicesToRun.size() * sizeof(comboVolExitFutures);
		} else if constexpr (is_same_v<T, comboAllVolPerps>) {
			auto firstCombo = cartesian_product(
					indWindows[0], buyVolPercentiles[0], sellVolPercentiles[0],
					deltaVolPercentiles[0], deltaVolPercentiles[0], stopLosses, targets,
					leverageMultipliers, periodsAfterOpen);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i],
						deltaVolPercentiles[i], deltaVolPercentiles[i], stopLosses, targets,
						leverageMultipliers, periodsAfterOpen));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboAllVolPerps c = {
							get<0>(combos[i][j]), get<1>(combos[i][j]), get<2>(combos[i][j]),
							get<3>(combos[i][j]), get<4>(combos[i][j]), get<5>(combos[i][j]),
							get<6>(combos[i][j]), get<7>(combos[i][j]), get<8>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1 &&
							c.buyVolPercentile >= c.buyVolDeltaPercentile &&
							c.sellVolPercentile >= -c.sellVolDeltaPercentile &&
							c.buyVolDeltaPercentile >= 0 && c.sellVolDeltaPercentile <= 0)
						tempCombos.emplace_back(c);
				}
			}
			comboVecSize = indicesToRun.size() * sizeof(comboAllVolPerps);
		} else if constexpr (is_same_v<T, comboAllVolFutures>) {
			auto firstCombo = cartesian_product(
					indWindows[0], buyVolPercentiles[0], sellVolPercentiles[0],
					deltaVolPercentiles[0], deltaVolPercentiles[0], stopLosses, targets,
					contractMultipliers, periodsAfterOpen);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i],
						deltaVolPercentiles[i], deltaVolPercentiles[i], stopLosses, targets,
						contractMultipliers, periodsAfterOpen));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboAllVolFutures c = {
							get<0>(combos[i][j]), get<1>(combos[i][j]), get<2>(combos[i][j]),
							get<3>(combos[i][j]), get<4>(combos[i][j]), get<5>(combos[i][j]),
							get<6>(combos[i][j]), get<7>(combos[i][j]), get<8>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1 &&
							c.buyVolPercentile >= c.buyVolDeltaPercentile &&
							c.sellVolPercentile >= -c.sellVolDeltaPercentile &&
							c.buyVolDeltaPercentile >= 0 && c.sellVolDeltaPercentile <= 0)
						tempCombos.emplace_back(c);
				}
			}
			comboVecSize = indicesToRun.size() * sizeof(comboAllVolFutures);
		} else if constexpr (is_same_v<T, comboTrendPerps>) {
			auto combos = cartesian_product(entryThresholds, stopLosses, targets,
																			leverageMultipliers, periodsAfterOpen);

			for (unsigned int j = 0; j < combos.size(); j++) {
				comboTrendPerps c = {get<0>(combos[j]), get<1>(combos[j]),
														 get<2>(combos[j]), get<3>(combos[j]),
														 get<4>(combos[j])};
				// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
				// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
				if (c.stopLoss < c.target + 1.1)
					tempCombos.emplace_back(c);
			}
			comboVecSize = indicesToRun.size() * sizeof(comboTrendPerps);
		} else if constexpr (is_same_v<T, comboTrendFutures>) {
			auto combos = cartesian_product(entryThresholds, stopLosses, targets,
																			contractMultipliers, periodsAfterOpen);

			for (unsigned int j = 0; j < combos.size(); j++) {
				comboTrendFutures c = {get<0>(combos[j]), get<1>(combos[j]),
															 get<2>(combos[j]), get<3>(combos[j]),
															 get<4>(combos[j])};
				// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
				// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
				if (c.stopLoss < c.target + 1.1)
					tempCombos.emplace_back(c);
			}
			comboVecSize = indicesToRun.size() * sizeof(comboTrendFutures);
		} else if constexpr (is_same_v<T, comboVolTrendPerps>) {
			auto firstCombo =
					cartesian_product(indWindows[0], buyVolPercentiles[0],
														sellVolPercentiles[0], entryThresholds, stopLosses,
														targets, leverageMultipliers, periodsAfterOpen);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i],
						entryThresholds, stopLosses, targets, leverageMultipliers,
						periodsAfterOpen));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboVolTrendPerps c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
																	get<2>(combos[i][j]), get<3>(combos[i][j]),
																	get<4>(combos[i][j]), get<5>(combos[i][j]),
																	get<6>(combos[i][j]), get<7>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1)
						tempCombos.emplace_back(c);
				}
			}
			comboVecSize = indicesToRun.size() * sizeof(comboVolTrendPerps);
		} else if constexpr (is_same_v<T, comboVolTrendFutures>) {
			auto firstCombo =
					cartesian_product(indWindows[0], buyVolPercentiles[0],
														sellVolPercentiles[0], entryThresholds, stopLosses,
														targets, contractMultipliers, periodsAfterOpen);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i],
						entryThresholds, stopLosses, targets, contractMultipliers,
						periodsAfterOpen));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboVolTrendFutures c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
																		get<2>(combos[i][j]), get<3>(combos[i][j]),
																		get<4>(combos[i][j]), get<5>(combos[i][j]),
																		get<6>(combos[i][j]), get<7>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1)
						tempCombos.emplace_back(c);
				}
			}
			comboVecSize = indicesToRun.size() * sizeof(comboVolTrendFutures);
		}
	};

	visit(visitor, tempCombos);

	auto nextVisitor = [](auto &tempCombos, auto &comboVec) {
		using T1 = typename decay_t<decltype(tempCombos)>::value_type;
		using T2 = typename decay_t<decltype(comboVec)>::value_type;

		if constexpr (is_same_v<T1, T2>) {
			for (size_t i : indicesToRun) {
				comboVec.push_back(tempCombos[i]);
			}
		} else {
			cout << "Incorrect combo type" << endl;
		}
		tempCombos.clear();

#ifdef DEBUG
		assert(comboVec.size() == indicesToRun.size());
#endif
	};

	visit(nextVisitor, tempCombos, comboVec);

	vector<positionData> positionDatasVec(numCombos, {0.0, 0.0, 0});
	size_t positionDatasVecSize = 0;

	switch (mode) {
	case Mode::Volume:
	case Mode::VolumeDelta:
	case Mode::VolumeExit:
	case Mode::AllVolume:
		break;
	case Mode::Trend:
	case Mode::VolumeTrend:
		positionDatasVec = vector<positionData>(numCombos, {0.0, 0.0, 0});
		positionDatasVecSize = numCombos * sizeof(positionData);
		break;
	}

	bool initializedPositions = false;

	vector<entry> entriesVec(numCombos, {0.0, 0});
	size_t entriesVecSize = entriesVec.size() * sizeof(entry);
	vector<tradeRecord> tradeRecordsVec(numCombos,
																			{STARTING_CAPITAL, 0, 0, 0, 0, 0, 0});
	size_t tradeRecordsVecSize = tradeRecordsVec.size() * sizeof(tradeRecord);

	int maxTradesPerInterval = 0;

	vector<drawdowns> drawdownsVec;
	vector<drawdownLengths> drawdownLengthsVec;
	vector<lossStreaks> lossStreaksVec;
	vector<tradeDurations> tradeDurationsVec;
	vector<monthlyReturns> monthlyReturnsVec;
	vector<wins> winsVec;
	vector<losses> lossesVec;
	vector<twap> twapsVec;

	vector<cl_int> numTradesInIntervalVec;

	cout << "Number of combinations: " << numCombos << endl;
	cout << "Size of combinations: " << comboVecSize << endl;
	// cout << "Size of trades: " << inputTradesSize << endl;
	// cout << "Size of indicators: " << indicatorsSize << endl;
	cout << "Size of entries: " << entriesVecSize << endl;
	cout << "Size of trade records: " << tradeRecordsVecSize << endl;
	cout << "Size of position datas: " << positionDatasVecSize << endl;
	// size_t totalSize = comboVecSize + inputTradesSize + indicatorsSize +
	// 									 entriesVecSize + tradeRecordsVecSize;
	size_t totalSize = comboVecSize + entriesVecSize + tradeRecordsVecSize +
										 positionDatasVecSize;
	if (listTrades) {
		drawdownsVec = vector<drawdowns>(
				numCombos, {0, 1.0, 0, 0, STARTING_CAPITAL, 1000000000, 0});
		size_t drawdownsVecSize = drawdownsVec.size() * sizeof(drawdowns);
		cout << "Size of drawdowns: " << drawdownsVecSize << endl;

		drawdownLengthsVec = vector<drawdownLengths>(numCombos, drawdownLengths{});
		size_t drawdownLengthsVecSize =
				drawdownLengthsVec.size() * sizeof(drawdownLengths);
		cout << "Size of drawdown lengths: " << drawdownLengthsVecSize << endl;

		lossStreaksVec = vector<lossStreaks>(numCombos, lossStreaks{});
		size_t lossStreaksVecSize = lossStreaksVec.size() * sizeof(lossStreaks);
		cout << "Size of loss streaks: " << lossStreaksVecSize << endl;

		tradeDurationsVec = vector<tradeDurations>(numCombos, tradeDurations{});
		size_t tradeDurationsVecSize =
				tradeDurationsVec.size() * sizeof(tradeDurations);
		cout << "Size of trade durations: " << tradeDurationsVecSize << endl;

		monthlyReturnsVec = vector<monthlyReturns>(numCombos, monthlyReturns{});
		size_t monthlyReturnsVecSize =
				monthlyReturnsVec.size() * sizeof(monthlyReturns);
		cout << "Size of monthly returns: " << monthlyReturnsVecSize << endl;

		winsVec = vector<wins>(numCombos, {0, 0, 0, 0, 1000000});
		size_t winsVecSize = winsVec.size() * sizeof(wins);
		cout << "Size of wins: " << winsVecSize << endl;

		lossesVec = vector<losses>(numCombos, {0, 0, 0, 1000000, 0});
		size_t lossesVecSize = lossesVec.size() * sizeof(losses);
		cout << "Size of losses: " << lossesVecSize << endl;

		twapsVec = vector<twap>(numCombos, {0, 0, 0, 0, 0, 0, 0, 0});
		size_t twapsVecSize = twapsVec.size() * sizeof(twap);
		cout << "Size of twaps: " << twapsVecSize << endl;

		totalSize += drawdownsVecSize + drawdownLengthsVecSize +
								 lossStreaksVecSize + tradeDurationsVecSize +
								 monthlyReturnsVecSize + winsVecSize + lossesVecSize +
								 twapsVecSize;
		cout << "Total size: " << totalSize << endl;
		cout << "Total size: " << (double)totalSize / (double)(1024 * 1024 * 1024)
				 << " GiB" << endl;
	} else {
		numTradesInIntervalVec = vector<cl_int>(numCombos, 0);
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

	vector<DetailedTradeVec> allTrades;
	vector<perfMetrics> allPerfMetrics;
	if (listTrades) {
		auto visitor = [numCombos, &allTrades](auto &combos) {
			using T = typename decay_t<decltype(combos)>::value_type;
			if constexpr (is_same_v<T, comboVolPerps> ||
										is_same_v<T, comboVolFutures> ||
										is_same_v<T, comboVolDeltaPerps> ||
										is_same_v<T, comboVolDeltaFutures> ||
										is_same_v<T, comboAllVolPerps> ||
										is_same_v<T, comboAllVolFutures>) {
				allTrades.resize(numCombos, vector<volDetailedTrade>{});
			} else if constexpr (is_same_v<T, comboVolExitPerps> ||
													 is_same_v<T, comboVolExitFutures>) {
				allTrades.resize(numCombos, vector<volExitDetailedTrade>{});
			} else if constexpr (is_same_v<T, comboTrendPerps> ||
													 is_same_v<T, comboTrendFutures>) {
				allTrades.resize(numCombos, vector<trendDetailedTrade>{});
			} else if constexpr (is_same_v<T, comboVolTrendPerps> ||
													 is_same_v<T, comboVolTrendFutures>) {
				allTrades.resize(numCombos, vector<volTrendDetailedTrade>{});
			}
		};

		visit(visitor, comboVec);
		allPerfMetrics.resize(numCombos);
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
					if (mode == Mode::VolumeTrend || mode == Mode::Trend) {
						if (positionDatasVec[0].maxPrice == 0.0 &&
								positionDatasVec[0].minPrice == 0.0) {
							for (auto &pos : positionDatasVec) {
								pos.maxPrice = t.price;
								pos.minPrice = t.price;
							}
						} else {
#ifdef DEBUG
							assert(false);
#endif
						}
					}
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
						processTradesWithOnlineAlgs(
								tradesWithoutDates, indicators, tws, numCombos, comboVec,
								entriesVec, tradeRecordsVec, positionDatasVec, drawdownsVec,
								drawdownLengthsVec, lossStreaksVec, tradeDurationsVec,
								monthlyReturnsVec, winsVec, lossesVec, twapsVec, allTrades);
					else
						maxTradesPerInterval = max(
								maxTradesPerInterval,
								// processTrades(queue, volKernel, tradesWithoutDates,
								// comboVec, 							inputTrades, inputSize,
								// twBetweenRunData, 							positionDatas,
								// numTradesInIntervalBuf));
								processTrades(tradesWithoutDates, indicators, tws, numCombos,
															comboVec, entriesVec, tradeRecordsVec,
															positionDatasVec, numTradesInIntervalVec));
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
						tradesWithoutDates, indicators, tws, numCombos, comboVec,
						entriesVec, tradeRecordsVec, positionDatasVec, drawdownsVec,
						drawdownLengthsVec, lossStreaksVec, tradeDurationsVec,
						monthlyReturnsVec, winsVec, lossesVec, twapsVec, allTrades);
			else
				maxTradesPerInterval =
						max(maxTradesPerInterval,
								// processTrades(queue, volKernel, tradesWithoutDates, comboVec,
								// 							inputTrades, inputSize, twBetweenRunData,
								// 							positionDatas, numTradesInIntervalBuf));
								processTrades(tradesWithoutDates, indicators, tws, numCombos,
															comboVec, entriesVec, tradeRecordsVec,
															positionDatasVec, numTradesInIntervalVec));
		}
	}

	for (size_t i = 0; i < tradeRecordsVec.size(); i++) {
		tradeRecordsVec[i].capital =
				capitalToAnnualizedReturn(tradeRecordsVec[i].capital / STARTING_CAPITAL,
																	firstTimestamp, lastTimestamp);
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

	if (futures)
		resultsFilename += "Futures";
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
			for (size_t i = 0; i < numCombos; i++) {
				outputMetrics(outFile, i, comboVec, tradeRecordsVec, allPerfMetrics,
											drawdownsVec, drawdownLengthsVec, lossStreaksVec,
											tradeDurationsVec, winsVec, lossesVec, twapsVec,
											allTrades, listTrades);
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

	for (size_t i = 0; i < numCombos; i++) {
		outputMetrics(cout, i, comboVec, tradeRecordsVec, allPerfMetrics,
									drawdownsVec, drawdownLengthsVec, lossStreaksVec,
									tradeDurationsVec, winsVec, lossesVec, twapsVec, allTrades,
									listTrades);
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
