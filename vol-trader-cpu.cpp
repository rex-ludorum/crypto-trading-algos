#include "ThreadPool.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <ranges>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

using std::accumulate;
using std::array;
using std::boolalpha;
using std::defaultfloat;
using std::fixed;
using std::format;
using std::generate;
using std::get;
using std::get_time;
using std::getline;
using std::gmtime;
using std::ifstream;
using std::inner_product;
using std::istream_iterator;
using std::istringstream;
using std::lock_guard;
using std::max;
using std::max_element;
using std::min;
using std::min_element;
using std::minus;
using std::mktime;
using std::mutex;
using std::nullopt;
using std::ofstream;
using std::optional;
using std::ostream;
using std::ostringstream;
using std::put_time;
using std::setprecision;
using std::sqrt;
using std::stod;
using std::stoi;
using std::string;
using std::thread;
using std::time_t;
using std::tm;
using std::to_string;
using std::transform;
using std::tuple;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::system_clock;
using std::ranges::views::cartesian_product;

using std::cerr;
using std::cout;
using std::endl;

#define ONE_HOUR_MICROSECONDS 3600000000
#define FIFTEEN_MINUTES_MICROSECONDS 900000000
#define ONE_MINUTE_MICROSECONDS 60000000
#define ONE_YEAR_MICROSECONDS 31536000000000

#define MICROSECONDS_IN_HOUR 3600000000
#define MICROSECONDS_IN_DAY 86400000000
#define MICROSECONDS_IN_WEEK 604800000000
#define CME_CLOSE 79200000000
#define CME_OPEN 82800000000
#define CME_CLOSE_FRIDAY 165600000000
#define CME_OPEN_SUNDAY 342000000000

#define NUM_WINDOWS 4

#define MAX_TOTAL_TRADES 3189

#define INCREMENT 10000
#define TRADE_CHUNK 50000000

#define PERCENTILE_CEILING 30
#define PERCENTILE_FLOOR 6

#define MARCH_1_1972_IN_SECONDS 68256000
#define DAYS_IN_LEAP_YEAR_CYCLE 1461
#define SECONDS_IN_DAY 86400

#define RISK_FREE_RATE 1.01

constexpr array<int, 12> daysInMonths = {31, 30, 31, 30, 31, 31,
																				 30, 31, 30, 31, 31, 28};

static size_t newStart;

mutex coutMutex;

vector<string> split(string const &input) {
	istringstream buffer(input);
	vector<string> ret((istream_iterator<string>(buffer)),
										 istream_iterator<string>());
	return ret;
}

vector<string> splitByComma(const string &input) {
	istringstream buffer(input);
	vector<string> ret;
	string token;

	while (getline(buffer, token, ',')) {
		ret.push_back(token);
	}

	return ret;
}

long long getTsOfNextMonth(long long ts) {
	int timestamp = ts / 1000000;
	int days = (timestamp - MARCH_1_1972_IN_SECONDS) / SECONDS_IN_DAY;
	int daysInCurrentCycle = days % DAYS_IN_LEAP_YEAR_CYCLE;
	int daysInCurrentYear = daysInCurrentCycle % 365;
	int daysInCurrentYearCopy = daysInCurrentYear;
	bool isLeapYear = daysInCurrentCycle / 365 >= 3;

	int daysUntilNextMonthFromStartOfCurrentYear = 0;
	if (daysInCurrentCycle != 1460) {
		int monthIdx = 0;
		while (daysInCurrentYear >= 0) {
			daysUntilNextMonthFromStartOfCurrentYear += daysInMonths[monthIdx];
			daysInCurrentYear -= daysInMonths[monthIdx++];
		}
		if (isLeapYear && monthIdx == 12)
			daysUntilNextMonthFromStartOfCurrentYear += 1;
	} else
		daysUntilNextMonthFromStartOfCurrentYear = 1;

	long long daysUntilNextMonthFromMarchOne =
			days - daysInCurrentYearCopy + daysUntilNextMonthFromStartOfCurrentYear;
	return (daysUntilNextMonthFromMarchOne * SECONDS_IN_DAY +
					MARCH_1_1972_IN_SECONDS) *
				 1000000;
}

int isDst(long long ts) {
	int timestamp = ts / 1000000;
	int days = (timestamp - MARCH_1_1972_IN_SECONDS) / SECONDS_IN_DAY;
	int daysInCurrentCycle = days % DAYS_IN_LEAP_YEAR_CYCLE;
	int daysInCurrentYear = daysInCurrentCycle % 365;

	int timeInCurrentDay = timestamp % SECONDS_IN_DAY;

	int marchFirstDayOfWeekInCurrentYear = (days - daysInCurrentYear) % 7;

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

struct trade {
	int tradeId;
	long long timestamp;
	double price;
	double qty;
	bool isBuyerMaker;
	string date;
};

struct __attribute__((packed)) tradeWithoutDate {
	long long timestamp;
	double price;
	double qty;
	bool isBuyerMaker;
};

struct __attribute__((packed)) indicators {
	array<double, 2 * NUM_WINDOWS> vols;
};

struct timeWindow {
	size_t tradeIdx;
	long long timestamp;
};

struct __attribute__((packed)) combo {
	long long window;
	double buyVolPercentile;
	double sellVolPercentile;
	double stopLoss;
	double target;
};

struct __attribute__((packed)) entryData {
	double buyVol;
	double sellVol;
};

struct __attribute__((packed)) entryAndExit {
	double profitMargin;
	int entryIndex;
	int exitIndex;
	entryData e;
	bool isLong;
};

struct __attribute__((packed)) detailedTrade {
	double profitMargin;
	long long entryTimestamp;
	long long exitTimestamp;
	entryData e;
	bool isLong;
};

struct perfMetrics {
	double sharpe;
	double avgDrawdown;
	double maxDrawdown;
	double avgDrawdownLength;
	double maxDrawdownLength;
	double avgLossStreak;
	int maxLossStreak;
	double avgTradeDuration;
	double maxTradeDuration;
};

struct __attribute__((packed)) entry {
	double price;
	bool isLong;
};

struct __attribute__((packed)) tradeRecord {
	double capital;
	int shorts;
	int shortWins;
	int shortLosses;
	int longs;
	int longWins;
	int longLosses;
};

tradeWithoutDate convertTrade(const trade &orig) {
	tradeWithoutDate newTrade;
	// newTrade.tradeId = orig.tradeId;
	newTrade.timestamp = orig.timestamp;
	newTrade.price = orig.price;
	newTrade.qty = orig.qty;
	newTrade.isBuyerMaker = orig.isBuyerMaker;
	return newTrade;
}

string joinStrings(tuple<bool, double, string, int> t) {
	string s = get<0>(t) ? "LONG " : "SHORT ";
	s += to_string(get<1>(t));
	s += " ";
	s += get<2>(t);
	s += " ";
	s += to_string(get<3>(t));
	return s;
}

string convertTsToDate(long long ts) {
	system_clock::time_point tp = system_clock::time_point{microseconds(ts)};
	int micros = ts % 1000000;
	time_t time = system_clock::to_time_t(tp);
	ostringstream oss;
	oss << put_time(gmtime(&time), "%F %T");

	return oss.str() + "." + format("{:06}", micros);
}

double capitalToAnnualizedReturn(double capital, long long t1, long long t2) {
	return (pow(capital, ONE_YEAR_MICROSECONDS / (double)(t2 - t1)) - 1) * 100;
}

void computeIndicators(const vector<tradeWithoutDate> &tradesWithoutDates,
											 vector<indicators> &inds, vector<timeWindow> &tws,
											 const vector<long long> &windows) {
#ifdef DEBUG
	static size_t testIdx;
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
		if (testIdx % 200000 == 0) {
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
								 vector<int> &numTradesInInterval) {
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

	for (size_t i = currIdx; i < currSize; i++) {
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
	numTradesInInterval[index] = tradesInInterval;
}

int processTradesWithIndicators(vector<tradeWithoutDate> &tradesWithoutDates,
																vector<indicators> &inds,
																vector<timeWindow> &tws,
																const vector<combo> &comboVect,
																vector<entry> &entries,
																vector<tradeRecord> &tradeRecords) {
	size_t currIdx = newStart;
	static size_t globalIdx = 0;

	vector<int> numTradesInInterval(comboVect.size(), 0);

	int maxTradesPerInterval = 0;

	while (true) {
		size_t currSize =
				min((size_t)INCREMENT, tradesWithoutDates.size() - currIdx);

		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << " (" << globalIdx << "-" << globalIdx + currSize - 1 << ")" << endl;
		ThreadPool pool(thread::hardware_concurrency());
		for (size_t i = 0; i < comboVect.size(); ++i) {
			pool.enqueue([=, &tradesWithoutDates, &inds, &comboVect, &entries,
										&tradeRecords, &numTradesInInterval]() {
				/*
				{
					lock_guard<mutex> lock(coutMutex);
					cout << "Task " << i << " running in thread " <<
				std::this_thread::get_id() << endl;
				}
				*/
				performWork(i, currIdx, currSize, tradesWithoutDates, inds, comboVect,
										entries, tradeRecords, numTradesInInterval);
			});
		}
		pool.wait();

		/*
		err = queue.enqueueWriteBuffer(inputTrades, CL_FALSE, 0,
																	 currSize * sizeof(tradeWithoutDate),
																	 &tradesWithoutDates[currIdx]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputTrades: " << err << endl;
			return -1;
		}
		err =
				queue.enqueueWriteBuffer(indicatorBuffer, CL_FALSE, 0,
																 currSize * sizeof(indicators), &inds[currIdx]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer indicatorBuffer: " << err << endl;
			return -1;
		}
		err = queue.enqueueWriteBuffer(inputSize, CL_FALSE, 0, sizeof(int),
																	 &currSize);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputSize: " << err << endl;
			return -1;
		}
		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << " (" << globalIdx << "-" << globalIdx + currSize - 1 << ")" << endl;
		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange,
																		 cl::NDRange(comboVect.size()));
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel: " << err << endl;
			return -1;
		}
		err = queue.enqueueReadBuffer(numTradesInIntervalBuf, CL_FALSE, 0,
																	comboVect.size() * sizeof(cl_int),
																	&numTradesInInterval[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for reading numTradesInInterval: " << err << endl;
			return -1;
		}

		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return -1;
		}
		*/

		// cout << *max_element(numTradesInInterval.begin(),
		// numTradesInInterval.end())
		// 		 << endl;
		maxTradesPerInterval =
				max(maxTradesPerInterval, *max_element(numTradesInInterval.begin(),
																							 numTradesInInterval.end()));

		globalIdx += currSize;
		currIdx += currSize;

		if (currIdx >= tradesWithoutDates.size()) {
			break;
		}
	}

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

/*
void processTradesWithListingAndIndicators(
		const cl::CommandQueue &queue, const cl::Kernel &kernel,
		vector<tradeWithoutDate> &tradesWithoutDates, vector<indicators> &inds,
		vector<timeWindow> &tws, const vector<combo> &comboVect,
		const cl::Buffer &inputTrades, const cl::Buffer &indicatorBuffer,
		const cl::Buffer &inputSize, const cl::Buffer &entriesAndExitsBuf,
		vector<vector<detailedTrade>> &allTrades) {
	size_t currIdx = newStart;
	static size_t globalIdx = 0;

	cl_int err;

	vector<entryAndExit> entriesAndExits =
			vector<entryAndExit>(comboVect.size() * MAX_TOTAL_TRADES, entryAndExit{});

	while (true) {
		size_t currSize =
				min((size_t)INCREMENT, tradesWithoutDates.size() - currIdx);

		err = queue.enqueueWriteBuffer(inputTrades, CL_FALSE, 0,
																	 currSize * sizeof(tradeWithoutDate),
																	 &tradesWithoutDates[currIdx]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputTrades: " << err << endl;
			return;
		}
		err =
				queue.enqueueWriteBuffer(indicatorBuffer, CL_FALSE, 0,
																 currSize * sizeof(indicators), &inds[currIdx]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer indicatorBuffer: " << err << endl;
			return;
		}
		err = queue.enqueueWriteBuffer(inputSize, CL_FALSE, 0, sizeof(int),
																	 &currSize);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputSize: " << err << endl;
			return;
		}
		err = queue.enqueueWriteBuffer(entriesAndExitsBuf, CL_FALSE, 0,
																	 comboVect.size() * sizeof(entryAndExit) *
																			 MAX_TOTAL_TRADES,
																	 &entriesAndExits[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for writing entriesAndExits: " << err << endl;
			return;
		}
		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << " (" << globalIdx << "-" << globalIdx + currSize - 1 << ")" << endl;
		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange,
																		 cl::NDRange(comboVect.size()));
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(entriesAndExitsBuf, CL_TRUE, 0,
																	comboVect.size() * sizeof(entryAndExit) *
																			MAX_TOTAL_TRADES,
																	&entriesAndExits[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for reading entriesAndExits: " << err << endl;
			return;
		}

		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return;
		}

		// handle trade info
		for (size_t i = 0; i < comboVect.size(); i++) {
			for (size_t j = 0; j < MAX_TOTAL_TRADES; j++) {
				entryAndExit e = entriesAndExits[i * MAX_TOTAL_TRADES + j];
				if (e.entryIndex == 0 && e.exitIndex == 0) {
					break;
				}
				if (e.exitIndex != 0) {
					if (e.entryIndex == 0) {
#ifdef DEBUG
						assert(e.e.buyVol == 0);
						assert(e.e.sellVol == 0);
						assert(!e.isLong);
						assert(allTrades[i].back().exitTimestamp == 0);
						assert(allTrades[i].back().profitMargin == 0.0);
#endif
						allTrades[i].back().profitMargin = e.profitMargin;
						allTrades[i].back().exitTimestamp =
								tradesWithoutDates[e.exitIndex].timestamp;
					} else {
						detailedTrade d = {
								e.profitMargin, tradesWithoutDates[e.entryIndex].timestamp,
								tradesWithoutDates[e.exitIndex].timestamp, e.e, (bool)e.isLong};
						allTrades[i].emplace_back(d);
					}
				} else {
#ifdef DEBUG
					assert(e.profitMargin == 0.0);
#endif
					detailedTrade d = {0, tradesWithoutDates[e.entryIndex].timestamp, 0,
														 e.e, (bool)e.isLong};
					allTrades[i].emplace_back(d);
					break;
				}
			}
		}

		globalIdx += currSize;
		currIdx += currSize;

		if (currIdx >= tradesWithoutDates.size()) {
			break;
		}
	}

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
*/

void analyzePerf(vector<perfMetrics> &allPerfMetrics,
								 vector<vector<detailedTrade>> &allTrades) {
	for (size_t i = 0; i < allTrades.size(); i++) {
		if (allTrades[i].empty())
			continue;

		// Discard incomplete trades
		if (allTrades[i].back().exitTimestamp == 0)
			allTrades[i].pop_back();
		if (allTrades[i].empty())
			continue;

		perfMetrics p;
		vector<double> monthlyReturns, drawdowns;
		vector<long long> drawdownLengths, tradeDurations;
		vector<int> lossStreaks;

		bool onLossStreak = false;
		double currDrawdown = 1.0, currMonthlyRet = 1.0;
		int currLossStreak = 0;
		long long lastLosingExitTimestamp;
		long long nextMonthTimestamp =
				getTsOfNextMonth(allTrades[i].front().exitTimestamp);

		for (size_t j = 0; j < allTrades[i].size(); j++) {
			detailedTrade d = allTrades[i][j];
			bool win = d.profitMargin >= 1.0;
			tradeDurations.push_back(d.exitTimestamp - d.entryTimestamp);

			if (win && onLossStreak) {
				onLossStreak = false;
				lossStreaks.push_back(currLossStreak);
				currLossStreak = 0;
				drawdownLengths.push_back(d.exitTimestamp - lastLosingExitTimestamp);
				drawdowns.push_back(currDrawdown);
				currDrawdown = 1.0;
			} else if (!win) {
				if (!onLossStreak) {
					onLossStreak = true;
					lastLosingExitTimestamp = d.exitTimestamp;
				}

				currLossStreak++;
				currDrawdown *= d.profitMargin;
			}

			if (getTsOfNextMonth(d.exitTimestamp) == nextMonthTimestamp)
				currMonthlyRet *= d.profitMargin;
			else {
				nextMonthTimestamp = getTsOfNextMonth(d.exitTimestamp);
				monthlyReturns.push_back(currMonthlyRet);
				currMonthlyRet = d.profitMargin;
			}
		}

		if (!monthlyReturns.empty())
			monthlyReturns.pop_back();

		double meanMonthlyReturn =
				accumulate(monthlyReturns.begin(), monthlyReturns.end(), 0.0) /
				monthlyReturns.size();
		vector<double> diff(monthlyReturns.size());
		transform(monthlyReturns.begin(), monthlyReturns.end(), diff.begin(),
							[meanMonthlyReturn](double x) { return x - meanMonthlyReturn; });
		double sqSum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
		double stdev = sqrt(sqSum / monthlyReturns.size());
		p.sharpe = (meanMonthlyReturn - RISK_FREE_RATE) / stdev;

		p.avgDrawdown =
				accumulate(drawdowns.begin(), drawdowns.end(), 0.0) / drawdowns.size();
		p.avgDrawdown = (p.avgDrawdown - 1) * 100;
		p.maxDrawdown = *min_element(drawdowns.begin(), drawdowns.end());
		p.maxDrawdown = (p.maxDrawdown - 1) * 100;

		p.avgDrawdownLength =
				(double)accumulate(drawdownLengths.begin(), drawdownLengths.end(), 0) /
				drawdownLengths.size();
		p.maxDrawdownLength =
				*max_element(drawdownLengths.begin(), drawdownLengths.end());

		p.avgLossStreak =
				(double)accumulate(lossStreaks.begin(), lossStreaks.end(), 0) /
				lossStreaks.size();
		p.maxLossStreak = *max_element(lossStreaks.begin(), lossStreaks.end());

		p.avgTradeDuration =
				(double)accumulate(tradeDurations.begin(), tradeDurations.end(), 0) /
				tradeDurations.size();
		p.maxTradeDuration =
				*max_element(tradeDurations.begin(), tradeDurations.end());

		allPerfMetrics[i] = p;
	}
}

void outputMetrics(ostream &os, size_t idx, const vector<combo> &comboVect,
									 const vector<tradeRecord> &tradeRecordsVec,
									 const vector<perfMetrics> &allPerfMetrics, bool listTrades) {
	os << fixed;
	os << "Annualized return: " << tradeRecordsVec[idx].capital << endl;
	os << "Target: " << format("{:.2f}", (double)comboVect[idx].target) << endl;
	os << "Stop loss: " << format("{:.2f}", (double)comboVect[idx].stopLoss)
		 << endl;
	os << "Window: " << comboVect[idx].window / ONE_MINUTE_MICROSECONDS
		 << " minutes" << endl;
	os << "Buy volume threshold: " << comboVect[idx].buyVolPercentile << endl;
	os << "Sell volume threshold: " << comboVect[idx].sellVolPercentile << endl;
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

	string symbol(argv[optind]);
	if (symbol.find("BTC") != string::npos)
		isBTC = true;
	else
		isBTC = false;

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

	vector<combo> comboVect;

	for (int i = 0; i < NUM_WINDOWS; i++) {
		for (unsigned int j = 0; j < combos[i].size(); j++) {
			combo c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
								 get<2>(combos[i][j]), get<3>(combos[i][j]),
								 get<4>(combos[i][j])};
			// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
			// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
			if (c.stopLoss < c.target + 2.1)
				comboVect.emplace_back(c);
		}
	}

	bool initializedPositions = false;

	vector<entry> entriesVec(comboVect.size(), {0.0, 0});
	vector<tradeRecord> tradeRecordsVec(comboVect.size(),
																			{1.0, 0, 0, 0, 0, 0, 0});

	int maxTradesPerInterval = 0;

	vector<entryAndExit> entriesAndExits;

	vector<int> numTradesInInterval;

	cout << comboVect.size() * MAX_TOTAL_TRADES * sizeof(entryAndExit) << endl;
	cout << comboVect.size() << endl;
	if (listTrades) {
		entriesAndExits = vector<entryAndExit>(comboVect.size() * MAX_TOTAL_TRADES,
																					 entryAndExit{});
	} else {
		numTradesInInterval = vector<int>(comboVect.size(), 0);
	}

	vector<tradeWithoutDate> tradesWithoutDates;
	vector<indicators> indicators;
	vector<timeWindow> tws(NUM_WINDOWS, {0, 0});

	bool justProcessed = false;

	vector<vector<detailedTrade>> allTrades;
	vector<perfMetrics> allPerfMetrics;
	if (listTrades) {
		allTrades.resize(comboVect.size());
		allPerfMetrics.resize(comboVect.size());
	}

	auto setupTime = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(setupTime - startTime);
	cout << "Time taken for setup: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	long long firstTimestamp = 0, lastTimestamp = 0;
#ifdef DEBUG
	int lastId = 0;
#endif

	for (int i = optind; i < argc; i++) {
		myFile.open(argv[i]);
		if (myFile.is_open()) {
			cout << "Reading " << argv[i] << endl;
			string line;
			getline(myFile, line);
			while (getline(myFile, line)) {
				if (justProcessed)
					cout << "Going back to reading" << endl;

				vector<string> splits = splitByComma(line);
				trade t;
				t.price = stod(splits[4]);
				t.qty = stod(splits[5]);
				string isBuyerMakerString = splits[6];
				transform(isBuyerMakerString.begin(), isBuyerMakerString.end(),
									isBuyerMakerString.begin(),
									[](unsigned char c) { return tolower(c); });
				istringstream(isBuyerMakerString) >> boolalpha >> t.isBuyerMaker;

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
				tradesWithoutDates.emplace_back(convertTrade(t));

				if (!initializedPositions) {
					for (auto &tw : tws) {
						tw.timestamp = t.timestamp;
					}
					initializedPositions = true;
				}

				justProcessed = false;
				if (tradesWithoutDates.size() - newStart == TRADE_CHUNK) {
					auto beforeIndicatorTime = high_resolution_clock::now();
					computeIndicators(tradesWithoutDates, indicators, tws, windows);
					auto afterIndicatorTime = high_resolution_clock::now();
					duration = duration_cast<microseconds>(afterIndicatorTime -
																								 beforeIndicatorTime);
					cout << "Time taken to compute indicators: "
							 << (double)duration.count() / 1000000 << " seconds" << endl;
					if (listTrades) {
					}
					// processTradesWithListingAndIndicators(
					//		queue, volKernel, tradesWithoutDates, indicators, tws,
					//		comboVect, inputTrades, indicatorBuffer, inputSize,
					//		entriesAndExitsBuf, allTrades);
					else
						maxTradesPerInterval =
								max(maxTradesPerInterval,
										processTradesWithIndicators(tradesWithoutDates, indicators,
																								tws, comboVect, entriesVec,
																								tradeRecordsVec));
					// maxTradesPerInterval =
					//		max(maxTradesPerInterval,
					//				processTradesWithIndicators(
					//						queue, volKernel, tradesWithoutDates, indicators, tws,
					//						comboVect, inputTrades, indicatorBuffer, inputSize,
					//						numTradesInIntervalBuf));
					justProcessed = true;
				}
			}
			myFile.close();
		}
	}

	if (!justProcessed) {
		auto beforeIndicatorTime = high_resolution_clock::now();
		computeIndicators(tradesWithoutDates, indicators, tws, windows);
		auto afterIndicatorTime = high_resolution_clock::now();
		duration =
				duration_cast<microseconds>(afterIndicatorTime - beforeIndicatorTime);
		cout << "Time taken to compute indicators: "
				 << (double)duration.count() / 1000000 << " seconds" << endl;
		if (listTrades) {
		}
		// processTradesWithListingAndIndicators(
		//		queue, volKernel, tradesWithoutDates, indicators, tws, comboVect,
		//		inputTrades, indicatorBuffer, inputSize, entriesAndExitsBuf,
		//		allTrades);
		else
			maxTradesPerInterval = max(
					maxTradesPerInterval,
					processTradesWithIndicators(tradesWithoutDates, indicators, tws,
																			comboVect, entriesVec, tradeRecordsVec));
	}

	/*
	err = queue.enqueueReadBuffer(tradeRecords, CL_FALSE, 0,
																comboVect.size() * sizeof(tradeRecord),
																&tradeRecordsVec[0]);
	if (err != CL_SUCCESS) {
		cout << "Error for reading tradeRecords: " << err << endl;
		return 1;
	}
	err = queue.finish();
	if (err != CL_SUCCESS) {
		cout << "Error for finish: " << err << endl;
		return 1;
	}
	*/

	for (size_t i = 0; i < tradeRecordsVec.size(); i++) {
		tradeRecordsVec[i].capital = capitalToAnnualizedReturn(
				tradeRecordsVec[i].capital, firstTimestamp, lastTimestamp);
	}

	auto afterKernelTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(afterKernelTime - setupTime);
	cout << "Time taken to run kernel: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	if (listTrades) {
		analyzePerf(allPerfMetrics, allTrades);

		auto afterAnalysisTime = high_resolution_clock::now();
		duration = duration_cast<microseconds>(afterAnalysisTime - afterKernelTime);
		cout << "Time taken to analyze models: "
				 << (double)duration.count() / 1000000 << " seconds" << endl;
	}

	if (writeResults) {
		ofstream outFile;
		if (isBTC)
			outFile.open("resultsVolBTC");
		else
			outFile.open("resultsVolETH");
		if (outFile.is_open()) {
			int maxElementIdx =
					max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
											[](tradeRecord t1, tradeRecord t2) {
												return t1.capital < t2.capital;
											}) -
					tradeRecordsVec.begin();
			outFile << "Max return:" << endl;
			outputMetrics(outFile, maxElementIdx, comboVect, tradeRecordsVec,
										allPerfMetrics, listTrades);
			outFile << endl;

			maxElementIdx =
					max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
											[](tradeRecord t1, tradeRecord t2) {
												return (double)(t1.shortWins + t1.longWins) /
																	 (t1.shorts + t1.longs) <
															 (double)(t2.shortWins + t2.longWins) /
																	 (t2.shorts + t2.longs);
											}) -
					tradeRecordsVec.begin();
			outFile << "Best win rate:" << endl;
			outputMetrics(outFile, maxElementIdx, comboVect, tradeRecordsVec,
										allPerfMetrics, listTrades);
			outFile << endl;

			maxElementIdx =
					max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
											[](tradeRecord t1, tradeRecord t2) {
												return t1.shorts + t1.longs < t2.shorts + t2.longs;
											}) -
					tradeRecordsVec.begin();
			outFile << "Most trades:" << endl;
			outputMetrics(outFile, maxElementIdx, comboVect, tradeRecordsVec,
										allPerfMetrics, listTrades);
			outFile << endl;

			if (listTrades) {
				maxElementIdx =
						max_element(allPerfMetrics.begin(), allPerfMetrics.end(),
												[](perfMetrics p1, perfMetrics p2) {
													return p1.sharpe < p2.sharpe;
												}) -
						allPerfMetrics.begin();
				outFile << "Best sharpe ratio:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVect, tradeRecordsVec,
											allPerfMetrics, listTrades);
				outFile << endl;

				maxElementIdx =
						max_element(allPerfMetrics.begin(), allPerfMetrics.end(),
												[](perfMetrics p1, perfMetrics p2) {
													return p1.avgDrawdown < p2.avgDrawdown;
												}) -
						allPerfMetrics.begin();
				outFile << "Smallest average drawdown:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVect, tradeRecordsVec,
											allPerfMetrics, listTrades);
				outFile << endl;

				maxElementIdx =
						max_element(allPerfMetrics.begin(), allPerfMetrics.end(),
												[](perfMetrics p1, perfMetrics p2) {
													return p1.maxDrawdown < p2.maxDrawdown;
												}) -
						allPerfMetrics.begin();
				outFile << "Smallest max drawdown:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVect, tradeRecordsVec,
											allPerfMetrics, listTrades);
				outFile << endl;

				maxElementIdx =
						min_element(allPerfMetrics.begin(), allPerfMetrics.end(),
												[](perfMetrics p1, perfMetrics p2) {
													return p1.avgDrawdownLength < p2.avgDrawdownLength;
												}) -
						allPerfMetrics.begin();
				outFile << "Shortest average drawdown length:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVect, tradeRecordsVec,
											allPerfMetrics, listTrades);
				outFile << endl;

				maxElementIdx =
						min_element(allPerfMetrics.begin(), allPerfMetrics.end(),
												[](perfMetrics p1, perfMetrics p2) {
													return p1.avgLossStreak < p2.avgLossStreak;
												}) -
						allPerfMetrics.begin();
				outFile << "Shortest average loss streak:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVect, tradeRecordsVec,
											allPerfMetrics, listTrades);
				outFile << endl;

				maxElementIdx =
						min_element(allPerfMetrics.begin(), allPerfMetrics.end(),
												[](perfMetrics p1, perfMetrics p2) {
													return p1.avgTradeDuration < p2.avgTradeDuration;
												}) -
						allPerfMetrics.begin();
				outFile << "Shortest average trade duration:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVect, tradeRecordsVec,
											allPerfMetrics, listTrades);
				outFile << endl;

				maxElementIdx =
						max_element(allPerfMetrics.begin(), allPerfMetrics.end(),
												[](perfMetrics p1, perfMetrics p2) {
													return p1.avgTradeDuration < p2.avgTradeDuration;
												}) -
						allPerfMetrics.begin();
				outFile << "Longest average trade duration:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVect, tradeRecordsVec,
											allPerfMetrics, listTrades);
				outFile << endl;
			}

			for (size_t i = 0; i < comboVect.size(); i++) {
				outputMetrics(outFile, i, comboVect, tradeRecordsVec, allPerfMetrics,
											listTrades);
				if (listTrades) {
					for (size_t j = 0; j < allTrades[i].size(); j++) {
						detailedTrade e = allTrades[i][j];
						if (e.isLong) {
							outFile << "Long ";
						} else {
							outFile << "Short ";
						}
						outFile << e.entryTimestamp << endl;
						outFile << convertTsToDate(e.entryTimestamp) << endl;
						outFile << "Buy volume: " << e.e.buyVol << endl;
						outFile << "Sell volume: " << e.e.sellVol << endl;
						if (e.profitMargin >= 1.0) {
							outFile << "Profit: " << e.profitMargin;
						} else {
							outFile << "Loss: " << e.profitMargin;
						}
						outFile << " " << e.exitTimestamp << endl;
						outFile << convertTsToDate(e.exitTimestamp) << endl;
					}
				}
				outFile << endl;
			}
			outFile.close();

			auto outputTime = high_resolution_clock::now();
			duration = duration_cast<microseconds>(outputTime - afterKernelTime);
			cout << "Time taken to write output: "
					 << (double)duration.count() / 1000000 << " seconds" << endl;
		}
	}

	int maxElementIdx =
			max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
									[](tradeRecord t1, tradeRecord t2) {
										return t1.capital < t2.capital;
									}) -
			tradeRecordsVec.begin();
	cout << "Max return:" << endl;
	outputMetrics(cout, maxElementIdx, comboVect, tradeRecordsVec, allPerfMetrics,
								listTrades);
	cout << endl;

	maxElementIdx = max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
															[](tradeRecord t1, tradeRecord t2) {
																return (double)(t1.shortWins + t1.longWins) /
																					 (t1.shorts + t1.longs) <
																			 (double)(t2.shortWins + t2.longWins) /
																					 (t2.shorts + t2.longs);
															}) -
									tradeRecordsVec.begin();
	cout << "Best win rate:" << endl;
	outputMetrics(cout, maxElementIdx, comboVect, tradeRecordsVec, allPerfMetrics,
								listTrades);
	cout << endl;

	maxElementIdx =
			max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
									[](tradeRecord t1, tradeRecord t2) {
										return t1.shorts + t1.longs < t2.shorts + t2.longs;
									}) -
			tradeRecordsVec.begin();
	cout << "Most trades:" << endl;
	outputMetrics(cout, maxElementIdx, comboVect, tradeRecordsVec, allPerfMetrics,
								listTrades);
	cout << endl;

	if (!listTrades)
		cout << "Max trades per interval: " << maxTradesPerInterval << endl;
	else {
		maxElementIdx = max_element(allPerfMetrics.begin(), allPerfMetrics.end(),
																[](perfMetrics p1, perfMetrics p2) {
																	return p1.sharpe < p2.sharpe;
																}) -
										allPerfMetrics.begin();
		cout << "Best sharpe ratio:" << endl;
		outputMetrics(cout, maxElementIdx, comboVect, tradeRecordsVec,
									allPerfMetrics, listTrades);
		cout << endl;

		maxElementIdx = max_element(allPerfMetrics.begin(), allPerfMetrics.end(),
																[](perfMetrics p1, perfMetrics p2) {
																	return p1.avgDrawdown < p2.avgDrawdown;
																}) -
										allPerfMetrics.begin();
		cout << "Smallest average drawdown:" << endl;
		outputMetrics(cout, maxElementIdx, comboVect, tradeRecordsVec,
									allPerfMetrics, listTrades);
		cout << endl;

		maxElementIdx = max_element(allPerfMetrics.begin(), allPerfMetrics.end(),
																[](perfMetrics p1, perfMetrics p2) {
																	return p1.maxDrawdown < p2.maxDrawdown;
																}) -
										allPerfMetrics.begin();
		cout << "Smallest max drawdown:" << endl;
		outputMetrics(cout, maxElementIdx, comboVect, tradeRecordsVec,
									allPerfMetrics, listTrades);
		cout << endl;

		maxElementIdx =
				min_element(allPerfMetrics.begin(), allPerfMetrics.end(),
										[](perfMetrics p1, perfMetrics p2) {
											return p1.avgDrawdownLength < p2.avgDrawdownLength;
										}) -
				allPerfMetrics.begin();
		cout << "Shortest average drawdown length:" << endl;
		outputMetrics(cout, maxElementIdx, comboVect, tradeRecordsVec,
									allPerfMetrics, listTrades);
		cout << endl;

		maxElementIdx = min_element(allPerfMetrics.begin(), allPerfMetrics.end(),
																[](perfMetrics p1, perfMetrics p2) {
																	return p1.avgLossStreak < p2.avgLossStreak;
																}) -
										allPerfMetrics.begin();
		cout << "Shortest average loss streak:" << endl;
		outputMetrics(cout, maxElementIdx, comboVect, tradeRecordsVec,
									allPerfMetrics, listTrades);
		cout << endl;

		maxElementIdx =
				min_element(allPerfMetrics.begin(), allPerfMetrics.end(),
										[](perfMetrics p1, perfMetrics p2) {
											return p1.avgTradeDuration < p2.avgTradeDuration;
										}) -
				allPerfMetrics.begin();
		cout << "Shortest average trade duration:" << endl;
		outputMetrics(cout, maxElementIdx, comboVect, tradeRecordsVec,
									allPerfMetrics, listTrades);
		cout << endl;

		maxElementIdx =
				max_element(allPerfMetrics.begin(), allPerfMetrics.end(),
										[](perfMetrics p1, perfMetrics p2) {
											return p1.avgTradeDuration < p2.avgTradeDuration;
										}) -
				allPerfMetrics.begin();
		cout << "Longest average trade duration:" << endl;
		outputMetrics(cout, maxElementIdx, comboVect, tradeRecordsVec,
									allPerfMetrics, listTrades);
		cout << endl;
	}

	auto endTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(endTime - startTime);
	cout << "Total time taken: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	return 0;
}
