#include <CL/opencl.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ranges>
#include <string>
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
using std::max;
using std::max_element;
using std::min;
using std::min_element;
using std::minus;
using std::mktime;
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

#define NUM_WINDOWS 4

#define MAX_TOTAL_TRADES 15489

#define INCREMENT 250000
#define TRADE_CHUNK 50000000

#define PERCENTILE_CEILING 30
#define PERCENTILE_FLOOR 4

#define MARCH_1_1972_IN_SECONDS 68256000
#define DAYS_IN_LEAP_YEAR_CYCLE 1461
#define SECONDS_IN_DAY 86400

#define RISK_FREE_RATE 1.01

constexpr array<int, 12> daysInMonths = {31, 30, 31, 30, 31, 31,
																				 30, 31, 30, 31, 31, 28};

cl::Device getDefaultDevice(); // Return a device found in this OpenCL platform.

void initializeDevice(); // Initialize device and compile kernel code.

cl::Program program; // The program that will run on the device.
cl::Context context; // The context which holds the device.
cl::Device device;	 // The device where the kernel will run.

cl::Device getDefaultDevice() {
	/**
	 * Search for all the OpenCL platforms available and check
	 * if there are any.
	 * */
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.empty()) {
		cerr << "No platforms found!" << endl;
		exit(1);
	}

	/**
	 * Search for all the devices on the first platform
	 * and check if there are any available.
	 * */
	auto platform = platforms.front();
	std::vector<cl::Device> devices;
	cout << "Using platform " << platform.getInfo<CL_PLATFORM_NAME>() << endl;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	if (devices.empty()) {
		cerr << "No devices found!" << endl;
		exit(1);
	}
	cout << "It has " << devices.size() << " devices" << endl;

	int maxComputeUnits = devices.front().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	int maxIdx = 0;

	for (size_t i = 1; i < devices.size(); i++) {
		int tmp = devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		if (tmp > maxComputeUnits) {
			maxComputeUnits = tmp;
			maxIdx = i;
		}
	}

	/**
	 * Return the first device found.
	 * */
	// return devices.front();

	// Return most powerful device
	cout << "Using device " << devices[maxIdx].getInfo<CL_DEVICE_NAME>() << endl;
	return devices[maxIdx];
}

void initializeDevice() {
	/**
	 * Select the first available device.
	 * */
	device = getDefaultDevice();

	/**
	 * Read OpenCL kernel file as a string.
	 * */
	std::ifstream kernel_file("vol-trader.cl");
	std::string src(std::istreambuf_iterator<char>(kernel_file),
									(std::istreambuf_iterator<char>()));

	/**
	 * Compile kernel program which will run on the device.
	 * */
	cl::Program::Sources sources(1, src);
	cl_int err;
	context = cl::Context(device, nullptr, nullptr, nullptr, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for context creation: " << err << endl;
	}
	program = cl::Program(context, sources, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for program creation: " << err << endl;
	}

	err = program.build();
	if (err != CL_BUILD_SUCCESS) {
		cerr << "Error!\nBuild Status: "
				 << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
				 << "\nBuild Log:\t "
				 << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
		exit(1);
	}
}

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

struct trade {
	int tradeId;
	long long timestamp;
	double price;
	double qty;
	bool isBuyerMaker;
	string date;
};

struct __attribute__((packed)) tradeWithoutDate {
	cl_long timestamp;
	cl_double price;
	cl_double qty;
	cl_uchar isBuyerMaker;
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
	cl_double price;
	cl_uchar isLong;
};

struct __attribute__((packed)) tradeRecord {
	cl_double capital;
	cl_int shorts;
	cl_int shortWins;
	cl_int shortLosses;
	cl_int longs;
	cl_int longWins;
	cl_int longLosses;
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

int processTrades(cl::CommandQueue &queue, cl::Kernel &kernel,
									vector<tradeWithoutDate> &tradesWithoutDates,
									vector<combo> &comboVect, cl::Buffer &inputTrades,
									cl::Buffer &inputSize, cl::Buffer &twBetweenRunData,
									cl::Buffer &positionDatas,
									cl::Buffer &numTradesInIntervalBuf) {
	size_t currIdx = 0;
	static size_t globalIdx = 0;

	cl_int err;

	static twMetadata tw = {0, 0};
	vector<positionData> positionDatasVec(comboVect.size(), {0, 0.0, 0.0, 0});
	vector<cl_int> numTradesInInterval(comboVect.size(), 0);

	bool first = true;

	int maxTradesPerInterval = 0;

	while (true) {
		size_t currSize =
				min((size_t)INCREMENT, tradesWithoutDates.size() - currIdx);
		if (first && tw.twStart != 0) {
			tw.twStart = currSize - tw.twTranslation;
			first = false;
		}
		err = queue.enqueueWriteBuffer(inputTrades, CL_FALSE, 0,
																	 currSize * sizeof(tradeWithoutDate),
																	 &tradesWithoutDates[currIdx]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputTrades: " << err << endl;
			return -1;
		}
		err = queue.enqueueWriteBuffer(inputSize, CL_FALSE, 0, sizeof(int),
																	 &currSize);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputSize: " << err << endl;
			return -1;
		}
		err = queue.enqueueWriteBuffer(twBetweenRunData, CL_FALSE, 0,
																	 sizeof(twMetadata), &tw);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer twBetweenRunData: " << err << endl;
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
		err = queue.enqueueReadBuffer(positionDatas, CL_FALSE, 0,
																	comboVect.size() * sizeof(positionData),
																	&positionDatasVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for reading positionDatas: " << err << endl;
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

		// cout << *max_element(numTradesInInterval.begin(),
		// numTradesInInterval.end())
		// 		 << endl;
		maxTradesPerInterval =
				max(maxTradesPerInterval, *max_element(numTradesInInterval.begin(),
																							 numTradesInInterval.end()));
		int minElementIdx =
				min_element(positionDatasVec.begin(), positionDatasVec.end(),
										[](positionData p1, positionData p2) {
											return p1.tradeIdx < p2.tradeIdx;
										}) -
				positionDatasVec.begin();
		int minTradeIdx = positionDatasVec[minElementIdx].tradeIdx;
		tw.twTranslation = minTradeIdx;
		globalIdx += minTradeIdx;

		if (currIdx + currSize >= tradesWithoutDates.size()) {
			currIdx += minTradeIdx;
			break;
		}

		tw.twStart = currSize - minTradeIdx;
		currIdx += minTradeIdx;
	}

	tradesWithoutDates.erase(tradesWithoutDates.begin(),
													 tradesWithoutDates.begin() + currIdx);
	return maxTradesPerInterval;
}

void processTradesWithListing(cl::CommandQueue &queue, cl::Kernel &kernel,
															vector<tradeWithoutDate> &tradesWithoutDates,
															vector<combo> &comboVect, cl::Buffer &inputTrades,
															cl::Buffer &inputSize,
															cl::Buffer &twBetweenRunData,
															cl::Buffer &positionDatas,
															cl::Buffer &entriesAndExitsBuf,
															vector<vector<detailedTrade>> &allTrades) {
	size_t currIdx = 0;
	static size_t globalIdx = 0;

	cl_int err;

	static twMetadata tw = {0, 0};
	vector<positionData> positionDatasVec(comboVect.size(), {0, 0.0, 0.0, 0});
	vector<cl_int> numTradesInInterval(comboVect.size(), 0);

	bool first = true;

	vector<entryAndExit> entriesAndExits =
			vector<entryAndExit>(comboVect.size() * MAX_TOTAL_TRADES, entryAndExit{});

	while (true) {
		size_t currSize =
				min((size_t)INCREMENT, tradesWithoutDates.size() - currIdx);
		if (first && tw.twStart != 0) {
			tw.twStart = currSize - tw.twTranslation;
			first = false;
		}
		err = queue.enqueueWriteBuffer(inputTrades, CL_FALSE, 0,
																	 currSize * sizeof(tradeWithoutDate),
																	 &tradesWithoutDates[currIdx]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputTrades: " << err << endl;
			return;
		}
		err = queue.enqueueWriteBuffer(inputSize, CL_FALSE, 0, sizeof(int),
																	 &currSize);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputSize: " << err << endl;
			return;
		}
		err = queue.enqueueWriteBuffer(twBetweenRunData, CL_FALSE, 0,
																	 sizeof(twMetadata), &tw);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer twBetweenRunData: " << err << endl;
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
		err = queue.enqueueReadBuffer(positionDatas, CL_FALSE, 0,
																	comboVect.size() * sizeof(positionData),
																	&positionDatasVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for reading positionDatas: " << err << endl;
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
					detailedTrade d = {0, tradesWithoutDates[e.entryIndex].timestamp, 0,
														 e.e, (bool)e.isLong};
					allTrades[i].emplace_back(d);
					break;
				}
			}
		}

		int minElementIdx =
				min_element(positionDatasVec.begin(), positionDatasVec.end(),
										[](positionData p1, positionData p2) {
											return p1.tradeIdx < p2.tradeIdx;
										}) -
				positionDatasVec.begin();
		int minTradeIdx = positionDatasVec[minElementIdx].tradeIdx;
		tw.twTranslation = minTradeIdx;
		globalIdx += minTradeIdx;

		if (currIdx + currSize >= tradesWithoutDates.size()) {
			currIdx += minTradeIdx;
			break;
		}

		tw.twStart = currSize - minTradeIdx;
		currIdx += minTradeIdx;
	}

	tradesWithoutDates.erase(tradesWithoutDates.begin(),
													 tradesWithoutDates.begin() + currIdx);
}

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

void outputMetrics(ostream &os, size_t idx, vector<combo> &comboVect,
									 vector<tradeRecord> &tradeRecordsVec,
									 vector<perfMetrics> &allPerfMetrics, bool listTrades) {
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

	initializeDevice();

	vector<positionData> positionDatasVec(comboVect.size(), {0, 0.0, 0.0, 0});
	bool initializedPositions = false;

	vector<entry> entriesVec(comboVect.size(), {0.0, 0});
	vector<tradeRecord> tradeRecordsVec(comboVect.size(),
																			{1.0, 0, 0, 0, 0, 0, 0});

	int maxTradesPerInterval = 0;

	cl_int err;

	cl::CommandQueue queue(context, device, 0, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for queue: " << err << endl;
		return 1;
	}

	cl::Kernel volKernel;
	if (listTrades) {
		volKernel = cl::Kernel(program, "volTraderWithTrades", &err);
	} else {
		volKernel = cl::Kernel(program, "volTrader", &err);
	}

	cl::Buffer inputTrades(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
												 INCREMENT * sizeof(tradeWithoutDate), NULL, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputTrades: " << err << endl;
		return 1;
	}
	cl::Buffer inputSize(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
											 sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputSize: " << err << endl;
		return 1;
	}
	cl::Buffer twBetweenRunData(context,
															CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
															sizeof(twMetadata), NULL, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for twBetweenRunData: " << err << endl;
		return 1;
	}
	cl::Buffer inputCombos(
			context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
			comboVect.size() * sizeof(combo), &comboVect[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputCombos: " << err << endl;
		return 1;
	}
	cl::Buffer entries(
			context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			comboVect.size() * sizeof(entry), &entriesVec[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for entries: " << err << endl;
		return 1;
	}
	cl::Buffer tradeRecords(
			context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			comboVect.size() * sizeof(tradeRecord), &tradeRecordsVec[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for tradeRecords: " << err << endl;
		return 1;
	}
	cl::Buffer positionDatas(
			context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			comboVect.size() * sizeof(positionData), &positionDatasVec[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for positionDatas: " << err << endl;
		return 1;
	}

	vector<entryAndExit> entriesAndExits;
	cl::Buffer entriesAndExitsBuf;

	vector<cl_int> numTradesInInterval;
	cl::Buffer numTradesInIntervalBuf;

	// cout << comboVect.size() * MAX_TOTAL_TRADES * sizeof(entryAndExit) << endl;
	if (listTrades) {
		entriesAndExits = vector<entryAndExit>(comboVect.size() * MAX_TOTAL_TRADES,
																					 entryAndExit{});
		entriesAndExitsBuf = cl::Buffer(
				context,
				CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				comboVect.size() * sizeof(entryAndExit) * MAX_TOTAL_TRADES,
				&entriesAndExits[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for entriesAndExitsBuf: " << err << endl;
			return 1;
		}
	} else {
		numTradesInInterval = vector<cl_int>(comboVect.size(), 0);
		numTradesInIntervalBuf = cl::Buffer(
				context,
				CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				comboVect.size() * sizeof(cl_int), &numTradesInInterval[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for numTradesInIntervalBuf: " << err << endl;
			return 1;
		}
	}

	if (err != CL_SUCCESS) {
		cout << "Error for creating volKernel: " << err << endl;
		return 1;
	}
	err = volKernel.setArg(0, inputSize);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 0: " << err << endl;
	}
	err = volKernel.setArg(1, inputTrades);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 1: " << err << endl;
	}
	err = volKernel.setArg(2, inputCombos);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 2: " << err << endl;
	}
	err = volKernel.setArg(3, entries);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 3: " << err << endl;
	}
	err = volKernel.setArg(4, tradeRecords);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 4: " << err << endl;
	}
	err = volKernel.setArg(5, positionDatas);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 5: " << err << endl;
	}
	err = volKernel.setArg(6, twBetweenRunData);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 6: " << err << endl;
	}

	if (listTrades) {
		err = volKernel.setArg(7, entriesAndExitsBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel setArg 7: " << err << endl;
		}
	} else {
		err = volKernel.setArg(7, numTradesInIntervalBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel setArg 7: " << err << endl;
		}
	}

	vector<trade> trades;
	vector<tradeWithoutDate> tradesWithoutDates;

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
				lastTimestamp = t.timestamp;
				// trades.emplace_back(t);
				tradesWithoutDates.emplace_back(convertTrade(t));

				if (!initializedPositions) {
					for (auto &pos : positionDatasVec) {
						pos.timestamp = t.timestamp;
					}
					initializedPositions = true;
				}

				justProcessed = false;
				if (tradesWithoutDates.size() == TRADE_CHUNK) {
					if (listTrades)
						processTradesWithListing(queue, volKernel, tradesWithoutDates,
																		 comboVect, inputTrades, inputSize,
																		 twBetweenRunData, positionDatas,
																		 numTradesInIntervalBuf, allTrades);
					else
						maxTradesPerInterval = max(
								maxTradesPerInterval,
								processTrades(queue, volKernel, tradesWithoutDates, comboVect,
															inputTrades, inputSize, twBetweenRunData,
															positionDatas, numTradesInIntervalBuf));
					justProcessed = true;
				}
			}
			myFile.close();
		}
	}

	if (!justProcessed) {
		if (listTrades)
			processTradesWithListing(queue, volKernel, tradesWithoutDates, comboVect,
															 inputTrades, inputSize, twBetweenRunData,
															 positionDatas, numTradesInIntervalBuf,
															 allTrades);
		else
			maxTradesPerInterval =
					max(maxTradesPerInterval,
							processTrades(queue, volKernel, tradesWithoutDates, comboVect,
														inputTrades, inputSize, twBetweenRunData,
														positionDatas, numTradesInIntervalBuf));
	}

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
