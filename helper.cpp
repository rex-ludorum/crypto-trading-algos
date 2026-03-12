#include "helper.h"
#include <CL/opencl.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

using std::array;
using std::format;
using std::istream_iterator;
using std::istringstream;
using std::ostringstream;
using std::put_time;
using std::string;
using std::vector;
using std::chrono::microseconds;
using std::chrono::system_clock;

using std::cerr;
using std::cout;
using std::endl;

#define ONE_YEAR_MICROSECONDS 31536000000000

#define MARCH_1_1972_IN_SECONDS 68256000
#define DAYS_IN_LEAP_YEAR_CYCLE 1461
#define SECONDS_IN_DAY 86400

#define RISK_FREE_RATE 1.01

constexpr array<int, 12> daysInMonths = {31, 30, 31, 30, 31, 31,
																				 30, 31, 30, 31, 31, 28};

extern cl::Program program; // The program that will run on the device.
extern cl::Context context; // The context which holds the device.
extern cl::Device device;		// The device where the kernel will run.

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

void initializeDevice(string filename) {
	/**
	 * Select the first available device.
	 * */
	device = getDefaultDevice();

	/**
	 * Read OpenCL kernel file as a string.
	 * */
	std::ifstream kernel_file(filename);
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

tradeWithoutDate convertTrade(const trade &orig) {
	tradeWithoutDate newTrade;
	// newTrade.tradeId = orig.tradeId;
	newTrade.timestamp = orig.timestamp;
	newTrade.price = orig.price;
	newTrade.qty = orig.qty;
	newTrade.isBuyerMaker = orig.isBuyerMaker;
	return newTrade;
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

/*
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
		vector<double> monthlyReturnsVec, drawdowns;
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
				monthlyReturnsVec.push_back(currMonthlyRet);
				currMonthlyRet = d.profitMargin;
			}
		}

		if (!monthlyReturnsVec.empty())
			monthlyReturnsVec.pop_back();

		double meanMonthlyReturn =
				accumulate(monthlyReturnsVec.begin(), monthlyReturnsVec.end(), 0.0) /
				monthlyReturnsVec.size();
		vector<double> diff(monthlyReturnsVec.size());
		transform(monthlyReturnsVec.begin(), monthlyReturnsVec.end(), diff.begin(),
							[meanMonthlyReturn](double x) { return x - meanMonthlyReturn; });
		double sqSum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
		double stdev = sqrt(sqSum / monthlyReturnsVec.size());
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
*/

void analyzePerf(vector<perfMetrics> &allPerfMetrics,
								 const vector<monthlyReturns> &monthlyReturnsVec,
								 vector<drawdowns> &drawdownsVec) {
	for (size_t i = 0; i < allPerfMetrics.size(); i++) {
		perfMetrics p;

		double stdev =
				sqrt(monthlyReturnsVec[i].m2 / (double)(monthlyReturnsVec[i].n - 1));
		p.sharpe = (monthlyReturnsVec[i].mean - RISK_FREE_RATE) / stdev;

		drawdownsVec[i].mean = (drawdownsVec[i].mean - 1) * 100;
		drawdownsVec[i].max = (drawdownsVec[i].max - 1) * 100;

		allPerfMetrics[i] = p;
	}
}
