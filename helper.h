#ifndef HELPER_H
#define HELPER_H

#include <CL/opencl.hpp>
#include <string>
#include <vector>

using std::string;
using std::vector;

#define ONE_HOUR_MICROSECONDS 3600000000
#define FIFTEEN_MINUTES_MICROSECONDS 900000000
#define ONE_MINUTE_MICROSECONDS 60000000
#define ONE_YEAR_MICROSECONDS 31536000000000

#define MARCH_1_1972_IN_SECONDS 68256000
#define DAYS_IN_LEAP_YEAR_CYCLE 1461
#define SECONDS_IN_DAY 86400

#define RISK_FREE_RATE 1.01

cl::Device getDefaultDevice(); // Return a device found in this OpenCL platform.

void initializeDevice(
		string filename); // Initialize device and compile kernel code.

vector<string> split(string const &input);

vector<string> splitByComma(const string &input);

long long getTsOfNextMonth(long long ts);

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

struct __attribute__((packed)) drawdowns {
	cl_double max;
	cl_double mean;
	cl_double current;
};

struct __attribute__((packed)) drawdownLengths {
	cl_long max;
	cl_double mean;
	cl_long drawdownStart;
};

struct __attribute__((packed)) lossStreaks {
	cl_int n;
	cl_int max;
	cl_double mean;
	cl_int current;
};

struct __attribute__((packed)) tradeDurations {
	cl_int n;
	cl_long max;
	cl_double mean;
	cl_long entryTimestamp;
};

struct __attribute__((packed)) monthlyReturns {
	cl_int n;
	cl_double current;
	cl_double mean;
	cl_double m2;
	cl_long nextMonth;
};

struct __attribute__((packed)) wins {
	cl_double mean;
	cl_double max;
	cl_double min;
};

struct __attribute__((packed)) losses {
	cl_double mean;
	cl_double max;
	cl_double min;
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

tradeWithoutDate convertTrade(const trade &orig);

string convertTsToDate(long long ts);

double capitalToAnnualizedReturn(double capital, long long t1, long long t2);

void analyzePerf(vector<perfMetrics> &allPerfMetrics,
								 const vector<monthlyReturns> &monthlyReturnsVec,
								 vector<drawdowns> &drawdownsVec);

#endif
