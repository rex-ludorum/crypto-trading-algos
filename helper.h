#ifndef HELPER_H
#define HELPER_H

#include <CL/opencl.hpp>
#include <string>
#include <variant>
#include <vector>

using std::array;
using std::ostream;
using std::string;
using std::unordered_map;
using std::variant;
using std::vector;

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

#define MARCH_1_1972_IN_SECONDS 68256000
#define DAYS_IN_LEAP_YEAR_CYCLE 1461
#define SECONDS_IN_DAY 86400

#define RISK_FREE_RATE 1.01

#define STARTING_CAPITAL 400000

#define BTC_MARGIN_RATE 0.026666666
#define ETH_MARGIN_RATE 0.039145908

#define BTC_COMMISSION_PER_CONTRACT 1.92
#define ETH_COMMISSION_PER_CONTRACT 0.87

#define NUM_WINDOWS 4

#define FUTURES_PRICE_CHANGE_LIMIT 100

cl::Device getDefaultDevice(); // Return a device found in this OpenCL platform.

void initializeDevice(
		string filename); // Initialize device and compile kernel code.

vector<string> split(string const &input);

vector<string> splitByComma(const string &input);

long long getTsOfNextMonth(long long ts);

int isDst(long long ts);

enum class Mode {
	Volume,
	VolumeDelta,
	AllVolume,
	VolumeExit,
	Trend,
	VolumeTrend
};

const unordered_map<string, Mode> modeMap = {
		{"vol", Mode::Volume},			 {"volDelta", Mode::VolumeDelta},
		{"allVol", Mode::AllVolume}, {"volExit", Mode::VolumeExit},
		{"trend", Mode::Trend},			 {"volTrend", Mode::VolumeTrend}};

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
};

struct __attribute__((packed)) indicators {
	array<cl_double, 2 * NUM_WINDOWS> vols;
};

struct timeWindow {
	size_t tradeIdx;
	long long timestamp;
};

struct __attribute__((packed)) positionData {
	cl_double maxPrice;
	cl_double minPrice;
	cl_uchar bools;
};

struct __attribute__((packed)) twMetadata {
	cl_int twTranslation;
	cl_int twStart;
};

struct __attribute__((packed)) comboVolPerps {
	cl_long window;
	cl_double buyVolPercentile;
	cl_double sellVolPercentile;
	cl_double stopLoss;
	cl_double target;
	cl_double leverageMultiplier;
	cl_int periodsAfterOpen;
};

struct __attribute__((packed)) comboVolFutures {
	cl_long window;
	cl_double buyVolPercentile;
	cl_double sellVolPercentile;
	cl_double stopLoss;
	cl_double target;
	cl_double contractMultiplier;
	cl_int periodsAfterOpen;
};

struct __attribute__((packed)) comboVolDeltaPerps {
	cl_long window;
	cl_double buyVolDeltaPercentile;
	cl_double sellVolDeltaPercentile;
	cl_double stopLoss;
	cl_double target;
	cl_double leverageMultiplier;
	cl_int periodsAfterOpen;
};

struct __attribute__((packed)) comboVolDeltaFutures {
	cl_long window;
	cl_double buyVolDeltaPercentile;
	cl_double sellVolDeltaPercentile;
	cl_double stopLoss;
	cl_double target;
	cl_double contractMultiplier;
	cl_int periodsAfterOpen;
};

struct __attribute__((packed)) comboVolExitPerps {
	cl_long window;
	cl_double buyVolPercentile;
	cl_double sellVolPercentile;
	cl_double buyVolExitPercentile;
	cl_double sellVolExitPercentile;
	cl_double leverageMultiplier;
	cl_int periodsAfterOpen;
};

struct __attribute__((packed)) comboVolExitFutures {
	cl_long window;
	cl_double buyVolPercentile;
	cl_double sellVolPercentile;
	cl_double buyVolExitPercentile;
	cl_double sellVolExitPercentile;
	cl_double contractMultiplier;
	cl_int periodsAfterOpen;
};

struct __attribute__((packed)) comboAllVolPerps {
	cl_long window;
	cl_double buyVolPercentile;
	cl_double sellVolPercentile;
	cl_double buyVolDeltaPercentile;
	cl_double sellVolDeltaPercentile;
	cl_double stopLoss;
	cl_double target;
	cl_double leverageMultiplier;
	cl_int periodsAfterOpen;
};

struct __attribute__((packed)) comboAllVolFutures {
	cl_long window;
	cl_double buyVolPercentile;
	cl_double sellVolPercentile;
	cl_double buyVolDeltaPercentile;
	cl_double sellVolDeltaPercentile;
	cl_double stopLoss;
	cl_double target;
	cl_double contractMultiplier;
	cl_int periodsAfterOpen;
};

struct __attribute__((packed)) comboTrendPerps {
	cl_double entryThreshold;
	cl_double stopLoss;
	cl_double target;
	cl_double leverageMultiplier;
	cl_int periodsAfterOpen;
};

struct __attribute__((packed)) comboTrendFutures {
	cl_double entryThreshold;
	cl_double stopLoss;
	cl_double target;
	cl_double contractMultiplier;
	cl_int periodsAfterOpen;
};

struct __attribute__((packed)) comboVolTrendPerps {
	cl_long window;
	cl_double buyVolPercentile;
	cl_double sellVolPercentile;
	cl_double entryThreshold;
	cl_double stopLoss;
	cl_double target;
	cl_double leverageMultiplier;
	cl_int periodsAfterOpen;
};

struct __attribute__((packed)) comboVolTrendFutures {
	cl_long window;
	cl_double buyVolPercentile;
	cl_double sellVolPercentile;
	cl_double entryThreshold;
	cl_double stopLoss;
	cl_double target;
	cl_double contractMultiplier;
	cl_int periodsAfterOpen;
};

using ComboVec =
		variant<vector<comboVolPerps>, vector<comboVolFutures>,
						vector<comboVolDeltaPerps>, vector<comboVolDeltaFutures>,
						vector<comboVolExitPerps>, vector<comboVolExitFutures>,
						vector<comboAllVolPerps>, vector<comboAllVolFutures>,
						vector<comboTrendPerps>, vector<comboTrendFutures>,
						vector<comboVolTrendPerps>, vector<comboVolTrendFutures>>;

struct __attribute__((packed)) drawdowns {
	cl_int n;
	cl_double max;
	cl_double mean;
	cl_double m2;
	cl_double currentMax;
	cl_double currentMin;
	cl_uchar enabled;
};

struct __attribute__((packed)) drawdownLengths {
	cl_long max;
	cl_double mean;
	cl_double m2;
	cl_long drawdownStart;
};

struct __attribute__((packed)) lossStreaks {
	cl_int n;
	cl_int max;
	cl_double mean;
	cl_double m2;
	cl_int current;
};

struct __attribute__((packed)) tradeDurations {
	cl_int n;
	cl_long max;
	cl_double mean;
	cl_double m2;
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
	cl_int n;
	cl_double mean;
	cl_double m2;
	cl_double max;
	cl_double min;
};

struct __attribute__((packed)) losses {
	cl_int n;
	cl_double mean;
	cl_double m2;
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

struct __attribute__((packed)) twap {
	cl_int n;
	cl_double mean;
	cl_double max;
	cl_double min;
	cl_double prevPrice;
	cl_double currentNumer;
	cl_long currentDenom;
	cl_long currentTimestamp;
};

tradeWithoutDate convertTrade(const trade &orig);

string convertTsToDate(long long ts);

double capitalToAnnualizedReturn(double capital, long long t1, long long t2);

void analyzePerf(vector<perfMetrics> &allPerfMetrics,
								 const vector<monthlyReturns> &monthlyReturnsVec,
								 vector<drawdowns> &drawdownsVec);

void outputMetrics(ostream &os, size_t idx, const ComboVec &comboVec,
									 const vector<tradeRecord> &tradeRecordsVec,
									 const vector<perfMetrics> &allPerfMetrics,
									 const vector<drawdowns> &drawdownsVec,
									 const vector<drawdownLengths> &drawdownLengthsVec,
									 const vector<lossStreaks> &lossStreaksVec,
									 const vector<tradeDurations> &tradeDurationsVec,
									 const vector<wins> &winsVec, const vector<losses> &lossesVec,
									 const vector<twap> &twapsVec, bool listTrades);

#endif
