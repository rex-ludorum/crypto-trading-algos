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
using std::max;
using std::max_element;
using std::min;
using std::min_element;
using std::mktime;
using std::ofstream;
using std::ostream;
using std::stod;
using std::stoi;
using std::string;
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

#define ONE_HOUR_MICROSECONDS 3600000000
#define FIFTEEN_MINUTES_MICROSECONDS 900000000
#define ONE_MINUTE_MICROSECONDS 60000000

#define NUM_WINDOWS 2

#define MAX_TOTAL_TRADES 3189

#define INCREMENT 1000000
#define TRADE_CHUNK 50000000

#define PERCENTILE_CEILING 30
#define PERCENTILE_FLOOR 12

#define DELTA_PERCENTILE_CUTOFF 3

static size_t newStart = 0;
static size_t globalIdx = 0;
static string snapshotFilename;

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
	cl_double buyVolDeltaPercentile;
	cl_double sellVolDeltaPercentile;
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
			/*
			cout << "45 minute buy volume: " << ind.vols[5] << endl;
			cout << "45 minute sell volume: " << ind.vols[4] << endl;
			cout << "60 minute buy volume: " << ind.vols[7] << endl;
			cout << "60 minute sell volume: " << ind.vols[6] << endl;
			*/
			cout << defaultfloat;
		}
#endif
		inds.push_back(ind);
	}
}

void saveSnapshot(size_t index, size_t start, const vector<entry> &entriesVec,
									const vector<tradeRecord> &tradeRecordsVec,
									const vector<cl_int> &numTradesInIntervalVec) {
	ofstream file(snapshotFilename, ios::binary);
	if (!file) {
		cout << "Error opening snapshot file" << endl;
		return;
	}

	file.write(reinterpret_cast<const char *>(&index), sizeof(index));
	file.write(reinterpret_cast<const char *>(&start), sizeof(start));
	file.write(reinterpret_cast<const char *>(entriesVec.data()),
						 entriesVec.size() * sizeof(entry));
	file.write(reinterpret_cast<const char *>(tradeRecordsVec.data()),
						 tradeRecordsVec.size() * sizeof(tradeRecord));
	file.write(reinterpret_cast<const char *>(numTradesInIntervalVec.data()),
						 numTradesInIntervalVec.size() * sizeof(cl_int));
	file.close();
}

void saveSnapshot(size_t index, size_t start, const vector<entry> &entriesVec,
									const vector<tradeRecord> &tradeRecordsVec,
									vector<drawdowns> &drawdownsVec,
									vector<drawdownLengths> &drawdownLengthsVec,
									vector<lossStreaks> &lossStreaksVec,
									vector<tradeDurations> &tradeDurationsVec,
									vector<monthlyReturns> &monthlyReturnsVec) {
	ofstream file(snapshotFilename, ios::binary);
	if (!file) {
		cout << "Error opening snapshot file" << endl;
		return;
	}

	file.write(reinterpret_cast<const char *>(&index), sizeof(index));
	file.write(reinterpret_cast<const char *>(&start), sizeof(start));
	file.write(reinterpret_cast<const char *>(entriesVec.data()),
						 entriesVec.size() * sizeof(entry));
	file.write(reinterpret_cast<const char *>(tradeRecordsVec.data()),
						 tradeRecordsVec.size() * sizeof(tradeRecord));
	file.write(reinterpret_cast<const char *>(drawdownsVec.data()),
						 drawdownsVec.size() * sizeof(drawdowns));
	file.write(reinterpret_cast<const char *>(drawdownLengthsVec.data()),
						 drawdownLengthsVec.size() * sizeof(drawdownLengths));
	file.write(reinterpret_cast<const char *>(lossStreaksVec.data()),
						 lossStreaksVec.size() * sizeof(lossStreaks));
	file.write(reinterpret_cast<const char *>(tradeDurationsVec.data()),
						 tradeDurationsVec.size() * sizeof(tradeDurations));
	file.write(reinterpret_cast<const char *>(monthlyReturnsVec.data()),
						 monthlyReturnsVec.size() * sizeof(monthlyReturns));
	file.close();
}

int processTradesWithIndicators(
		const cl::CommandQueue &queue, const cl::Kernel &kernel,
		vector<tradeWithoutDate> &tradesWithoutDates, vector<indicators> &inds,
		vector<timeWindow> &tws, const vector<combo> &comboVec,
		const cl::Buffer &inputTrades, const cl::Buffer &indicatorBuffer,
		const cl::Buffer &inputSize, const cl::Buffer &entries,
		const cl::Buffer &tradeRecords, const cl::Buffer &numTradesInIntervalBuf,
		vector<entry> &entriesVec, vector<tradeRecord> &tradeRecordsVec,
		vector<cl_int> &numTradesInIntervalVec, bool snapshotting) {
	size_t currIdx = newStart;

	cl_int err;

	int maxTradesPerInterval = 0;

	while (true) {
		size_t currSize =
				min((size_t)INCREMENT, tradesWithoutDates.size() - currIdx);

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
																		 cl::NDRange(comboVec.size()));
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel: " << err << endl;
			return -1;
		}
		err = queue.enqueueReadBuffer(numTradesInIntervalBuf, CL_FALSE, 0,
																	comboVec.size() * sizeof(cl_int),
																	&numTradesInIntervalVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer numTradesInIntervalBuf: " << err
					 << endl;
			return -1;
		}

		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return -1;
		}

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

	if (snapshotting) {
		auto beforeSnapshotTime = high_resolution_clock::now();
		err = queue.enqueueReadBuffer(
				entries, CL_FALSE, 0, comboVec.size() * sizeof(entry), &entriesVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer entries: " << err << endl;
			return -1;
		}
		err = queue.enqueueReadBuffer(tradeRecords, CL_FALSE, 0,
																	comboVec.size() * sizeof(tradeRecord),
																	&tradeRecordsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer tradeRecords: " << err << endl;
			return -1;
		}
		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return -1;
		}
		saveSnapshot(globalIdx, newStart, entriesVec, tradeRecordsVec,
								 numTradesInIntervalVec);
		auto duration = duration_cast<microseconds>(high_resolution_clock::now() -
																								beforeSnapshotTime);
		cout << "Time taken to take snapshot: "
				 << (double)duration.count() / 1000000 << " seconds" << endl;
	}

	return maxTradesPerInterval;
}

void processTradesWithOnlineAlgs(
		const cl::CommandQueue &queue, const cl::Kernel &kernel,
		vector<tradeWithoutDate> &tradesWithoutDates, vector<indicators> &inds,
		vector<timeWindow> &tws, const vector<combo> &comboVec,
		const cl::Buffer &inputTrades, const cl::Buffer &indicatorBuffer,
		const cl::Buffer &inputSize, const cl::Buffer &entries,
		const cl::Buffer &tradeRecords, const cl::Buffer &drawdownsBuf,
		const cl::Buffer &drawdownLengthsBuf, const cl::Buffer &lossStreaksBuf,
		const cl::Buffer &tradeDurationsBuf, const cl::Buffer &monthlyReturnsBuf,
		vector<entry> &entriesVec, vector<tradeRecord> &tradeRecordsVec,
		vector<drawdowns> &drawdownsVec,
		vector<drawdownLengths> &drawdownLengthsVec,
		vector<lossStreaks> &lossStreaksVec,
		vector<tradeDurations> &tradeDurationsVec,
		vector<monthlyReturns> &monthlyReturnsVec, bool snapshotting) {
	size_t currIdx = newStart;

	cl_int err;

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
		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << " (" << globalIdx << "-" << globalIdx + currSize - 1 << ")" << endl;
		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange,
																		 cl::NDRange(comboVec.size()));
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel: " << err << endl;
			return;
		}

		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return;
		}

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

	if (snapshotting) {
		auto beforeSnapshotTime = high_resolution_clock::now();
		err = queue.enqueueReadBuffer(
				entries, CL_FALSE, 0, comboVec.size() * sizeof(entry), &entriesVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer entries: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(tradeRecords, CL_FALSE, 0,
																	comboVec.size() * sizeof(tradeRecord),
																	&tradeRecordsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer tradeRecords: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(drawdownsBuf, CL_FALSE, 0,
																	comboVec.size() * sizeof(drawdowns),
																	&drawdownsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer drawdownsBuf: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(drawdownLengthsBuf, CL_FALSE, 0,
																	comboVec.size() * sizeof(drawdownLengths),
																	&drawdownLengthsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer drawdownLengthsBuf: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(lossStreaksBuf, CL_FALSE, 0,
																	comboVec.size() * sizeof(lossStreaks),
																	&lossStreaksVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer lossStreaksBuf: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(tradeDurationsBuf, CL_FALSE, 0,
																	comboVec.size() * sizeof(tradeDurations),
																	&tradeDurationsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer tradeDurationsBuf: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(monthlyReturnsBuf, CL_FALSE, 0,
																	comboVec.size() * sizeof(monthlyReturns),
																	&monthlyReturnsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer monthlyReturnsBuf: " << err << endl;
			return;
		}
		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return;
		}
		saveSnapshot(globalIdx, newStart, entriesVec, tradeRecordsVec, drawdownsVec,
								 drawdownLengthsVec, lossStreaksVec, tradeDurationsVec,
								 monthlyReturnsVec);
		auto duration = duration_cast<microseconds>(high_resolution_clock::now() -
																								beforeSnapshotTime);
		cout << "Time taken to take snapshot: "
				 << (double)duration.count() / 1000000 << " seconds" << endl;
	}

	return;
}

void outputMetrics(ostream &os, size_t idx, const vector<combo> &comboVec,
									 const vector<tradeRecord> &tradeRecordsVec,
									 const vector<perfMetrics> &allPerfMetrics, bool listTrades) {
	os << fixed;
	os << "Combo index: " << idx << endl;
	os << "Annualized return: " << tradeRecordsVec[idx].capital << endl;
	os << "Target: " << format("{:.2f}", (double)comboVec[idx].target) << endl;
	os << "Stop loss: " << format("{:.2f}", (double)comboVec[idx].stopLoss)
		 << endl;
	os << "Window: " << comboVec[idx].window / ONE_MINUTE_MICROSECONDS
		 << " minutes" << endl;
	os << "Buy volume threshold: " << comboVec[idx].buyVolPercentile << endl;
	os << "Sell volume threshold: " << comboVec[idx].sellVolPercentile << endl;
	os << "Buy volume delta threshold: " << comboVec[idx].buyVolDeltaPercentile
		 << endl;
	os << "Sell volume delta threshold: " << comboVec[idx].sellVolDeltaPercentile
		 << endl;
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
									 bool listTrades) {
	os << fixed;
	os << "Combo index: " << idx << endl;
	os << "Annualized return: " << tradeRecordsVec[idx].capital << endl;
	os << "Target: " << format("{:.2f}", (double)comboVec[idx].target) << endl;
	os << "Stop loss: " << format("{:.2f}", (double)comboVec[idx].stopLoss)
		 << endl;
	os << "Window: " << comboVec[idx].window / ONE_MINUTE_MICROSECONDS
		 << " minutes" << endl;
	os << "Buy volume threshold: " << comboVec[idx].buyVolPercentile << endl;
	os << "Sell volume threshold: " << comboVec[idx].sellVolPercentile << endl;
	os << "Buy volume delta threshold: " << comboVec[idx].buyVolDeltaPercentile
		 << endl;
	os << "Sell volume delta threshold: " << comboVec[idx].sellVolDeltaPercentile
		 << endl;
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
	}
	os << defaultfloat;
}

int main(int argc, char *argv[]) {
	bool writeResults = false, listTrades = false, snapshotting = false;

	bool isBTC;

	int opt;
	while ((opt = getopt(argc, argv, "wls")) != -1) {
		switch (opt) {
		case 'w':
			writeResults = true;
			cout << "Enabled writing results" << endl;
			break;
		case 'l':
			listTrades = true;
			cout << "Enabled listing trades" << endl;
			break;
		case 's':
			snapshotting = true;
			cout << "Enabled snapshotting" << endl;
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
		snapshotFilename = "snapshotAllVolBTC";
	} else {
		isBTC = false;
		snapshotFilename = "snapshotAllVolETH";
	}
	if (listTrades)
		snapshotFilename += "Detailed";
	snapshotFilename += "_";
	snapshotFilename += dateSuffix + ".bin";

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
	vector<vector<double>> volDeltaPercentiles;

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
			rowPercentiles.erase(rowPercentiles.end() - DELTA_PERCENTILE_CUTOFF,
													 rowPercentiles.end());
			rowPercentiles.erase(rowPercentiles.begin(),
													 rowPercentiles.begin() + DELTA_PERCENTILE_CUTOFF);
			volDeltaPercentiles.emplace_back(rowPercentiles);
		}
		myFile.close();
	}

	vector<vector<long long>> indWindows(NUM_WINDOWS);
	for (int i = 0; i < NUM_WINDOWS; i++) {
		indWindows[i] = vector<long long>(1, windows[i]);
	}
	auto firstCombo = cartesian_product(
			indWindows[0], buyVolPercentiles[0], sellVolPercentiles[0],
			volDeltaPercentiles[0], volDeltaPercentiles[0], stopLosses, targets);
	vector<decltype(firstCombo)> combos;
	combos.emplace_back(firstCombo);

	for (int i = 1; i < NUM_WINDOWS; i++) {
		combos.emplace_back(cartesian_product(
				indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i],
				volDeltaPercentiles[i], volDeltaPercentiles[i], stopLosses, targets));
	}

	vector<combo> comboVec;

	for (int i = 0; i < NUM_WINDOWS; i++) {
		for (unsigned int j = 0; j < combos[i].size(); j++) {
			combo c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
								 get<2>(combos[i][j]), get<3>(combos[i][j]),
								 get<4>(combos[i][j]), get<5>(combos[i][j]),
								 get<6>(combos[i][j])};
			// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
			// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
			if (c.stopLoss < c.target + 1.1 &&
					c.buyVolDeltaPercentile >= c.sellVolDeltaPercentile)
				comboVec.emplace_back(c);
		}
	}
	size_t comboVecSize = comboVec.size() * sizeof(combo);

	initializeDevice("all-vol.cl");

	bool initializedPositions = false;

	vector<entry> entriesVec(comboVec.size(), {0.0, 0});
	size_t entriesVecSize = entriesVec.size() * sizeof(entry);
	vector<tradeRecord> tradeRecordsVec(comboVec.size(), {1.0, 0, 0, 0, 0, 0, 0});
	size_t tradeRecordsVecSize = tradeRecordsVec.size() * sizeof(tradeRecord);

	ifstream snapshotFile;
	if (snapshotting) {
		snapshotFile.open(snapshotFilename, ios::binary);
		if (snapshotFile.good()) {
			if (!snapshotFile.read(reinterpret_cast<char *>(&globalIdx),
														 sizeof(globalIdx))) {
				cout << "Failed to read previous index" << endl;
				return 1;
			}
			cout << "globalIdx: " << globalIdx << endl;
			if (!snapshotFile.read(reinterpret_cast<char *>(&newStart),
														 sizeof(newStart))) {
				cout << "Failed to read new start" << endl;
				return 1;
			}
			cout << "newStart: " << newStart << endl;
			if (!snapshotFile.read(reinterpret_cast<char *>(entriesVec.data()),
														 comboVec.size() * sizeof(entry))) {
				cout << "Failed to read entries" << endl;
				return 1;
			}
			if (!snapshotFile.read(reinterpret_cast<char *>(tradeRecordsVec.data()),
														 comboVec.size() * sizeof(tradeRecord))) {
				cout << "Failed to read trade records" << endl;
				return 1;
			}
		} else {
			cout << "No snapshot file detected" << endl;
		}
	}

	int maxTradesPerInterval = 0;

	cl_int err;

	cl::CommandQueue queue(context, device, 0, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for queue: " << err << endl;
		return 1;
	}

	cl::Kernel volKernel;
	if (listTrades) {
		// volKernel = cl::Kernel(program, "volTraderWithTradesAndIndicators",
		// &err);
		volKernel = cl::Kernel(program, "volTraderWithOnlineAlgs", &err);
	} else {
		volKernel = cl::Kernel(program, "volTraderWithIndicators", &err);
	}
	if (err != CL_SUCCESS) {
		cout << "Error for creating kernel: " << err << endl;
		return 1;
	}

	cl::Buffer inputTrades(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
												 INCREMENT * sizeof(tradeWithoutDate), NULL, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputTrades: " << err << endl;
		return 1;
	}
	size_t inputTradesSize = INCREMENT * sizeof(tradeWithoutDate);
	cl::Buffer inputSize(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
											 sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputSize: " << err << endl;
		return 1;
	}
	cl::Buffer inputCombos(
			context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
			comboVec.size() * sizeof(combo), &comboVec[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputCombos: " << err << endl;
		return 1;
	}
	cl::Buffer entries(
			context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			comboVec.size() * sizeof(entry), &entriesVec[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for entries: " << err << endl;
		return 1;
	}
	cl::Buffer tradeRecords(
			context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			comboVec.size() * sizeof(tradeRecord), &tradeRecordsVec[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for tradeRecords: " << err << endl;
		return 1;
	}
	cl::Buffer indicatorBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
														 INCREMENT * sizeof(indicators), NULL, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for indicatorBuffer: " << err << endl;
		return 1;
	}
	size_t indicatorsSize = INCREMENT * sizeof(indicators);

	vector<entryAndExit> entriesAndExitsVec;
	cl::Buffer entriesAndExitsBuf;

	vector<drawdowns> drawdownsVec;
	cl::Buffer drawdownsBuf;
	vector<drawdownLengths> drawdownLengthsVec;
	cl::Buffer drawdownLengthsBuf;
	vector<lossStreaks> lossStreaksVec;
	cl::Buffer lossStreaksBuf;
	vector<tradeDurations> tradeDurationsVec;
	cl::Buffer tradeDurationsBuf;
	vector<monthlyReturns> monthlyReturnsVec;
	cl::Buffer monthlyReturnsBuf;

	vector<cl_int> numTradesInIntervalVec;
	cl::Buffer numTradesInIntervalBuf;

	cout << "Number of combinations: " << comboVec.size() << endl;
	cout << "Size of combinations: " << comboVecSize << endl;
	cout << "Size of trades: " << inputTradesSize << endl;
	cout << "Size of indicators: " << indicatorsSize << endl;
	cout << "Size of entries: " << entriesVecSize << endl;
	cout << "Size of trade records: " << tradeRecordsVecSize << endl;
	size_t totalSize = comboVecSize + inputTradesSize + indicatorsSize +
										 entriesVecSize + tradeRecordsVecSize;
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

		if (snapshotting && snapshotFile.good()) {
			if (!snapshotFile.read(reinterpret_cast<char *>(drawdownsVec.data()),
														 comboVec.size() * sizeof(drawdowns))) {
				cout << "Failed to read drawdowns" << endl;
				return 1;
			}
			if (!snapshotFile.read(
							reinterpret_cast<char *>(drawdownLengthsVec.data()),
							comboVec.size() * sizeof(drawdownLengths))) {
				cout << "Failed to read drawdown lengths" << endl;
				return 1;
			}
			if (!snapshotFile.read(reinterpret_cast<char *>(lossStreaksVec.data()),
														 comboVec.size() * sizeof(lossStreaks))) {
				cout << "Failed to read loss streaks" << endl;
				return 1;
			}
			if (!snapshotFile.read(reinterpret_cast<char *>(tradeDurationsVec.data()),
														 comboVec.size() * sizeof(tradeDurations))) {
				cout << "Failed to read trade durations" << endl;
				return 1;
			}
			if (!snapshotFile.read(reinterpret_cast<char *>(monthlyReturnsVec.data()),
														 comboVec.size() * sizeof(monthlyReturns))) {
				cout << "Failed to read monthly returns" << endl;
				return 1;
			}
		}

		drawdownsBuf = cl::Buffer(
				context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				comboVec.size() * sizeof(drawdowns), &drawdownsVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for drawdownsBuf: " << err << endl;
			return 1;
		}

		drawdownLengthsBuf = cl::Buffer(context,
																		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY |
																				CL_MEM_COPY_HOST_PTR,
																		comboVec.size() * sizeof(drawdownLengths),
																		&drawdownLengthsVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for drawdownLengthsBuf: " << err << endl;
			return 1;
		}

		lossStreaksBuf = cl::Buffer(
				context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				comboVec.size() * sizeof(lossStreaks), &lossStreaksVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for lossStreaksBuf: " << err << endl;
			return 1;
		}

		tradeDurationsBuf = cl::Buffer(
				context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				comboVec.size() * sizeof(tradeDurations), &tradeDurationsVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for tradeDurationsBuf: " << err << endl;
			return 1;
		}

		monthlyReturnsBuf = cl::Buffer(
				context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				comboVec.size() * sizeof(monthlyReturns), &monthlyReturnsVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for monthlyReturnsBuf: " << err << endl;
			return 1;
		}
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
		if (snapshotting && snapshotFile.good()) {
			if (!snapshotFile.read(
							reinterpret_cast<char *>(numTradesInIntervalVec.data()),
							comboVec.size() * sizeof(cl_int))) {
				cout << "Failed to read number of trades in interval" << endl;
				return 1;
			}
		}
		numTradesInIntervalBuf = cl::Buffer(
				context,
				CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				comboVec.size() * sizeof(cl_int), &numTradesInIntervalVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for numTradesInIntervalBuf: " << err << endl;
			return 1;
		}
		totalSize += numTradesInIntervalVecSize;
		cout << "Total size: " << totalSize << endl;
		cout << "Total size: " << (double)totalSize / (double)(1024 * 1024 * 1024)
				 << " GiB" << endl;
	}
	if (snapshotFile.good())
		snapshotFile.close();

	err = volKernel.setArg(0, inputSize);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 0: " << err << endl;
	}
	err = volKernel.setArg(1, inputTrades);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 1: " << err << endl;
	}
	err = volKernel.setArg(2, indicatorBuffer);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 2: " << err << endl;
	}
	err = volKernel.setArg(3, inputCombos);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 3: " << err << endl;
	}
	err = volKernel.setArg(4, entries);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 4: " << err << endl;
	}
	err = volKernel.setArg(5, tradeRecords);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 5: " << err << endl;
	}

	if (listTrades) {
		err = volKernel.setArg(6, drawdownsBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel setArg 6: " << err << endl;
		}
		err = volKernel.setArg(7, drawdownLengthsBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel setArg 7: " << err << endl;
		}
		err = volKernel.setArg(8, lossStreaksBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel setArg 8: " << err << endl;
		}
		err = volKernel.setArg(9, tradeDurationsBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel setArg 9: " << err << endl;
		}
		err = volKernel.setArg(10, monthlyReturnsBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel setArg 10: " << err << endl;
		}
	} else {
		err = volKernel.setArg(6, numTradesInIntervalBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel setArg 6: " << err << endl;
		}
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
	bool startedCollectingTrades = false;
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

				if (!snapshotting ||
						(snapshotting && currIdx >= globalIdx - newStart)) {
#ifdef DEBUG
					startedCollectingTrades = true;
#endif
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
				} else {
#ifdef DEBUG
					if (startedCollectingTrades) {
						cout << "Gap in collecting trades!" << endl;
						cout << "currIdx: " << currIdx << endl;
						cout << "globalIdx: " << globalIdx << endl;
						cout << "newStart: " << newStart << endl;
					}
#endif
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
						// processTradesWithListingAndIndicators(
						//		queue, volKernel, tradesWithoutDates, indicators, tws,
						//		comboVec, inputTrades, indicatorBuffer, inputSize,
						//		entriesAndExitsBuf, allTrades);
						processTradesWithOnlineAlgs(
								queue, volKernel, tradesWithoutDates, indicators, tws, comboVec,
								inputTrades, indicatorBuffer, inputSize, entries, tradeRecords,
								drawdownsBuf, drawdownLengthsBuf, lossStreaksBuf,
								tradeDurationsBuf, monthlyReturnsBuf, entriesVec,
								tradeRecordsVec, drawdownsVec, drawdownLengthsVec,
								lossStreaksVec, tradeDurationsVec, monthlyReturnsVec,
								snapshotting);
					else
						maxTradesPerInterval = max(
								maxTradesPerInterval,
								// processTrades(queue, volKernel, tradesWithoutDates,
								// comboVec, 							inputTrades, inputSize,
								// twBetweenRunData, 							positionDatas,
								// numTradesInIntervalBuf));
								processTradesWithIndicators(
										queue, volKernel, tradesWithoutDates, indicators, tws,
										comboVec, inputTrades, indicatorBuffer, inputSize, entries,
										tradeRecords, numTradesInIntervalBuf, entriesVec,
										tradeRecordsVec, numTradesInIntervalVec, snapshotting));
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
						queue, volKernel, tradesWithoutDates, indicators, tws, comboVec,
						inputTrades, indicatorBuffer, inputSize, entries, tradeRecords,
						drawdownsBuf, drawdownLengthsBuf, lossStreaksBuf, tradeDurationsBuf,
						monthlyReturnsBuf, entriesVec, tradeRecordsVec, drawdownsVec,
						drawdownLengthsVec, lossStreaksVec, tradeDurationsVec,
						monthlyReturnsVec, snapshotting);
			else
				maxTradesPerInterval =
						max(maxTradesPerInterval,
								// processTrades(queue, volKernel, tradesWithoutDates, comboVec,
								// 							inputTrades, inputSize, twBetweenRunData,
								// 							positionDatas, numTradesInIntervalBuf));
								processTradesWithIndicators(
										queue, volKernel, tradesWithoutDates, indicators, tws,
										comboVec, inputTrades, indicatorBuffer, inputSize, entries,
										tradeRecords, numTradesInIntervalBuf, entriesVec,
										tradeRecordsVec, numTradesInIntervalVec, snapshotting));
		} else
			cout << "Run was already finished in the snapshot" << endl;
	}

	err = queue.enqueueReadBuffer(tradeRecords, CL_FALSE, 0,
																comboVec.size() * sizeof(tradeRecord),
																&tradeRecordsVec[0]);
	if (err != CL_SUCCESS) {
		cout << "Error for enqueueReadBuffer tradeRecords: " << err << endl;
		return 1;
	}

	if (listTrades) {
		err = queue.enqueueReadBuffer(drawdownsBuf, CL_FALSE, 0,
																	comboVec.size() * sizeof(drawdowns),
																	&drawdownsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer drawdownsBuf: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(drawdownLengthsBuf, CL_FALSE, 0,
																	comboVec.size() * sizeof(drawdownLengths),
																	&drawdownLengthsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer drawdownLengthsBuf: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(lossStreaksBuf, CL_FALSE, 0,
																	comboVec.size() * sizeof(lossStreaks),
																	&lossStreaksVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer lossStreaksBuf: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(tradeDurationsBuf, CL_FALSE, 0,
																	comboVec.size() * sizeof(tradeDurations),
																	&tradeDurationsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer tradeDurationsBuf: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(monthlyReturnsBuf, CL_FALSE, 0,
																	comboVec.size() * sizeof(monthlyReturns),
																	&monthlyReturnsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer monthlyReturnsBuf: " << err << endl;
			return 1;
		}
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
	cout << endl;
	cout << "Time taken to run kernel: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	string resultsFilename = "resultsAllVol";
	if (isBTC)
		resultsFilename += "BTC";
	else
		resultsFilename += "ETH";

	if (listTrades) {
		resultsFilename += "Detailed";
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
			int maxElementIdx =
					max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
											[](tradeRecord t1, tradeRecord t2) {
												return t1.capital < t2.capital;
											}) -
					tradeRecordsVec.begin();
			outFile << "Max return:" << endl;
			outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
										allPerfMetrics, drawdownsVec, drawdownLengthsVec,
										lossStreaksVec, tradeDurationsVec, listTrades);
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
			outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
										allPerfMetrics, drawdownsVec, drawdownLengthsVec,
										lossStreaksVec, tradeDurationsVec, listTrades);
			outFile << endl;

			maxElementIdx =
					max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
											[](tradeRecord t1, tradeRecord t2) {
												return t1.shorts + t1.longs < t2.shorts + t2.longs;
											}) -
					tradeRecordsVec.begin();
			outFile << "Most trades:" << endl;
			outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
										allPerfMetrics, drawdownsVec, drawdownLengthsVec,
										lossStreaksVec, tradeDurationsVec, listTrades);
			outFile << endl;

			if (listTrades) {
				maxElementIdx =
						max_element(allPerfMetrics.begin(), allPerfMetrics.end(),
												[](perfMetrics p1, perfMetrics p2) {
													return p1.sharpe < p2.sharpe;
												}) -
						allPerfMetrics.begin();
				outFile << "Best sharpe ratio:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
											allPerfMetrics, drawdownsVec, drawdownLengthsVec,
											lossStreaksVec, tradeDurationsVec, listTrades);
				outFile << endl;

				maxElementIdx = max_element(drawdownsVec.begin(), drawdownsVec.end(),
																		[](drawdowns d1, drawdowns d2) {
																			return d1.mean < d2.mean;
																		}) -
												drawdownsVec.begin();
				outFile << "Smallest average drawdown:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
											allPerfMetrics, drawdownsVec, drawdownLengthsVec,
											lossStreaksVec, tradeDurationsVec, listTrades);
				outFile << endl;

				maxElementIdx = max_element(drawdownsVec.begin(), drawdownsVec.end(),
																		[](drawdowns d1, drawdowns d2) {
																			return d1.max < d2.max;
																		}) -
												drawdownsVec.begin();
				outFile << "Smallest max drawdown:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
											allPerfMetrics, drawdownsVec, drawdownLengthsVec,
											lossStreaksVec, tradeDurationsVec, listTrades);
				outFile << endl;

				maxElementIdx =
						min_element(drawdownLengthsVec.begin(), drawdownLengthsVec.end(),
												[](drawdownLengths d1, drawdownLengths d2) {
													return d1.mean < d2.mean;
												}) -
						drawdownLengthsVec.begin();
				outFile << "Shortest average drawdown length:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
											allPerfMetrics, drawdownsVec, drawdownLengthsVec,
											lossStreaksVec, tradeDurationsVec, listTrades);
				outFile << endl;

				maxElementIdx =
						min_element(lossStreaksVec.begin(), lossStreaksVec.end(),
												[](lossStreaks l1, lossStreaks l2) {
													return l1.mean < l2.mean;
												}) -
						lossStreaksVec.begin();
				outFile << "Shortest average loss streak:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
											allPerfMetrics, drawdownsVec, drawdownLengthsVec,
											lossStreaksVec, tradeDurationsVec, listTrades);
				outFile << endl;

				maxElementIdx =
						min_element(lossStreaksVec.begin(), lossStreaksVec.end(),
												[](lossStreaks l1, lossStreaks l2) {
													return l1.max < l2.max;
												}) -
						lossStreaksVec.begin();
				outFile << "Shortest max loss streak:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
											allPerfMetrics, drawdownsVec, drawdownLengthsVec,
											lossStreaksVec, tradeDurationsVec, listTrades);
				outFile << endl;

				maxElementIdx =
						min_element(tradeDurationsVec.begin(), tradeDurationsVec.end(),
												[](tradeDurations t1, tradeDurations t2) {
													return t1.mean < t2.mean;
												}) -
						tradeDurationsVec.begin();
				outFile << "Shortest average trade duration:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
											allPerfMetrics, drawdownsVec, drawdownLengthsVec,
											lossStreaksVec, tradeDurationsVec, listTrades);
				outFile << endl;

				maxElementIdx =
						max_element(tradeDurationsVec.begin(), tradeDurationsVec.end(),
												[](tradeDurations t1, tradeDurations t2) {
													return t1.mean < t2.mean;
												}) -
						tradeDurationsVec.begin();
				outFile << "Longest average trade duration:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
											allPerfMetrics, drawdownsVec, drawdownLengthsVec,
											lossStreaksVec, tradeDurationsVec, listTrades);
				outFile << endl;
			}

			for (size_t i = 0; i < comboVec.size(); i++) {
				outputMetrics(outFile, i, comboVec, tradeRecordsVec, allPerfMetrics,
											drawdownsVec, drawdownLengthsVec, lossStreaksVec,
											tradeDurationsVec, listTrades);
				/*
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
				*/
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

	int maxElementIdx =
			max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
									[](tradeRecord t1, tradeRecord t2) {
										return t1.capital < t2.capital;
									}) -
			tradeRecordsVec.begin();
	cout << "Max return:" << endl;
	outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec, allPerfMetrics,
								drawdownsVec, drawdownLengthsVec, lossStreaksVec,
								tradeDurationsVec, listTrades);
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
	outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec, allPerfMetrics,
								drawdownsVec, drawdownLengthsVec, lossStreaksVec,
								tradeDurationsVec, listTrades);
	cout << endl;

	maxElementIdx =
			max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
									[](tradeRecord t1, tradeRecord t2) {
										return t1.shorts + t1.longs < t2.shorts + t2.longs;
									}) -
			tradeRecordsVec.begin();
	cout << "Most trades:" << endl;
	outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec, allPerfMetrics,
								drawdownsVec, drawdownLengthsVec, lossStreaksVec,
								tradeDurationsVec, listTrades);
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
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, listTrades);
		cout << endl;

		maxElementIdx = max_element(drawdownsVec.begin(), drawdownsVec.end(),
																[](drawdowns d1, drawdowns d2) {
																	return d1.mean < d2.mean;
																}) -
										drawdownsVec.begin();
		cout << "Smallest average drawdown:" << endl;
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, listTrades);
		cout << endl;

		maxElementIdx = max_element(drawdownsVec.begin(), drawdownsVec.end(),
																[](drawdowns d1, drawdowns d2) {
																	return d1.max < d2.max;
																}) -
										drawdownsVec.begin();
		cout << "Smallest max drawdown:" << endl;
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, listTrades);
		cout << endl;

		maxElementIdx =
				min_element(drawdownLengthsVec.begin(), drawdownLengthsVec.end(),
										[](drawdownLengths d1, drawdownLengths d2) {
											return d1.mean < d2.mean;
										}) -
				drawdownLengthsVec.begin();
		cout << "Shortest average drawdown length:" << endl;
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, listTrades);
		cout << endl;

		maxElementIdx = min_element(lossStreaksVec.begin(), lossStreaksVec.end(),
																[](lossStreaks l1, lossStreaks l2) {
																	return l1.mean < l2.mean;
																}) -
										lossStreaksVec.begin();
		cout << "Shortest average loss streak:" << endl;
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, listTrades);
		cout << endl;

		maxElementIdx = min_element(lossStreaksVec.begin(), lossStreaksVec.end(),
																[](lossStreaks l1, lossStreaks l2) {
																	return l1.max < l2.max;
																}) -
										lossStreaksVec.begin();
		cout << "Shortest max loss streak:" << endl;
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, listTrades);
		cout << endl;

		maxElementIdx =
				min_element(tradeDurationsVec.begin(), tradeDurationsVec.end(),
										[](tradeDurations t1, tradeDurations t2) {
											return t1.mean < t2.mean;
										}) -
				tradeDurationsVec.begin();
		cout << "Shortest average trade duration:" << endl;
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, listTrades);
		cout << endl;

		maxElementIdx =
				max_element(tradeDurationsVec.begin(), tradeDurationsVec.end(),
										[](tradeDurations t1, tradeDurations t2) {
											return t1.mean < t2.mean;
										}) -
				tradeDurationsVec.begin();
		cout << "Longest average trade duration:" << endl;
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, listTrades);
		cout << endl;
	}

	auto endTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(endTime - startTime);
	cout << "Total time taken: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	return 0;
}
