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
using std::max;
using std::max_element;
using std::min;
using std::min_element;
using std::mktime;
using std::ofstream;
using std::stod;
using std::stoi;
using std::string;
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
static string snapshotFilename;

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

void saveSnapshot(size_t index, size_t start, const vector<entry> &entriesVec,
									const vector<tradeRecord> &tradeRecordsVec,
									const vector<positionData> &positionDatasVec,
									const vector<cl_int> &numTradesInIntervalVec, Mode mode) {
	ofstream file(snapshotFilename, ios::binary);
	if (!file) {
		cout << "Error opening snapshot file" << endl;
		return;
	}

	file.write(reinterpret_cast<const char *>(&index), sizeof(index));

	switch (mode) {
	case Mode::Volume:
	case Mode::VolumeDelta:
	case Mode::VolumeExit:
	case Mode::AllVolume:
	case Mode::VolumeTrend:
		file.write(reinterpret_cast<const char *>(&start), sizeof(start));
		break;
	case Mode::Trend:
		break;
	}

	file.write(reinterpret_cast<const char *>(entriesVec.data()),
						 entriesVec.size() * sizeof(entry));
	file.write(reinterpret_cast<const char *>(tradeRecordsVec.data()),
						 tradeRecordsVec.size() * sizeof(tradeRecord));

	switch (mode) {
	case Mode::Volume:
	case Mode::VolumeDelta:
	case Mode::VolumeExit:
	case Mode::AllVolume:
		break;
	case Mode::Trend:
	case Mode::VolumeTrend:
		file.write(reinterpret_cast<const char *>(positionDatasVec.data()),
							 positionDatasVec.size() * sizeof(positionData));
		break;
	}

	file.write(reinterpret_cast<const char *>(numTradesInIntervalVec.data()),
						 numTradesInIntervalVec.size() * sizeof(cl_int));
	file.close();
}

void saveSnapshot(size_t index, size_t start, const vector<entry> &entriesVec,
									const vector<tradeRecord> &tradeRecordsVec,
									const vector<positionData> &positionDatasVec,
									vector<drawdowns> &drawdownsVec,
									vector<drawdownLengths> &drawdownLengthsVec,
									vector<lossStreaks> &lossStreaksVec,
									vector<tradeDurations> &tradeDurationsVec,
									vector<monthlyReturns> &monthlyReturnsVec,
									vector<wins> &winsVec, vector<losses> &lossesVec, Mode mode) {
	ofstream file(snapshotFilename, ios::binary);
	if (!file) {
		cout << "Error opening snapshot file" << endl;
		return;
	}

	file.write(reinterpret_cast<const char *>(&index), sizeof(index));

	switch (mode) {
	case Mode::Volume:
	case Mode::VolumeDelta:
	case Mode::VolumeExit:
	case Mode::AllVolume:
	case Mode::VolumeTrend:
		file.write(reinterpret_cast<const char *>(&start), sizeof(start));
		break;
	case Mode::Trend:
		break;
	}

	file.write(reinterpret_cast<const char *>(entriesVec.data()),
						 entriesVec.size() * sizeof(entry));
	file.write(reinterpret_cast<const char *>(tradeRecordsVec.data()),
						 tradeRecordsVec.size() * sizeof(tradeRecord));

	switch (mode) {
	case Mode::Volume:
	case Mode::VolumeDelta:
	case Mode::VolumeExit:
	case Mode::AllVolume:
		break;
	case Mode::Trend:
	case Mode::VolumeTrend:
		file.write(reinterpret_cast<const char *>(positionDatasVec.data()),
							 positionDatasVec.size() * sizeof(positionData));
		break;
	}

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
	file.write(reinterpret_cast<const char *>(winsVec.data()),
						 winsVec.size() * sizeof(wins));
	file.write(reinterpret_cast<const char *>(lossesVec.data()),
						 lossesVec.size() * sizeof(losses));
	file.close();
}

int processTrades(
		const cl::CommandQueue &queue, const cl::Kernel &kernel,
		vector<tradeWithoutDate> &tradesWithoutDates, vector<indicators> &inds,
		vector<timeWindow> &tws, const size_t numCombos,
		const cl::Buffer &inputTrades, const cl::Buffer &inputSize,
		const cl::Buffer &entries, const cl::Buffer &tradeRecords,
		const cl::Buffer &indicatorBuffer, const cl::Buffer &positionDatas,
		const cl::Buffer &numTradesInIntervalBuf, vector<entry> &entriesVec,
		vector<tradeRecord> &tradeRecordsVec,
		vector<positionData> &positionDatasVec,
		vector<cl_int> &numTradesInIntervalVec, bool snapshotting, Mode mode) {
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
		err = queue.enqueueWriteBuffer(inputSize, CL_FALSE, 0, sizeof(int),
																	 &currSize);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputSize: " << err << endl;
			return -1;
		}

		switch (mode) {
		case Mode::Volume:
		case Mode::VolumeDelta:
		case Mode::VolumeExit:
		case Mode::AllVolume:
		case Mode::VolumeTrend:
			err = queue.enqueueWriteBuffer(indicatorBuffer, CL_FALSE, 0,
																		 currSize * sizeof(indicators),
																		 &inds[currIdx]);
			if (err != CL_SUCCESS) {
				cout << "Error for enqueueWriteBuffer indicatorBuffer: " << err << endl;
				return -1;
			}
			break;
		case Mode::Trend:
			break;
		}

		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << " (" << globalIdx << "-" << globalIdx + currSize - 1 << ")" << endl;
		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange,
																		 cl::NDRange(numCombos));
		if (err != CL_SUCCESS) {
			cout << "Error for kernel: " << err << endl;
			return -1;
		}
		err = queue.enqueueReadBuffer(numTradesInIntervalBuf, CL_FALSE, 0,
																	numCombos * sizeof(cl_int),
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
		err = queue.enqueueReadBuffer(entries, CL_FALSE, 0,
																	numCombos * sizeof(entry), &entriesVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer entries: " << err << endl;
			return -1;
		}
		err = queue.enqueueReadBuffer(tradeRecords, CL_FALSE, 0,
																	numCombos * sizeof(tradeRecord),
																	&tradeRecordsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer tradeRecords: " << err << endl;
			return -1;
		}

		switch (mode) {
		case Mode::Volume:
		case Mode::VolumeDelta:
		case Mode::VolumeExit:
		case Mode::AllVolume:
			break;
		case Mode::Trend:
		case Mode::VolumeTrend:
			err = queue.enqueueReadBuffer(positionDatas, CL_FALSE, 0,
																		numCombos * sizeof(positionData),
																		&positionDatasVec[0]);
			if (err != CL_SUCCESS) {
				cout << "Error for enqueueReadBuffer positionDatas: " << err << endl;
				return -1;
			}
			break;
		}

		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return -1;
		}
		saveSnapshot(globalIdx, newStart, entriesVec, tradeRecordsVec,
								 positionDatasVec, numTradesInIntervalVec, mode);
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
		vector<timeWindow> &tws, const size_t numCombos,
		const cl::Buffer &inputTrades, const cl::Buffer &inputSize,
		const cl::Buffer &entries, const cl::Buffer &tradeRecords,
		const cl::Buffer &indicatorBuffer, const cl::Buffer &positionDatas,
		const cl::Buffer &drawdownsBuf, const cl::Buffer &drawdownLengthsBuf,
		const cl::Buffer &lossStreaksBuf, const cl::Buffer &tradeDurationsBuf,
		const cl::Buffer &monthlyReturnsBuf, const cl::Buffer &winsBuf,
		const cl::Buffer &lossesBuf, vector<entry> &entriesVec,
		vector<tradeRecord> &tradeRecordsVec,
		vector<positionData> &positionDatasVec, vector<drawdowns> &drawdownsVec,
		vector<drawdownLengths> &drawdownLengthsVec,
		vector<lossStreaks> &lossStreaksVec,
		vector<tradeDurations> &tradeDurationsVec,
		vector<monthlyReturns> &monthlyReturnsVec, vector<wins> &winsVec,
		vector<losses> &lossesVec, bool snapshotting, Mode mode) {
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
		err = queue.enqueueWriteBuffer(inputSize, CL_FALSE, 0, sizeof(int),
																	 &currSize);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputSize: " << err << endl;
			return;
		}

		switch (mode) {
		case Mode::Volume:
		case Mode::VolumeDelta:
		case Mode::VolumeExit:
		case Mode::AllVolume:
		case Mode::VolumeTrend:
			err = queue.enqueueWriteBuffer(indicatorBuffer, CL_FALSE, 0,
																		 currSize * sizeof(indicators),
																		 &inds[currIdx]);
			if (err != CL_SUCCESS) {
				cout << "Error for enqueueWriteBuffer indicatorBuffer: " << err << endl;
				return;
			}
			break;
		case Mode::Trend:
			break;
		}

		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << " (" << globalIdx << "-" << globalIdx + currSize - 1 << ")" << endl;
		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange,
																		 cl::NDRange(numCombos));
		if (err != CL_SUCCESS) {
			cout << "Error for kernel: " << err << endl;
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
		err = queue.enqueueReadBuffer(entries, CL_FALSE, 0,
																	numCombos * sizeof(entry), &entriesVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer entries: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(tradeRecords, CL_FALSE, 0,
																	numCombos * sizeof(tradeRecord),
																	&tradeRecordsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer tradeRecords: " << err << endl;
			return;
		}

		switch (mode) {
		case Mode::Volume:
		case Mode::VolumeDelta:
		case Mode::VolumeExit:
		case Mode::AllVolume:
			break;
		case Mode::Trend:
		case Mode::VolumeTrend:
			err = queue.enqueueReadBuffer(positionDatas, CL_FALSE, 0,
																		numCombos * sizeof(positionData),
																		&positionDatasVec[0]);
			if (err != CL_SUCCESS) {
				cout << "Error for enqueueReadBuffer positionDatas: " << err << endl;
				return;
			}
			break;
		}

		err = queue.enqueueReadBuffer(drawdownsBuf, CL_FALSE, 0,
																	numCombos * sizeof(drawdowns),
																	&drawdownsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer drawdownsBuf: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(drawdownLengthsBuf, CL_FALSE, 0,
																	numCombos * sizeof(drawdownLengths),
																	&drawdownLengthsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer drawdownLengthsBuf: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(lossStreaksBuf, CL_FALSE, 0,
																	numCombos * sizeof(lossStreaks),
																	&lossStreaksVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer lossStreaksBuf: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(tradeDurationsBuf, CL_FALSE, 0,
																	numCombos * sizeof(tradeDurations),
																	&tradeDurationsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer tradeDurationsBuf: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(monthlyReturnsBuf, CL_FALSE, 0,
																	numCombos * sizeof(monthlyReturns),
																	&monthlyReturnsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer monthlyReturnsBuf: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(winsBuf, CL_FALSE, 0,
																	numCombos * sizeof(wins), &winsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer winsBuf: " << err << endl;
			return;
		}
		err = queue.enqueueReadBuffer(lossesBuf, CL_FALSE, 0,
																	numCombos * sizeof(losses), &lossesVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer lossesBuf: " << err << endl;
			return;
		}
		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return;
		}
		saveSnapshot(globalIdx, newStart, entriesVec, tradeRecordsVec,
								 positionDatasVec, drawdownsVec, drawdownLengthsVec,
								 lossStreaksVec, tradeDurationsVec, monthlyReturnsVec, winsVec,
								 lossesVec, mode);
		auto duration = duration_cast<microseconds>(high_resolution_clock::now() -
																								beforeSnapshotTime);
		cout << "Time taken to take snapshot: "
				 << (double)duration.count() / 1000000 << " seconds" << endl;
	}

	return;
}

int main(int argc, char *argv[]) {
	bool writeResults = false, listTrades = false, snapshotting = false,
			 futures = false;

	bool isBTC;
	Mode mode;
	string modeString;

	int opt;
	while ((opt = getopt(argc, argv, ":wlsft:")) != -1) {
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
	snapshotFilename = "snapshot" + modeString;
	if (symbol.find("BTC") != string::npos) {
		isBTC = true;
		snapshotFilename += "BTC";
	} else {
		isBTC = false;
		snapshotFilename += "ETH";
	}
	if (futures)
		snapshotFilename += "Futures";
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

	initializeDevice("trader.cl");

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

	int y = 10;
	vector<int> periodsAfterOpen(11);
	generate(periodsAfterOpen.begin(), periodsAfterOpen.end(),
					 [y]() mutable { return y += 1; });

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

	ComboVec comboVec;
	size_t comboVecSize;
	size_t numCombos;

	switch (mode) {
	case Mode::Volume:
		comboVec = vector<comboVol>{};
		break;
	case Mode::VolumeDelta:
		comboVec = vector<comboVolDelta>{};
		break;
	case Mode::VolumeExit:
		comboVec = vector<comboVolExit>{};
		break;
	case Mode::AllVolume:
		comboVec = vector<comboAllVol>{};
		break;
	case Mode::Trend:
		comboVec = vector<comboTrend>{};
		break;
	case Mode::VolumeTrend:
		comboVec = vector<comboVolTrend>{};
		break;
	}

	auto visitor = [&indWindows, &buyVolPercentiles, &sellVolPercentiles,
									&stopLosses, &targets, &deltaVolPercentiles, &entryThresholds,
									&comboVecSize, &numCombos, &contractMultipliers,
									&periodsAfterOpen](auto &comboVec) {
		using T = typename decay_t<decltype(comboVec)>::value_type;
		if constexpr (is_same_v<T, comboVol>) {
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
					comboVol c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
												get<2>(combos[i][j]), get<3>(combos[i][j]),
												get<4>(combos[i][j]), get<5>(combos[i][j]),
												get<6>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1)
						comboVec.emplace_back(c);
				}
			}
			comboVecSize = comboVec.size() * sizeof(comboVol);
		} else if constexpr (is_same_v<T, comboVolDelta>) {
			auto firstCombo =
					cartesian_product(indWindows[0], deltaVolPercentiles[0],
														deltaVolPercentiles[0], stopLosses, targets);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(
						cartesian_product(indWindows[i], deltaVolPercentiles[i],
															deltaVolPercentiles[i], stopLosses, targets));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboVolDelta c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
														 get<2>(combos[i][j]), get<3>(combos[i][j]),
														 get<4>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1 &&
							c.buyVolDeltaPercentile >= c.sellVolDeltaPercentile)
						comboVec.emplace_back(c);
				}
			}
			comboVecSize = comboVec.size() * sizeof(comboVolDelta);
		} else if constexpr (is_same_v<T, comboVolExit>) {
			auto firstCombo = cartesian_product(
					indWindows[0], buyVolPercentiles[0], sellVolPercentiles[0],
					buyVolPercentiles[0], sellVolPercentiles[0]);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i],
						buyVolPercentiles[i], sellVolPercentiles[i]));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboVolExit c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
														get<2>(combos[i][j]), get<3>(combos[i][j]),
														get<4>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.buyVolPercentile > c.buyVolExitPercentile &&
							c.sellVolPercentile > c.sellVolExitPercentile)
						comboVec.emplace_back(c);
				}
			}
			comboVecSize = comboVec.size() * sizeof(comboVolExit);
		} else if constexpr (is_same_v<T, comboAllVol>) {
			auto firstCombo = cartesian_product(
					indWindows[0], buyVolPercentiles[0], sellVolPercentiles[0],
					deltaVolPercentiles[0], deltaVolPercentiles[0], stopLosses, targets);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(
						cartesian_product(indWindows[i], buyVolPercentiles[i],
															sellVolPercentiles[i], deltaVolPercentiles[i],
															deltaVolPercentiles[i], stopLosses, targets));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboAllVol c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
													 get<2>(combos[i][j]), get<3>(combos[i][j]),
													 get<4>(combos[i][j]), get<5>(combos[i][j]),
													 get<6>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1 &&
							c.buyVolPercentile >= c.buyVolDeltaPercentile &&
							c.sellVolPercentile >= -c.sellVolDeltaPercentile &&
							c.buyVolDeltaPercentile >= 0 && c.sellVolDeltaPercentile <= 0)
						comboVec.emplace_back(c);
				}
			}
			comboVecSize = comboVec.size() * sizeof(comboAllVol);
		} else if constexpr (is_same_v<T, comboTrend>) {
			auto combos = cartesian_product(entryThresholds, stopLosses, targets);

			for (unsigned int j = 0; j < combos.size(); j++) {
				comboTrend c = {get<0>(combos[j]), get<1>(combos[j]),
												get<2>(combos[j])};
				// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
				// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
				if (c.stopLoss < c.target + 1.1)
					comboVec.emplace_back(c);
			}
			comboVecSize = comboVec.size() * sizeof(comboTrend);
		} else if constexpr (is_same_v<T, comboVolTrend>) {
			auto firstCombo = cartesian_product(indWindows[0], buyVolPercentiles[0],
																					sellVolPercentiles[0],
																					entryThresholds, stopLosses, targets);
			vector<decltype(firstCombo)> combos;
			combos.emplace_back(firstCombo);

			for (int i = 1; i < NUM_WINDOWS; i++) {
				combos.emplace_back(cartesian_product(
						indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i],
						entryThresholds, stopLosses, targets));
			}

			for (int i = 0; i < NUM_WINDOWS; i++) {
				for (unsigned int j = 0; j < combos[i].size(); j++) {
					comboVolTrend c = {get<0>(combos[i][j]), get<1>(combos[i][j]),
														 get<2>(combos[i][j]), get<3>(combos[i][j]),
														 get<4>(combos[i][j]), get<5>(combos[i][j])};
					// cout << c.window << " " << c.target << " " << c.stopLoss << " " <<
					// c.buyVolPercentile << " " << c.sellVolPercentile << endl;
					if (c.stopLoss < c.target + 1.1)
						comboVec.emplace_back(c);
				}
			}
			comboVecSize = comboVec.size() * sizeof(comboVolTrend);
		}

		numCombos = comboVec.size();
	};

	visit(visitor, comboVec);

	vector<positionData> positionDatasVec;
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

	ifstream snapshotFile;
#ifdef DEBUG
	bool existingSnapshot;
#endif
	if (snapshotting) {
		snapshotFile.open(snapshotFilename, ios::binary);
		if (snapshotFile.good()) {
#ifdef DEBUG
			existingSnapshot = true;
#endif
			if (!snapshotFile.read(reinterpret_cast<char *>(&globalIdx),
														 sizeof(globalIdx))) {
				cout << "Failed to read previous index" << endl;
				return 1;
			}
			cout << "globalIdx: " << globalIdx << endl;

			switch (mode) {
			case Mode::Volume:
			case Mode::VolumeDelta:
			case Mode::VolumeExit:
			case Mode::AllVolume:
			case Mode::VolumeTrend:
				if (!snapshotFile.read(reinterpret_cast<char *>(&newStart),
															 sizeof(newStart))) {
					cout << "Failed to read new start" << endl;
					return 1;
				}
				cout << "newStart: " << newStart << endl;
				break;
			case Mode::Trend:
				break;
			}

			if (!snapshotFile.read(reinterpret_cast<char *>(entriesVec.data()),
														 numCombos * sizeof(entry))) {
				cout << "Failed to read entries" << endl;
				return 1;
			}
			if (!snapshotFile.read(reinterpret_cast<char *>(tradeRecordsVec.data()),
														 numCombos * sizeof(tradeRecord))) {
				cout << "Failed to read trade records" << endl;
				return 1;
			}

			switch (mode) {
			case Mode::Volume:
			case Mode::VolumeDelta:
			case Mode::VolumeExit:
			case Mode::AllVolume:
				break;
			case Mode::Trend:
			case Mode::VolumeTrend:
				if (!snapshotFile.read(
								reinterpret_cast<char *>(positionDatasVec.data()),
								numCombos * sizeof(positionData))) {
					cout << "Failed to read position datas" << endl;
					return 1;
				}
				break;
			}
		} else {
			cout << "No snapshot file detected" << endl;
#ifdef DEBUG
			existingSnapshot = false;
#endif
		}
	}

	int maxTradesPerInterval = 0;

	cl_int err;

	cl::CommandQueue queue(context, device, 0, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for queue: " << err << endl;
		return 1;
	}

	cl::Kernel kernel;

	switch (mode) {
	case Mode::Volume:
		if (listTrades) {
			if (futures)
				kernel = cl::Kernel(program, "volTraderFuturesWithOnlineAlgs", &err);
			else
				kernel = cl::Kernel(program, "volTraderWithOnlineAlgs", &err);
		} else {
			if (futures)
				kernel = cl::Kernel(program, "volTraderFutures", &err);
			else
				kernel = cl::Kernel(program, "volTrader", &err);
		}
		break;
	case Mode::VolumeDelta:
		if (listTrades) {
			if (futures)
				kernel =
						cl::Kernel(program, "volDeltaTraderFuturesWithOnlineAlgs", &err);
			else
				kernel = cl::Kernel(program, "volDeltaTraderWithOnlineAlgs", &err);
		} else {
			if (futures)
				kernel = cl::Kernel(program, "volDeltaTraderFutures", &err);
			else
				kernel = cl::Kernel(program, "volDeltaTrader", &err);
		}
		break;
	case Mode::VolumeExit:
		if (listTrades) {
			if (futures)
				kernel =
						cl::Kernel(program, "volExitTraderFuturesWithOnlineAlgs", &err);
			else
				kernel = cl::Kernel(program, "volExitTraderWithOnlineAlgs", &err);
		} else {
			if (futures)
				kernel = cl::Kernel(program, "volExitTraderFutures", &err);
			else
				kernel = cl::Kernel(program, "volExitTrader", &err);
		}
		break;
	case Mode::AllVolume:
		if (listTrades) {
			if (futures)
				kernel = cl::Kernel(program, "allVolFuturesWithOnlineAlgs", &err);
			else
				kernel = cl::Kernel(program, "allVolWithOnlineAlgs", &err);
		} else {
			if (futures)
				kernel = cl::Kernel(program, "allVolFutures", &err);
			else
				kernel = cl::Kernel(program, "allVol", &err);
		}
		break;
	case Mode::Trend:
		if (listTrades) {
			if (futures)
				kernel =
						cl::Kernel(program, "trendFollowerFuturesWithOnlineAlgs", &err);
			else
				kernel = cl::Kernel(program, "trendFollowerWithOnlineAlgs", &err);
		} else {
			if (futures)
				kernel = cl::Kernel(program, "trendFollowerFutures", &err);
			else
				kernel = cl::Kernel(program, "trendFollower", &err);
		}
		break;
	case Mode::VolumeTrend:
		if (listTrades) {
			if (futures)
				kernel =
						cl::Kernel(program, "volTrendTraderFuturesWithOnlineAlgs", &err);
			else
				kernel = cl::Kernel(program, "volTrendTraderWithOnlineAlgs", &err);
		} else {
			if (futures)
				kernel = cl::Kernel(program, "volTrendTraderFutures", &err);
			else
				kernel = cl::Kernel(program, "volTrendTrader", &err);
		}
		break;
	}

	if (err != CL_SUCCESS) {
		cout << "Error for creating kernel: " << err << endl;
		return 1;
	}

	void *ptr =
			visit([](auto &vec) -> void * { return static_cast<void *>(vec.data()); },
						comboVec);

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
			comboVecSize, ptr, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputCombos: " << err << endl;
		return 1;
	}
	cl::Buffer entries(
			context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			numCombos * sizeof(entry), &entriesVec[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for entries: " << err << endl;
		return 1;
	}
	cl::Buffer tradeRecords(
			context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			numCombos * sizeof(tradeRecord), &tradeRecordsVec[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for tradeRecords: " << err << endl;
		return 1;
	}

	cl::Buffer positionDatas, indicatorBuffer;
	size_t indicatorsSize = 0;

	switch (mode) {
	case Mode::Volume:
	case Mode::VolumeDelta:
	case Mode::VolumeExit:
	case Mode::AllVolume:
		indicatorBuffer =
				cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
									 INCREMENT * sizeof(indicators), NULL, &err);
		if (err != CL_SUCCESS) {
			cout << "Error for indicatorBuffer: " << err << endl;
			return 1;
		}
		indicatorsSize = INCREMENT * sizeof(indicators);
		break;
	case Mode::Trend:
		positionDatas = cl::Buffer(
				context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				numCombos * sizeof(positionData), &positionDatasVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for positionDatas: " << err << endl;
			return 1;
		}
		break;
	case Mode::VolumeTrend:
		indicatorBuffer =
				cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
									 INCREMENT * sizeof(indicators), NULL, &err);
		if (err != CL_SUCCESS) {
			cout << "Error for indicatorBuffer: " << err << endl;
			return 1;
		}
		indicatorsSize = INCREMENT * sizeof(indicators);
		positionDatas = cl::Buffer(
				context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				numCombos * sizeof(positionData), &positionDatasVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for positionDatas: " << err << endl;
			return 1;
		}
		break;
	}

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
	vector<wins> winsVec;
	cl::Buffer winsBuf;
	vector<losses> lossesVec;
	cl::Buffer lossesBuf;

	vector<cl_int> numTradesInIntervalVec;
	cl::Buffer numTradesInIntervalBuf;

	cout << "Number of combinations: " << numCombos << endl;
	cout << "Size of combinations: " << comboVecSize << endl;
	cout << "Size of trades: " << inputTradesSize << endl;
	cout << "Size of indicators: " << indicatorsSize << endl;
	cout << "Size of entries: " << entriesVecSize << endl;
	cout << "Size of trade records: " << tradeRecordsVecSize << endl;
	cout << "Size of position datas: " << positionDatasVecSize << endl;
	size_t totalSize = comboVecSize + inputTradesSize + indicatorsSize +
										 entriesVecSize + tradeRecordsVecSize +
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

		if (snapshotting && snapshotFile.good()) {
			if (!snapshotFile.read(reinterpret_cast<char *>(drawdownsVec.data()),
														 numCombos * sizeof(drawdowns))) {
				cout << "Failed to read drawdowns" << endl;
				return 1;
			}
			if (!snapshotFile.read(
							reinterpret_cast<char *>(drawdownLengthsVec.data()),
							numCombos * sizeof(drawdownLengths))) {
				cout << "Failed to read drawdown lengths" << endl;
				return 1;
			}
			if (!snapshotFile.read(reinterpret_cast<char *>(lossStreaksVec.data()),
														 numCombos * sizeof(lossStreaks))) {
				cout << "Failed to read loss streaks" << endl;
				return 1;
			}
			if (!snapshotFile.read(reinterpret_cast<char *>(tradeDurationsVec.data()),
														 numCombos * sizeof(tradeDurations))) {
				cout << "Failed to read trade durations" << endl;
				return 1;
			}
			if (!snapshotFile.read(reinterpret_cast<char *>(monthlyReturnsVec.data()),
														 numCombos * sizeof(monthlyReturns))) {
				cout << "Failed to read monthly returns" << endl;
				return 1;
			}
			if (!snapshotFile.read(reinterpret_cast<char *>(winsVec.data()),
														 numCombos * sizeof(wins))) {
				cout << "Failed to read wins" << endl;
				return 1;
			}
			if (!snapshotFile.read(reinterpret_cast<char *>(lossesVec.data()),
														 numCombos * sizeof(losses))) {
				cout << "Failed to read losses" << endl;
				return 1;
			}
		}

		drawdownsBuf = cl::Buffer(
				context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				numCombos * sizeof(drawdowns), &drawdownsVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for drawdownsBuf: " << err << endl;
			return 1;
		}

		drawdownLengthsBuf = cl::Buffer(
				context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				numCombos * sizeof(drawdownLengths), &drawdownLengthsVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for drawdownLengthsBuf: " << err << endl;
			return 1;
		}

		lossStreaksBuf = cl::Buffer(
				context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				numCombos * sizeof(lossStreaks), &lossStreaksVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for lossStreaksBuf: " << err << endl;
			return 1;
		}

		tradeDurationsBuf = cl::Buffer(
				context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				numCombos * sizeof(tradeDurations), &tradeDurationsVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for tradeDurationsBuf: " << err << endl;
			return 1;
		}

		monthlyReturnsBuf = cl::Buffer(
				context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				numCombos * sizeof(monthlyReturns), &monthlyReturnsVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for monthlyReturnsBuf: " << err << endl;
			return 1;
		}

		winsBuf = cl::Buffer(context,
												 CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY |
														 CL_MEM_COPY_HOST_PTR,
												 numCombos * sizeof(wins), &winsVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for winsBuf: " << err << endl;
			return 1;
		}

		lossesBuf = cl::Buffer(context,
													 CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY |
															 CL_MEM_COPY_HOST_PTR,
													 numCombos * sizeof(losses), &lossesVec[0], &err);
		if (err != CL_SUCCESS) {
			cout << "Error for lossesBuf: " << err << endl;
			return 1;
		}
		totalSize += drawdownsVecSize + drawdownLengthsVecSize +
								 lossStreaksVecSize + tradeDurationsVecSize +
								 monthlyReturnsVecSize + winsVecSize + lossesVecSize;
		cout << "Total size: " << totalSize << endl;
		cout << "Total size: " << (double)totalSize / (double)(1024 * 1024 * 1024)
				 << " GiB" << endl;
	} else {
		numTradesInIntervalVec = vector<cl_int>(numCombos, 0);
		size_t numTradesInIntervalVecSize =
				numTradesInIntervalVec.size() * sizeof(cl_int);
		cout << "Size of number of trades in interval: "
				 << numTradesInIntervalVecSize << endl;
		if (snapshotting && snapshotFile.good()) {
			if (!snapshotFile.read(
							reinterpret_cast<char *>(numTradesInIntervalVec.data()),
							numCombos * sizeof(cl_int))) {
				cout << "Failed to read number of trades in interval" << endl;
				return 1;
			}
		}
		numTradesInIntervalBuf = cl::Buffer(
				context,
				CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				numCombos * sizeof(cl_int), &numTradesInIntervalVec[0], &err);
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

	int argNum = -1;
	err = kernel.setArg(++argNum, inputSize);
	if (err != CL_SUCCESS) {
		cout << "Error for kernel setArg " << argNum << ": " << err << endl;
	}
	err = kernel.setArg(++argNum, inputTrades);
	if (err != CL_SUCCESS) {
		cout << "Error for kernel setArg " << argNum << ": " << err << endl;
	}
	err = kernel.setArg(++argNum, inputCombos);
	if (err != CL_SUCCESS) {
		cout << "Error for kernel setArg " << argNum << ": " << err << endl;
	}
	err = kernel.setArg(++argNum, entries);
	if (err != CL_SUCCESS) {
		cout << "Error for kernel setArg " << argNum << ": " << err << endl;
	}
	err = kernel.setArg(++argNum, tradeRecords);
	if (err != CL_SUCCESS) {
		cout << "Error for kernel setArg " << argNum << ": " << err << endl;
	}

	switch (mode) {
	case Mode::Volume:
	case Mode::VolumeDelta:
	case Mode::VolumeExit:
	case Mode::AllVolume:
		err = kernel.setArg(++argNum, indicatorBuffer);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
		break;
	case Mode::Trend:
		err = kernel.setArg(++argNum, positionDatas);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
		break;
	case Mode::VolumeTrend:
		err = kernel.setArg(++argNum, indicatorBuffer);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
		err = kernel.setArg(++argNum, positionDatas);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
		break;
	}

	if (listTrades) {
		err = kernel.setArg(++argNum, drawdownsBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
		err = kernel.setArg(++argNum, drawdownLengthsBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
		err = kernel.setArg(++argNum, lossStreaksBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
		err = kernel.setArg(++argNum, tradeDurationsBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
		err = kernel.setArg(++argNum, monthlyReturnsBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
		err = kernel.setArg(++argNum, winsBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
		err = kernel.setArg(++argNum, lossesBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
	} else {
		err = kernel.setArg(++argNum, numTradesInIntervalBuf);
		if (err != CL_SUCCESS) {
			cout << "Error for kernel setArg " << argNum << ": " << err << endl;
		}
	}

	vector<trade> trades;
	vector<tradeWithoutDate> tradesWithoutDates;
	vector<indicators> indicators;
	vector<timeWindow> tws(NUM_WINDOWS, {0, 0});

	bool justProcessed = false;

	vector<perfMetrics> allPerfMetrics;
	if (listTrades) {
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
						if (mode == Mode::VolumeTrend || mode == Mode::Trend) {
							if (positionDatasVec[0].maxPrice == 0.0 &&
									positionDatasVec[0].minPrice == 0.0) {
#ifdef DEBUG
								assert(!existingSnapshot);
#endif
								for (auto &pos : positionDatasVec) {
									pos.maxPrice = t.price;
									pos.minPrice = t.price;
								}
								err = queue.enqueueWriteBuffer(positionDatas, CL_FALSE, 0,
																							 numCombos * sizeof(positionData),
																							 &positionDatasVec[0]);
								if (err != CL_SUCCESS) {
									cout << "Error for enqueueWriteBuffer positionDatas: " << err
											 << endl;
									return -1;
								}
							} else {
#ifdef DEBUG
								assert(existingSnapshot);
#endif
							}
						}
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
						processTradesWithOnlineAlgs(
								queue, kernel, tradesWithoutDates, indicators, tws, numCombos,
								inputTrades, inputSize, entries, tradeRecords, indicatorBuffer,
								positionDatas, drawdownsBuf, drawdownLengthsBuf, lossStreaksBuf,
								tradeDurationsBuf, monthlyReturnsBuf, winsBuf, lossesBuf,
								entriesVec, tradeRecordsVec, positionDatasVec, drawdownsVec,
								drawdownLengthsVec, lossStreaksVec, tradeDurationsVec,
								monthlyReturnsVec, winsVec, lossesVec, snapshotting, mode);
					else
						maxTradesPerInterval =
								max(maxTradesPerInterval,
										processTrades(queue, kernel, tradesWithoutDates, indicators,
																	tws, numCombos, inputTrades, inputSize,
																	entries, tradeRecords, indicatorBuffer,
																	positionDatas, numTradesInIntervalBuf,
																	entriesVec, tradeRecordsVec, positionDatasVec,
																	numTradesInIntervalVec, snapshotting, mode));
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
				processTradesWithOnlineAlgs(
						queue, kernel, tradesWithoutDates, indicators, tws, numCombos,
						inputTrades, inputSize, entries, tradeRecords, indicatorBuffer,
						positionDatas, drawdownsBuf, drawdownLengthsBuf, lossStreaksBuf,
						tradeDurationsBuf, monthlyReturnsBuf, winsBuf, lossesBuf,
						entriesVec, tradeRecordsVec, positionDatasVec, drawdownsVec,
						drawdownLengthsVec, lossStreaksVec, tradeDurationsVec,
						monthlyReturnsVec, winsVec, lossesVec, snapshotting, mode);
			else
				maxTradesPerInterval =
						max(maxTradesPerInterval,
								processTrades(queue, kernel, tradesWithoutDates, indicators,
															tws, numCombos, inputTrades, inputSize, entries,
															tradeRecords, indicatorBuffer, positionDatas,
															numTradesInIntervalBuf, entriesVec,
															tradeRecordsVec, positionDatasVec,
															numTradesInIntervalVec, snapshotting, mode));
		} else
			cout << "Run was already finished in the snapshot" << endl;
	}

	err = queue.enqueueReadBuffer(tradeRecords, CL_FALSE, 0,
																numCombos * sizeof(tradeRecord),
																&tradeRecordsVec[0]);
	if (err != CL_SUCCESS) {
		cout << "Error for enqueueReadBuffer tradeRecords: " << err << endl;
		return 1;
	}

	if (listTrades) {
		err = queue.enqueueReadBuffer(drawdownsBuf, CL_FALSE, 0,
																	numCombos * sizeof(drawdowns),
																	&drawdownsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer drawdownsBuf: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(drawdownLengthsBuf, CL_FALSE, 0,
																	numCombos * sizeof(drawdownLengths),
																	&drawdownLengthsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer drawdownLengthsBuf: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(lossStreaksBuf, CL_FALSE, 0,
																	numCombos * sizeof(lossStreaks),
																	&lossStreaksVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer lossStreaksBuf: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(tradeDurationsBuf, CL_FALSE, 0,
																	numCombos * sizeof(tradeDurations),
																	&tradeDurationsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer tradeDurationsBuf: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(monthlyReturnsBuf, CL_FALSE, 0,
																	numCombos * sizeof(monthlyReturns),
																	&monthlyReturnsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer monthlyReturnsBuf: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(winsBuf, CL_FALSE, 0,
																	numCombos * sizeof(wins), &winsVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer winsBuf: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(lossesBuf, CL_FALSE, 0,
																	numCombos * sizeof(losses), &lossesVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueReadBuffer lossesBuf: " << err << endl;
			return 1;
		}
	}

	err = queue.finish();
	if (err != CL_SUCCESS) {
		cout << "Error for finish: " << err << endl;
		return 1;
	}

	for (size_t i = 0; i < tradeRecordsVec.size(); i++) {
		tradeRecordsVec[i].capital =
				capitalToAnnualizedReturn(tradeRecordsVec[i].capital / STARTING_CAPITAL,
																	firstTimestamp, lastTimestamp);
	}

	auto afterKernelTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(afterKernelTime - setupTime);
	cout << endl;
	cout << "Time taken to run kernel: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	string resultsFilename = "results" + modeString;
	if (isBTC)
		resultsFilename += "BTC";
	else
		resultsFilename += "ETH";

	if (futures)
		resultsFilename += "Futures";
	if (listTrades) {
		resultsFilename += "Detailed";
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
										lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
										listTrades);
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
										lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
										listTrades);
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
										lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
										listTrades);
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
											lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
											listTrades);
				outFile << endl;

				maxElementIdx = max_element(drawdownsVec.begin(), drawdownsVec.end(),
																		[](drawdowns d1, drawdowns d2) {
																			return d1.mean < d2.mean;
																		}) -
												drawdownsVec.begin();
				outFile << "Smallest average drawdown:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
											allPerfMetrics, drawdownsVec, drawdownLengthsVec,
											lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
											listTrades);
				outFile << endl;

				maxElementIdx = max_element(drawdownsVec.begin(), drawdownsVec.end(),
																		[](drawdowns d1, drawdowns d2) {
																			return d1.max < d2.max;
																		}) -
												drawdownsVec.begin();
				outFile << "Smallest max drawdown:" << endl;
				outputMetrics(outFile, maxElementIdx, comboVec, tradeRecordsVec,
											allPerfMetrics, drawdownsVec, drawdownLengthsVec,
											lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
											listTrades);
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
											lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
											listTrades);
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
											lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
											listTrades);
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
											lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
											listTrades);
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
											lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
											listTrades);
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
											lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
											listTrades);
				outFile << endl;
			}

			for (size_t i = 0; i < numCombos; i++) {
				outputMetrics(outFile, i, comboVec, tradeRecordsVec, allPerfMetrics,
											drawdownsVec, drawdownLengthsVec, lossStreaksVec,
											tradeDurationsVec, winsVec, lossesVec, listTrades);
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
								tradeDurationsVec, winsVec, lossesVec, listTrades);
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
								tradeDurationsVec, winsVec, lossesVec, listTrades);
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
								tradeDurationsVec, winsVec, lossesVec, listTrades);
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
									lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
									listTrades);
		cout << endl;

		maxElementIdx = max_element(drawdownsVec.begin(), drawdownsVec.end(),
																[](drawdowns d1, drawdowns d2) {
																	return d1.mean < d2.mean;
																}) -
										drawdownsVec.begin();
		cout << "Smallest average drawdown:" << endl;
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
									listTrades);
		cout << endl;

		maxElementIdx = max_element(drawdownsVec.begin(), drawdownsVec.end(),
																[](drawdowns d1, drawdowns d2) {
																	return d1.max < d2.max;
																}) -
										drawdownsVec.begin();
		cout << "Smallest max drawdown:" << endl;
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
									listTrades);
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
									lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
									listTrades);
		cout << endl;

		maxElementIdx = min_element(lossStreaksVec.begin(), lossStreaksVec.end(),
																[](lossStreaks l1, lossStreaks l2) {
																	return l1.mean < l2.mean;
																}) -
										lossStreaksVec.begin();
		cout << "Shortest average loss streak:" << endl;
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
									listTrades);
		cout << endl;

		maxElementIdx = min_element(lossStreaksVec.begin(), lossStreaksVec.end(),
																[](lossStreaks l1, lossStreaks l2) {
																	return l1.max < l2.max;
																}) -
										lossStreaksVec.begin();
		cout << "Shortest max loss streak:" << endl;
		outputMetrics(cout, maxElementIdx, comboVec, tradeRecordsVec,
									allPerfMetrics, drawdownsVec, drawdownLengthsVec,
									lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
									listTrades);
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
									lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
									listTrades);
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
									lossStreaksVec, tradeDurationsVec, winsVec, lossesVec,
									listTrades);
		cout << endl;
	}

	auto endTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(endTime - startTime);
	cout << "Total time taken: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	return 0;
}
