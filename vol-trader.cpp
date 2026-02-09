#include <CL/opencl.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <ranges>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

using std::boolalpha;
using std::fixed;
using std::generate;
using std::get;
using std::get_time;
using std::getline;
using std::ifstream;
using std::istream_iterator;
using std::istringstream;
using std::max;
using std::max_element;
using std::min;
using std::mktime;
using std::nullopt;
using std::ofstream;
using std::optional;
using std::setprecision;
using std::stod;
using std::stoi;
using std::string;
using std::tm;
using std::to_string;
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

#define NUM_WINDOWS 4

#define MAX_TOTAL_TRADES 90

#define LOSS_BIT 0
#define WIN_BIT 1
#define LONG_POS_BIT 2
#define SHORT_POS_BIT 3

#define INCREMENT 1000000

#define TRADE_CHUNK 50000000

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
	cl_int tradeId;
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
	cl_int entryIndex;
	cl_int exitIndex;
	cl_int longShortWinLoss;
	entryData e;
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
	cl_int tradeId;
};

struct __attribute__((packed)) twMetadata {
	cl_int twTranslation;
	cl_int twStart;
};

tradeWithoutDate convertTrade(const trade &orig) {
	tradeWithoutDate newTrade;
	newTrade.tradeId = orig.tradeId;
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
		maxTradesPerInterval =
				max(maxTradesPerInterval, *max_element(numTradesInInterval.begin(),
																							 numTradesInInterval.end()));

		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return -1;
		}

		int minElementIdx =
				std::min_element(positionDatasVec.begin(), positionDatasVec.end(),
												 [](positionData p1, positionData p2) {
													 return p1.tradeId < p2.tradeId;
												 }) -
				positionDatasVec.begin();
		int minTradeId = positionDatasVec[minElementIdx].tradeId;
		tw.twTranslation = minTradeId;
		globalIdx += minTradeId;

		if (currIdx + currSize >= tradesWithoutDates.size()) {
			break;
		}

		tw.twStart = currSize - minTradeId;
		currIdx += minTradeId;
	}

	tradesWithoutDates.erase(tradesWithoutDates.begin(),
													 tradesWithoutDates.begin() + currIdx);
	return maxTradesPerInterval;
}

/*
void processTradesWithListing(cl::CommandQueue &queue,
															vector<tradeWithoutDate> &tradesWithoutDates,
															size_t currIdx, vector<combo> &comboVect,
															cl::Buffer inputTrades, cl::Buffer inputSize,
															cl::Buffer twBetweenRunData, cl::Kernel kernel,
															cl::Buffer positionDatas,
															cl::Buffer entriesAndExitsBuf) {
	cl_int err;

	twMetadata tw = {0, 0};
	vector<positionData> positionDatasVec(comboVect.size(), {0, 0.0, 0.0, 0});

	while (true) {
		size_t currSize =
				min((size_t)INCREMENT, tradesWithoutDates.size() - currIdx);
		err = queue.enqueueWriteBuffer(inputTrades, CL_FALSE, 0,
																	 currSize * sizeof(tradeWithoutDate),
																	 &tradesWithoutDates[currIdx]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputTrades: " << err << endl;
			return 1;
		}
		err = queue.enqueueWriteBuffer(inputSize, CL_FALSE, 0, sizeof(int),
																	 &currSize);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputSize: " << err << endl;
			return 1;
		}
		err = queue.enqueueWriteBuffer(twBetweenRunData, CL_FALSE, 0,
																	 sizeof(twMetadata), &tw);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer twBetweenRunData: " << err << endl;
			return 1;
		}
		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << endl;
		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange,
																		 cl::NDRange(comboVect.size()));
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(positionDatas, CL_FALSE, 0,
																	comboVect.size() * sizeof(positionData),
																	&positionDatasVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for reading positionDatas: " << err << endl;
			return 1;
		}

		err = queue.enqueueReadBuffer(entriesAndExitsBuf, CL_TRUE, 0,
																	comboVect.size() * sizeof(entryAndExit) *
																			MAX_TOTAL_TRADES,
																	&entriesAndExits[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for reading entriesAndExits: " << err << endl;
			return 1;
		}
		// handle trade info
		for (size_t i = 0; i < comboVect.size(); i++) {
			for (size_t j = 0; j < MAX_TOTAL_TRADES; j++) {
				entryAndExit adjustedData = entriesAndExits[i * MAX_TOTAL_TRADES + j];
				if (adjustedData.entryIndex == 0) {
					// keep optional value in case there are no trades in this interval?
					actualEntryIndex = nullopt;
					break;
				}
				if (adjustedData.exitIndex != 0) {
					if (j == 0 && actualEntryIndex.has_value()) {
						adjustedData.entryIndex = *actualEntryIndex;
					} else {
						adjustedData.entryIndex += tradeOffset;
					}
					adjustedData.exitIndex += tradeOffset;
					allTrades[i].emplace_back(adjustedData);
				} else {
					actualEntryIndex = adjustedData.entryIndex + tradeOffset;
					break;
				}
			}
		}
		entriesAndExits = vector<entryAndExit>(comboVect.size() * MAX_TOTAL_TRADES,
																					 entryAndExit{});
		err = queue.enqueueWriteBuffer(entriesAndExitsBuf, CL_FALSE, 0,
																	 comboVect.size() * sizeof(entryAndExit) *
																			 MAX_TOTAL_TRADES,
																	 &entriesAndExits[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for writing entriesAndExits: " << err << endl;
			return 1;
		}

		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return 1;
		}
		if (currIdx + currSize >= tradesWithoutDates.size())
			break;
		int minElementIdx =
				std::min_element(positionDatasVec.begin(), positionDatasVec.end(),
												 [](positionData p1, positionData p2) {
													 return p1.tradeId < p2.tradeId;
												 }) -
				positionDatasVec.begin();
		int minTradeId = positionDatasVec[minElementIdx].tradeId;
		tw.twTranslation = minTradeId;
		tw.twStart = currSize - minTradeId;
		currIdx += minTradeId;
		tradeOffset += minTradeId;
	}
}
*/

int main(int argc, char *argv[]) {
	bool writeResults = false, listTrades = false;

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

#if defined(__WIN64)
	_putenv("TZ=/usr/share/zoneinfo/UTC");
#elif defined(__linux)
	putenv("TZ=/usr/share/zoneinfo/UTC");
#endif

	auto startTime = high_resolution_clock::now();

	vector<long long> windows(NUM_WINDOWS);
	long long n = 0;
	generate(windows.begin(), windows.end(),
					 [n]() mutable { return n += FIFTEEN_MINUTES_MICROSECONDS; });

	vector<double> stopLosses(6);
	double x = 0;
	generate(stopLosses.begin(), stopLosses.end(),
					 [x]() mutable { return x += 0.5; });

	vector<double> targets(11);
	x = 0.5;
	generate(targets.begin(), targets.end(), [x]() mutable { return x += 0.5; });

	vector<vector<double>> buyVolPercentiles;
	vector<vector<double>> sellVolPercentiles;

	ifstream myFile;
	myFile.open("buyPercentilesDollarsBTC");
	if (myFile.is_open()) {
		string line;
		while (getline(myFile, line)) {
			vector<string> splits = split(line);
			vector<double> rowPercentiles;
			for (string s : splits) {
				rowPercentiles.emplace_back(stod(s));
			}
			buyVolPercentiles.emplace_back(rowPercentiles);
		}
		myFile.close();
	}

	myFile.open("sellPercentilesDollarsBTC");
	if (myFile.is_open()) {
		string line;
		while (getline(myFile, line)) {
			vector<string> splits = split(line);
			vector<double> rowPercentiles;
			for (string s : splits) {
				rowPercentiles.emplace_back(stod(s));
			}
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
			comboVect.emplace_back(c);
		}
	}

	initializeDevice();

	// vector<positionData> positionDatasVec(comboVect.size(),
	// {tradesWithoutDates[0].timestamp, 0.0, 0.0, 0});

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

	vector<vector<entryAndExit>> allTrades(comboVect.size());

	auto setupTime = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(setupTime - startTime);
	cout << "Time taken for setup: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

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
				t.tradeId = stoi(splits[3]);
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
					processTrades(queue, volKernel, tradesWithoutDates, comboVect,
												inputTrades, inputSize, twBetweenRunData, positionDatas,
												numTradesInIntervalBuf);
					justProcessed = true;
				}
			}
			myFile.close();
		}
	}

	if (!justProcessed) {
		processTrades(queue, volKernel, tradesWithoutDates, comboVect, inputTrades,
									inputSize, twBetweenRunData, positionDatas,
									numTradesInIntervalBuf);
	}

	/*
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

	cl::Kernel volKernel;
	if (listTrades) {
		volKernel = cl::Kernel(program, "volTraderWithTrades", &err);
	} else {
		volKernel = cl::Kernel(program, "volTrader", &err);
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

	cl::CommandQueue queue(context, device, 0, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for queue: " << err << endl;
		return 1;
	}
	*/

	/*
	cout << tradesWithoutDates.size() << endl;
	cout << tradesWithoutDates.size() * sizeof(tradeWithoutDate) << endl;
	cout << comboVect.size() * sizeof(combo) << endl;
	cout << CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE << endl;
	*/

	/*
	auto beforeKernelTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(beforeKernelTime - beforeSetupTime);
	cout << "Time taken to set up kernel: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	size_t currIdx = 0;
	twMetadata tw = {0, 0};

	int maxTradesPerInterval = 0;

	vector<vector<entryAndExit>> allTrades(comboVect.size());

	size_t tradeOffset = 0;
	optional<size_t> actualEntryIndex;

	while (true) {
		size_t currSize =
				min((size_t)INCREMENT, tradesWithoutDates.size() - currIdx);
		err = queue.enqueueWriteBuffer(inputTrades, CL_FALSE, 0,
																	 currSize * sizeof(tradeWithoutDate),
																	 &tradesWithoutDates[currIdx]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputTrades: " << err << endl;
			return 1;
		}
		err = queue.enqueueWriteBuffer(inputSize, CL_FALSE, 0, sizeof(int),
																	 &currSize);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputSize: " << err << endl;
			return 1;
		}
		err = queue.enqueueWriteBuffer(twBetweenRunData, CL_FALSE, 0,
																	 sizeof(twMetadata), &tw);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer twBetweenRunData: " << err << endl;
			return 1;
		}
		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1
				 << endl;
		err = queue.enqueueNDRangeKernel(volKernel, cl::NullRange,
																		 cl::NDRange(comboVect.size()));
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(positionDatas, CL_FALSE, 0,
																	comboVect.size() * sizeof(positionData),
																	&positionDatasVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for reading positionDatas: " << err << endl;
			return 1;
		}
		if (listTrades) {
			err = queue.enqueueReadBuffer(entriesAndExitsBuf, CL_TRUE, 0,
																		comboVect.size() * sizeof(entryAndExit) *
																				MAX_TOTAL_TRADES,
																		&entriesAndExits[0]);
			if (err != CL_SUCCESS) {
				cout << "Error for reading entriesAndExits: " << err << endl;
				return 1;
			}
			// handle trade info
			for (size_t i = 0; i < comboVect.size(); i++) {
				for (size_t j = 0; j < MAX_TOTAL_TRADES; j++) {
					entryAndExit adjustedData = entriesAndExits[i * MAX_TOTAL_TRADES + j];
					if (adjustedData.entryIndex == 0) {
						// keep optional value in case there are no trades in this interval?
						actualEntryIndex = nullopt;
						break;
					}
					if (adjustedData.exitIndex != 0) {
						if (j == 0 && actualEntryIndex.has_value()) {
							adjustedData.entryIndex = *actualEntryIndex;
						} else {
							adjustedData.entryIndex += tradeOffset;
						}
						adjustedData.exitIndex += tradeOffset;
						allTrades[i].emplace_back(adjustedData);
					} else {
						actualEntryIndex = adjustedData.entryIndex + tradeOffset;
						break;
					}
				}
			}
			entriesAndExits = vector<entryAndExit>(
					comboVect.size() * MAX_TOTAL_TRADES, entryAndExit{});
			err = queue.enqueueWriteBuffer(entriesAndExitsBuf, CL_FALSE, 0,
																		 comboVect.size() * sizeof(entryAndExit) *
																				 MAX_TOTAL_TRADES,
																		 &entriesAndExits[0]);
			if (err != CL_SUCCESS) {
				cout << "Error for writing entriesAndExits: " << err << endl;
				return 1;
			}
		} else {
			err = queue.enqueueReadBuffer(numTradesInIntervalBuf, CL_FALSE, 0,
																		comboVect.size() * sizeof(cl_int),
																		&numTradesInInterval[0]);
			if (err != CL_SUCCESS) {
				cout << "Error for reading numTradesInInterval: " << err << endl;
				return 1;
			}
			maxTradesPerInterval =
					max(maxTradesPerInterval, *max_element(numTradesInInterval.begin(),
																								 numTradesInInterval.end()));
		}
		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return 1;
		}
		if (currIdx + currSize >= tradesWithoutDates.size())
			break;
		int minElementIdx =
				std::min_element(positionDatasVec.begin(), positionDatasVec.end(),
												 [](positionData p1, positionData p2) {
													 return p1.tradeId < p2.tradeId;
												 }) -
				positionDatasVec.begin();
		int minTradeId = positionDatasVec[minElementIdx].tradeId;
		tw.twTranslation = minTradeId;
		tw.twStart = currSize - minTradeId;
		currIdx += minTradeId;
		tradeOffset += minTradeId;
	}
	*/

	/*
	err = queue.enqueueReadBuffer(entries, CL_FALSE, 0, comboVect.size() *
	sizeof(entry), &entriesVec[0]); if (err != CL_SUCCESS) { cout << "Error for
	reading entries: " << err << endl; return 1;
	}
	err = queue.enqueueReadBuffer(positionDatas, CL_FALSE, 0, comboVect.size() *
	sizeof(positionData), &positionDatasVec[0]); if (err != CL_SUCCESS) { cout <<
	"Error for reading positionDatas: " << err << endl; return 1;
	}
	*/
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

	auto afterKernelTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(afterKernelTime - setupTime);
	cout << "Time taken to run kernel: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	if (writeResults) {
		ofstream outFile;
		outFile.open("resultsVol");
		if (outFile.is_open()) {
			for (size_t i = 0; i < comboVect.size(); i++) {
				outFile << "Stop loss: " << to_string(comboVect[i].stopLoss) << endl;
				outFile << "Target: " << to_string(comboVect[i].target) << endl;
				outFile << "Window: "
								<< to_string(comboVect[i].window / ONE_MINUTE_MICROSECONDS)
								<< " minutes" << endl;
				outFile << "Buy volume threshold: "
								<< to_string(comboVect[i].buyVolPercentile) << endl;
				outFile << "Sell volume threshold: "
								<< to_string(comboVect[i].sellVolPercentile) << endl;
				outFile << "Total trades: "
								<< to_string(tradeRecordsVec[i].shorts +
														 tradeRecordsVec[i].longs)
								<< endl;
				outFile << "Wins: "
								<< to_string(tradeRecordsVec[i].shortWins +
														 tradeRecordsVec[i].longWins)
								<< endl;
				outFile << "Losses: "
								<< to_string(tradeRecordsVec[i].shortLosses +
														 tradeRecordsVec[i].longLosses)
								<< endl;
				outFile << "Longs: " << to_string(tradeRecordsVec[i].longs) << endl;
				outFile << "Long wins: " << to_string(tradeRecordsVec[i].longWins)
								<< endl;
				outFile << "Long losses: " << to_string(tradeRecordsVec[i].longLosses)
								<< endl;
				outFile << "Shorts: " << to_string(tradeRecordsVec[i].shorts) << endl;
				outFile << "Short wins: " << to_string(tradeRecordsVec[i].shortWins)
								<< endl;
				outFile << "Short losses: " << to_string(tradeRecordsVec[i].shortLosses)
								<< endl;
				outFile << "Final capital: " << to_string(tradeRecordsVec[i].capital)
								<< endl;
				if (listTrades) {
					for (int j = 0; j < MAX_TOTAL_TRADES; j++) {
						entryAndExit e = allTrades[i][j];
						if (e.entryIndex == 0)
							break;
						else {
							bool longEntry;
							if (e.longShortWinLoss & (1 << LONG_POS_BIT)) {
								outFile << "LONG ";
								longEntry = true;
							} else if (e.longShortWinLoss & (1 << SHORT_POS_BIT)) {
								outFile << "SHORT ";
								longEntry = false;
							}
							outFile << to_string(trades[e.entryIndex].price) << " "
											<< trades[e.entryIndex].date << " "
											<< to_string(trades[e.entryIndex].tradeId) << endl;
							outFile << "Buy volume: " << to_string(e.e.buyVol) << endl;
							outFile << "Sell volume: " << to_string(e.e.sellVol) << endl;
							if ((longEntry &&
									 trades[e.exitIndex].price >= trades[e.entryIndex].price) ||
									(!longEntry &&
									 trades[e.exitIndex].price <= trades[e.entryIndex].price)) {
								outFile << "Profit: " << to_string(trades[e.exitIndex].price)
												<< " " << trades[e.exitIndex].date << " "
												<< to_string(trades[e.exitIndex].tradeId) << endl;
							} else {
								outFile << "Loss: " << to_string(trades[e.exitIndex].price)
												<< " " << trades[e.exitIndex].date << " "
												<< to_string(trades[e.exitIndex].tradeId) << endl;
								;
							}
						}
					}
				}
				outFile << endl;
			}
			int maxElementIdx =
					std::max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
													 [](tradeRecord t1, tradeRecord t2) {
														 return t1.capital < t2.capital;
													 }) -
					tradeRecordsVec.begin();
			outFile << fixed;
			outFile << "Max return:" << endl;
			outFile << "Final capital: " << tradeRecordsVec[maxElementIdx].capital
							<< endl;
			outFile << "Stop loss: " << comboVect[maxElementIdx].stopLoss << endl;
			outFile << "Target: " << comboVect[maxElementIdx].target << endl;
			outFile << "Window: "
							<< comboVect[maxElementIdx].window / ONE_MINUTE_MICROSECONDS
							<< " minutes" << endl;
			outFile << "Buy volume threshold: "
							<< comboVect[maxElementIdx].buyVolPercentile << endl;
			outFile << "Sell volume threshold: "
							<< comboVect[maxElementIdx].sellVolPercentile << endl;
			outFile << "Total trades: "
							<< tradeRecordsVec[maxElementIdx].shorts +
										 tradeRecordsVec[maxElementIdx].longs
							<< endl;
			outFile << "Wins: "
							<< tradeRecordsVec[maxElementIdx].shortWins +
										 tradeRecordsVec[maxElementIdx].longWins
							<< endl;
			outFile << "Losses: "
							<< tradeRecordsVec[maxElementIdx].shortLosses +
										 tradeRecordsVec[maxElementIdx].longLosses
							<< endl;
			outFile << "Longs: " << tradeRecordsVec[maxElementIdx].longs << endl;
			outFile << "Long wins: " << tradeRecordsVec[maxElementIdx].longWins
							<< endl;
			outFile << "Long losses: " << tradeRecordsVec[maxElementIdx].longLosses
							<< endl;
			outFile << "Shorts: " << tradeRecordsVec[maxElementIdx].shorts << endl;
			outFile << "Short wins: " << tradeRecordsVec[maxElementIdx].shortWins
							<< endl;
			outFile << "Short losses: " << tradeRecordsVec[maxElementIdx].shortLosses
							<< endl;
			outFile << endl;

			maxElementIdx =
					std::max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
													 [](tradeRecord t1, tradeRecord t2) {
														 return (double)(t1.shortWins + t1.longWins) /
																				(t1.shorts + t1.longs) <
																		(double)(t2.shortWins + t2.longWins) /
																				(t2.shorts + t2.longs);
													 }) -
					tradeRecordsVec.begin();
			outFile << "Best win rate:" << endl;
			outFile << "Final capital: " << tradeRecordsVec[maxElementIdx].capital
							<< endl;
			outFile << "Stop loss: " << comboVect[maxElementIdx].stopLoss << endl;
			outFile << "Target: " << comboVect[maxElementIdx].target << endl;
			outFile << "Window: "
							<< comboVect[maxElementIdx].window / ONE_MINUTE_MICROSECONDS
							<< " minutes" << endl;
			outFile << "Buy volume threshold: "
							<< comboVect[maxElementIdx].buyVolPercentile << endl;
			outFile << "Sell volume threshold: "
							<< comboVect[maxElementIdx].sellVolPercentile << endl;
			outFile << "Total trades: "
							<< tradeRecordsVec[maxElementIdx].shorts +
										 tradeRecordsVec[maxElementIdx].longs
							<< endl;
			outFile << "Wins: "
							<< tradeRecordsVec[maxElementIdx].shortWins +
										 tradeRecordsVec[maxElementIdx].longWins
							<< endl;
			outFile << "Losses: "
							<< tradeRecordsVec[maxElementIdx].shortLosses +
										 tradeRecordsVec[maxElementIdx].longLosses
							<< endl;
			outFile << "Longs: " << tradeRecordsVec[maxElementIdx].longs << endl;
			outFile << "Long wins: " << tradeRecordsVec[maxElementIdx].longWins
							<< endl;
			outFile << "Long losses: " << tradeRecordsVec[maxElementIdx].longLosses
							<< endl;
			outFile << "Shorts: " << tradeRecordsVec[maxElementIdx].shorts << endl;
			outFile << "Short wins: " << tradeRecordsVec[maxElementIdx].shortWins
							<< endl;
			outFile << "Short losses: " << tradeRecordsVec[maxElementIdx].shortLosses
							<< endl;
			outFile << endl;

			maxElementIdx =
					std::max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
													 [](tradeRecord t1, tradeRecord t2) {
														 return t1.shorts + t1.longs < t2.shorts + t2.longs;
													 }) -
					tradeRecordsVec.begin();
			outFile << "Most trades:" << endl;
			outFile << "Final capital: " << tradeRecordsVec[maxElementIdx].capital
							<< endl;
			outFile << "Stop loss: " << comboVect[maxElementIdx].stopLoss << endl;
			outFile << "Target: " << comboVect[maxElementIdx].target << endl;
			outFile << "Window: "
							<< comboVect[maxElementIdx].window / ONE_MINUTE_MICROSECONDS
							<< " minutes" << endl;
			outFile << "Buy volume threshold: "
							<< comboVect[maxElementIdx].buyVolPercentile << endl;
			outFile << "Sell volume threshold: "
							<< comboVect[maxElementIdx].sellVolPercentile << endl;
			outFile << "Total trades: "
							<< tradeRecordsVec[maxElementIdx].shorts +
										 tradeRecordsVec[maxElementIdx].longs
							<< endl;
			outFile << "Wins: "
							<< tradeRecordsVec[maxElementIdx].shortWins +
										 tradeRecordsVec[maxElementIdx].longWins
							<< endl;
			outFile << "Losses: "
							<< tradeRecordsVec[maxElementIdx].shortLosses +
										 tradeRecordsVec[maxElementIdx].longLosses
							<< endl;
			outFile << "Longs: " << tradeRecordsVec[maxElementIdx].longs << endl;
			outFile << "Long wins: " << tradeRecordsVec[maxElementIdx].longWins
							<< endl;
			outFile << "Long losses: " << tradeRecordsVec[maxElementIdx].longLosses
							<< endl;
			outFile << "Shorts: " << tradeRecordsVec[maxElementIdx].shorts << endl;
			outFile << "Short wins: " << tradeRecordsVec[maxElementIdx].shortWins
							<< endl;
			outFile << "Short losses: " << tradeRecordsVec[maxElementIdx].shortLosses
							<< endl;
			outFile.close();

			auto outputTime = high_resolution_clock::now();
			duration = duration_cast<microseconds>(outputTime - afterKernelTime);
			cout << "Time taken to write output: "
					 << (double)duration.count() / 1000000 << " seconds" << endl;
		}
	}

	int maxElementIdx =
			std::max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
											 [](tradeRecord t1, tradeRecord t2) {
												 return t1.capital < t2.capital;
											 }) -
			tradeRecordsVec.begin();
	cout << fixed;
	cout << "Max return:" << endl;
	cout << "Final capital: " << tradeRecordsVec[maxElementIdx].capital << endl;
	cout << "Stop loss: " << comboVect[maxElementIdx].stopLoss << endl;
	cout << "Target: " << comboVect[maxElementIdx].target << endl;
	cout << "Window: "
			 << comboVect[maxElementIdx].window / ONE_MINUTE_MICROSECONDS
			 << " minutes" << endl;
	cout << "Buy volume threshold: " << comboVect[maxElementIdx].buyVolPercentile
			 << endl;
	cout << "Sell volume threshold: "
			 << comboVect[maxElementIdx].sellVolPercentile << endl;
	cout << "Total trades: "
			 << tradeRecordsVec[maxElementIdx].shorts +
							tradeRecordsVec[maxElementIdx].longs
			 << endl;
	cout << "Wins: "
			 << tradeRecordsVec[maxElementIdx].shortWins +
							tradeRecordsVec[maxElementIdx].longWins
			 << endl;
	cout << "Losses: "
			 << tradeRecordsVec[maxElementIdx].shortLosses +
							tradeRecordsVec[maxElementIdx].longLosses
			 << endl;
	cout << "Longs: " << tradeRecordsVec[maxElementIdx].longs << endl;
	cout << "Long wins: " << tradeRecordsVec[maxElementIdx].longWins << endl;
	cout << "Long losses: " << tradeRecordsVec[maxElementIdx].longLosses << endl;
	cout << "Shorts: " << tradeRecordsVec[maxElementIdx].shorts << endl;
	cout << "Short wins: " << tradeRecordsVec[maxElementIdx].shortWins << endl;
	cout << "Short losses: " << tradeRecordsVec[maxElementIdx].shortLosses
			 << endl;
	cout << endl;

	maxElementIdx =
			std::max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
											 [](tradeRecord t1, tradeRecord t2) {
												 return (double)(t1.shortWins + t1.longWins) /
																		(t1.shorts + t1.longs) <
																(double)(t2.shortWins + t2.longWins) /
																		(t2.shorts + t2.longs);
											 }) -
			tradeRecordsVec.begin();
	cout << "Best win rate:" << endl;
	cout << "Final capital: " << tradeRecordsVec[maxElementIdx].capital << endl;
	cout << "Stop loss: " << comboVect[maxElementIdx].stopLoss << endl;
	cout << "Target: " << comboVect[maxElementIdx].target << endl;
	cout << "Window: "
			 << comboVect[maxElementIdx].window / ONE_MINUTE_MICROSECONDS
			 << " minutes" << endl;
	cout << "Buy volume threshold: " << comboVect[maxElementIdx].buyVolPercentile
			 << endl;
	cout << "Sell volume threshold: "
			 << comboVect[maxElementIdx].sellVolPercentile << endl;
	cout << "Total trades: "
			 << tradeRecordsVec[maxElementIdx].shorts +
							tradeRecordsVec[maxElementIdx].longs
			 << endl;
	cout << "Wins: "
			 << tradeRecordsVec[maxElementIdx].shortWins +
							tradeRecordsVec[maxElementIdx].longWins
			 << endl;
	cout << "Losses: "
			 << tradeRecordsVec[maxElementIdx].shortLosses +
							tradeRecordsVec[maxElementIdx].longLosses
			 << endl;
	cout << "Longs: " << tradeRecordsVec[maxElementIdx].longs << endl;
	cout << "Long wins: " << tradeRecordsVec[maxElementIdx].longWins << endl;
	cout << "Long losses: " << tradeRecordsVec[maxElementIdx].longLosses << endl;
	cout << "Shorts: " << tradeRecordsVec[maxElementIdx].shorts << endl;
	cout << "Short wins: " << tradeRecordsVec[maxElementIdx].shortWins << endl;
	cout << "Short losses: " << tradeRecordsVec[maxElementIdx].shortLosses
			 << endl;
	cout << endl;

	maxElementIdx =
			std::max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(),
											 [](tradeRecord t1, tradeRecord t2) {
												 return t1.shorts + t1.longs < t2.shorts + t2.longs;
											 }) -
			tradeRecordsVec.begin();
	cout << "Most trades:" << endl;
	cout << "Final capital: " << tradeRecordsVec[maxElementIdx].capital << endl;
	cout << "Stop loss: " << comboVect[maxElementIdx].stopLoss << endl;
	cout << "Target: " << comboVect[maxElementIdx].target << endl;
	cout << "Window: "
			 << comboVect[maxElementIdx].window / ONE_MINUTE_MICROSECONDS
			 << " minutes" << endl;
	cout << "Buy volume threshold: " << comboVect[maxElementIdx].buyVolPercentile
			 << endl;
	cout << "Sell volume threshold: "
			 << comboVect[maxElementIdx].sellVolPercentile << endl;
	cout << "Total trades: "
			 << tradeRecordsVec[maxElementIdx].shorts +
							tradeRecordsVec[maxElementIdx].longs
			 << endl;
	cout << "Wins: "
			 << tradeRecordsVec[maxElementIdx].shortWins +
							tradeRecordsVec[maxElementIdx].longWins
			 << endl;
	cout << "Losses: "
			 << tradeRecordsVec[maxElementIdx].shortLosses +
							tradeRecordsVec[maxElementIdx].longLosses
			 << endl;
	cout << "Longs: " << tradeRecordsVec[maxElementIdx].longs << endl;
	cout << "Long wins: " << tradeRecordsVec[maxElementIdx].longWins << endl;
	cout << "Long losses: " << tradeRecordsVec[maxElementIdx].longLosses << endl;
	cout << "Shorts: " << tradeRecordsVec[maxElementIdx].shorts << endl;
	cout << "Short wins: " << tradeRecordsVec[maxElementIdx].shortWins << endl;
	cout << "Short losses: " << tradeRecordsVec[maxElementIdx].shortLosses
			 << endl;
	cout << endl;

	cout << "Max trades per interval: " << maxTradesPerInterval << endl;

	auto endTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(endTime - startTime);
	cout << "Total time taken: " << (double)duration.count() / 1000000
			 << " seconds" << endl;

	return 0;
}
