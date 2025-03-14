#include <algorithm>
#include <CL/opencl.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <chrono>
#include <tuple>
#include <ranges>

using std::ifstream;
using std::istringstream;
using std::ofstream;
using std::string;
using std::getline;
using std::vector;
using std::istream_iterator;
using std::chrono::system_clock;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;
using std::stoi;
using std::stod;
using std::boolalpha;
using std::tm;
using std::get_time;
using std::mktime;
using std::generate;
using std::tuple;
using std::get;
using std::ranges::views::cartesian_product;
using std::to_string;
using std::max;
using std::min;
using std::max_element;
using std::setprecision;
using std::fixed;

using std::cout;
using std::cerr;
using std::endl;

#define ONE_HOUR_MICROSECONDS 3600000000
#define FIFTEEN_MINUTES_MICROSECONDS 900000000
#define ONE_MINUTE_MICROSECONDS 60000000

#define NUM_WINDOWS 16

#define MAX_TOTAL_TRADES 42

#define LOSS_BIT 0
#define WIN_BIT 1
#define LONG_POS_BIT 2
#define SHORT_POS_BIT 3

#define INCREMENT 1000000

cl::Device getDefaultDevice();                                    // Return a device found in this OpenCL platform.

void initializeDevice();                                          // Initialize device and compile kernel code.

cl::Program program;                // The program that will run on the device.
cl::Context context;                // The context which holds the device.
cl::Device device;                  // The device where the kernel will run.

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
#ifdef LIST_TRADES
	std::ifstream kernel_file("vol-delta-trader-with-trades.cl");
#else
	std::ifstream kernel_file("vol-delta-trader.cl");
#endif
	std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

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
		cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
				 << "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
		exit(1);
	}
}

vector<string> split(string const &input) {
	istringstream buffer(input);
	vector<string> ret((istream_iterator<string>(buffer)), istream_iterator<string>());
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

struct __attribute__ ((packed)) tradeWithoutDate {
	cl_long timestamp;
	cl_double price;
	cl_double qty;
	cl_int tradeId;
	cl_uchar isBuyerMaker;
};

struct __attribute__ ((packed)) combo {
	cl_long window;
	cl_double buyDeltaPercentile;
	cl_double sellDeltaPercentile;
	cl_double stopLoss;
	cl_double target;
};

struct __attribute__ ((packed)) entryAndExit {
	cl_int entryIndex;
	cl_int exitIndex;
	cl_int longShortWinLoss;
};

struct __attribute__ ((packed)) entry {
	cl_double price;
	cl_uchar isLong;
};

struct __attribute__ ((packed)) tradeRecord {
	cl_double capital;
	cl_int shorts;
	cl_int shortWins;
	cl_int shortLosses;
	cl_int longs;
	cl_int longWins;
	cl_int longLosses;
};

struct __attribute__ ((packed)) positionData {
	cl_long timestamp;
	cl_double buyVol;
	cl_double sellVol;
	cl_int tradeId;
};

struct __attribute__ ((packed)) twMetadata {
	cl_int twTranslation;
	cl_int twStart;
};

tradeWithoutDate convertTrade(const trade& orig) {
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

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cout << "No file name specified!" << endl;
		return 1;
	}

	auto startTime = high_resolution_clock::now();

	initializeDevice();

#if defined(__WIN64)
	_putenv("TZ=/usr/share/zoneinfo/UTC");
#elif defined(__linux)
	putenv("TZ=/usr/share/zoneinfo/UTC");
#endif

	ifstream myFile;
	vector<trade> trades;
	vector<tradeWithoutDate> tradesWithoutDates;

	vector<long long> windows(NUM_WINDOWS);
	long long n = 0;
	generate(windows.begin(), windows.end(), [n] () mutable { return n += FIFTEEN_MINUTES_MICROSECONDS; });

	vector<double> stopLosses(6);
	double x = 0;
	generate(stopLosses.begin(), stopLosses.end(), [x] () mutable { return x += 0.5; });

	vector<double> targets(11);
	x = 0.5;
	generate(targets.begin(), targets.end(), [x] () mutable { return x += 0.5; });

	for (int i = 1; i < argc; i++) {
		myFile.open(argv[i]);
		if (myFile.is_open()) {
			cout << "Reading " << argv[i] << endl;
			string line;
			while (getline(myFile, line)) {
				vector<string> splits = split(line);
				trade t;
				t.tradeId = stoi(splits[0]);
				t.price = stod(splits[3]);
				t.qty = stod(splits[4]);
				istringstream(splits[5]) >> boolalpha >> t.isBuyerMaker;
				string date = splits[1] + "T" + splits[2];
				t.date = date;
				tm localTimeTm;
				int micros;
				istringstream(date) >> get_time(&localTimeTm, "%Y-%m-%dT%H:%M:%S.") >> micros;
				micros /= 1000;
				localTimeTm.tm_isdst = 0;
				auto tpLocal = system_clock::from_time_t(mktime(&localTimeTm));
				t.timestamp = duration_cast<microseconds>(tpLocal.time_since_epoch()).count() + micros;
				// trades.emplace_back(t);
				tradesWithoutDates.emplace_back(convertTrade(t));
			}
			myFile.close();
		}
	}

	auto fileTime = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(fileTime - startTime);
	cout << "Time taken to read input: " << (double) duration.count() / 1000000 << " seconds" << endl;

	vector< vector<double> > volDeltaPercentiles;

	myFile.open("volDeltas");
	if (myFile.is_open()) {
		string line;
		while (getline(myFile, line)) {
			vector<string> splits = split(line);
			vector<double> rowPercentiles;
			for (string s : splits) {
				rowPercentiles.emplace_back(stod(s));
			}
			volDeltaPercentiles.emplace_back(rowPercentiles);
		}
		myFile.close();
	}

	auto beforeSetupTime = high_resolution_clock::now();

	cl_int err;
	cl::Buffer inputTrades(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, INCREMENT * sizeof(tradeWithoutDate), NULL, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputTrades: " << err << endl;
		return 1;
	}
	cl::Buffer inputSize(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputSize: " << err << endl;
		return 1;
	}
	cl::Buffer twBetweenRunData(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, sizeof(twMetadata), NULL, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for twBetweenRunData: " << err << endl;
		return 1;
	}

	vector< vector<long long> > indWindows(NUM_WINDOWS);
	for (int i = 0; i < NUM_WINDOWS; i++) {
		indWindows[i] = vector<long long>(1, windows[i]);
	}
	auto firstCombo = cartesian_product(indWindows[0], volDeltaPercentiles[0], volDeltaPercentiles[0], stopLosses, targets);
	vector< decltype(firstCombo) > combos;
	combos.emplace_back(firstCombo);

	for (int i = 1; i < NUM_WINDOWS; i++) {
		combos.emplace_back(cartesian_product(indWindows[i], volDeltaPercentiles[i], volDeltaPercentiles[i], stopLosses, targets));
	}

	vector<combo> comboVect;

	for (int i = 0; i < NUM_WINDOWS; i++) {
		for (unsigned int j = 0; j < combos[i].size(); j++) {
			combo c = {get<0>(combos[i][j]), get<1>(combos[i][j]), get<2>(combos[i][j]), get<3>(combos[i][j]), get<4>(combos[i][j])};
			// cout << c.window << " " << c.target << " " << c.stopLoss << " " << c.buyVolPercentile << " " << c.sellVolPercentile << endl;
			comboVect.emplace_back(c);
		}
	}

	vector<entry> entriesVec(comboVect.size(), {0.0, 0});
	vector<tradeRecord> tradeRecordsVec(comboVect.size(), {1.0, 0, 0, 0, 0, 0, 0});
	vector<positionData> positionDatasVec(comboVect.size(), {tradesWithoutDates[0].timestamp, 0.0, 0.0, 0});

	cl::Buffer inputCombos(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, comboVect.size() * sizeof(combo), &comboVect[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputCombos: " << err << endl;
		return 1;
	}
	cl::Buffer entries(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, comboVect.size() * sizeof(entry), &entriesVec[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for entries: " << err << endl;
		return 1;
	}
	cl::Buffer tradeRecords(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, comboVect.size() * sizeof(tradeRecord), &tradeRecordsVec[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for tradeRecords: " << err << endl;
		return 1;
	}
	cl::Buffer positionDatas(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, comboVect.size() * sizeof(positionData), &positionDatasVec[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for positionDatas: " << err << endl;
		return 1;
	}
#ifdef LIST_TRADES
	vector<entryAndExit> entriesAndExits(comboVect.size() * MAX_TOTAL_TRADES, entryAndExit{});
	cl::Buffer entriesAndExitsBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, comboVect.size() * sizeof(entryAndExit) * MAX_TOTAL_TRADES, &entriesAndExits[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for entriesAndExitsBuf: " << err << endl;
		return 1;
	}
#endif

	cl::Kernel volKernel(program, "volTrader", &err);
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
#ifdef LIST_TRADES
	err = volKernel.setArg(6, entriesAndExitsBuf);
	if (err != CL_SUCCESS) {
		cout << "Error for volKernel setArg 6: " << err << endl;
	}
#endif

	cl::CommandQueue queue(context, device, 0, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for queue: " << err << endl;
		return 1;
	}

	/*
	cout << tradesWithoutDates.size() << endl;
	cout << tradesWithoutDates.size() * sizeof(tradeWithoutDate) << endl;
	cout << comboVect.size() * sizeof(combo) << endl;
	cout << CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE << endl;
	*/

	auto beforeKernelTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(beforeKernelTime - beforeSetupTime);
	cout << "Time taken to set up kernel: " << (double) duration.count() / 1000000 << " seconds" << endl;

	size_t currIdx = 0;
	twMetadata tw = {0, 0};

	while (true) {
		size_t currSize = min((size_t) INCREMENT, tradesWithoutDates.size() - currIdx);
		err = queue.enqueueWriteBuffer(inputTrades, CL_FALSE, 0, currSize * sizeof(tradeWithoutDate), &tradesWithoutDates[currIdx]);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputTrades: " << err << endl;
			return 1;
		}
		err = queue.enqueueWriteBuffer(inputSize, CL_FALSE, 0, sizeof(int), &currSize);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer inputSize: " << err << endl;
			return 1;
		}
		err = queue.enqueueWriteBuffer(twBetweenRunData, CL_FALSE, 0, sizeof(twMetadata), &tw);
		if (err != CL_SUCCESS) {
			cout << "Error for enqueueWriteBuffer twBetweenRunData: " << err << endl;
			return 1;
		}
		cout << "Running trades " << currIdx << "-" << currIdx + currSize - 1 << endl;
		err = queue.enqueueNDRangeKernel(volKernel, cl::NullRange, cl::NDRange(comboVect.size()));
		if (err != CL_SUCCESS) {
			cout << "Error for volKernel: " << err << endl;
			return 1;
		}
		err = queue.enqueueReadBuffer(positionDatas, CL_FALSE, 0, comboVect.size() * sizeof(positionData), &positionDatasVec[0]);
		if (err != CL_SUCCESS) {
			cout << "Error for reading positionDatas: " << err << endl;
			return 1;
		}
		err = queue.finish();
		if (err != CL_SUCCESS) {
			cout << "Error for finish: " << err << endl;
			return 1;
		}
		if (currIdx + currSize >= tradesWithoutDates.size()) break;
		int minElementIdx = std::min_element(positionDatasVec.begin(), positionDatasVec.end(), [](positionData p1, positionData p2) { return p1.tradeId < p2.tradeId; }) - positionDatasVec.begin();
		int minTradeId= positionDatasVec[minElementIdx].tradeId;
		tw.twTranslation = minTradeId;
		tw.twStart = currSize - minTradeId;
		currIdx += minTradeId;
	}

	/*
	err = queue.enqueueReadBuffer(entries, CL_FALSE, 0, comboVect.size() * sizeof(entry), &entriesVec[0]);
	if (err != CL_SUCCESS) {
		cout << "Error for reading entries: " << err << endl;
		return 1;
	}
	err = queue.enqueueReadBuffer(positionDatas, CL_FALSE, 0, comboVect.size() * sizeof(positionData), &positionDatasVec[0]);
	if (err != CL_SUCCESS) {
		cout << "Error for reading positionDatas: " << err << endl;
		return 1;
	}
	*/
	err = queue.enqueueReadBuffer(tradeRecords, CL_FALSE, 0, comboVect.size() * sizeof(tradeRecord), &tradeRecordsVec[0]);
	if (err != CL_SUCCESS) {
		cout << "Error for reading tradeRecords: " << err << endl;
		return 1;
	}
#ifdef LIST_TRADES
	err = queue.enqueueReadBuffer(entriesAndExitsBuf, CL_TRUE, 0, comboVect.size() * sizeof(entryAndExit) * MAX_TOTAL_TRADES, &entriesAndExits[0]);
	if (err != CL_SUCCESS) {
		cout << "Error for reading entriesAndExits: " << err << endl;
		return 1;
	}
#endif
	err = queue.finish();
	if (err != CL_SUCCESS) {
		cout << "Error for finish: " << err << endl;
		return 1;
	}

	auto afterKernelTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(afterKernelTime - beforeKernelTime);
	cout << "Time taken to run kernel: " << (double) duration.count() / 1000000 << " seconds" << endl;

#ifdef WRITE_OUTPUT
	ofstream outFile;
	outFile.open("resultsVolDeltaTrader");
	if (outFile.is_open()) {
		for (size_t i = 0; i < comboVect.size(); i++) {
			outFile << "Stop loss: " << to_string(comboVect[i].stopLoss) << endl;
			outFile << "Target: " << to_string(comboVect[i].target) << endl;
			outFile << "Window: " << to_string(comboVect[i].window / ONE_MINUTE_MICROSECONDS) << " minutes" << endl;
			outFile << "Buy delta threshold: " << to_string(comboVect[i].buyDeltaPercentile) << endl;
			outFile << "Sell delta threshold: " << to_string(comboVect[i].sellDeltaPercentile) << endl;
			outFile << "Total trades: " << to_string(tradeRecordsVec[i].shorts + tradeRecordsVec[i].longs) << endl;
			outFile << "Wins: " << to_string(tradeRecordsVec[i].shortWins + tradeRecordsVec[i].longWins) << endl;
			outFile << "Losses: " << to_string(tradeRecordsVec[i].shortLosses + tradeRecordsVec[i].longLosses) << endl;
			outFile << "Longs: " << to_string(tradeRecordsVec[i].longs) << endl;
			outFile << "Long wins: " << to_string(tradeRecordsVec[i].longWins) << endl;
			outFile << "Long losses: " << to_string(tradeRecordsVec[i].longLosses) << endl;
			outFile << "Shorts: " << to_string(tradeRecordsVec[i].shorts) << endl;
			outFile << "Short wins: " << to_string(tradeRecordsVec[i].shortWins) << endl;
			outFile << "Short losses: " << to_string(tradeRecordsVec[i].shortLosses) << endl;
			outFile << "Final capital: " << to_string(tradeRecordsVec[i].capital) << endl;
#ifdef LIST_TRADES
			for (int j = 0; j < MAX_TOTAL_TRADES; j++) {
				entryAndExit e = entriesAndExits[i * MAX_TOTAL_TRADES + j];
				if (e.entryIndex == 0) break;
				else {
					if (e.longShortWinLoss & (1 << LONG_POS_BIT)) {
						outFile << "LONG ";
					} else if (e.longShortWinLoss & (1 << SHORT_POS_BIT)) {
						outFile << "SHORT ";
					}
					outFile << to_string(trades[e.entryIndex].price) << " " << trades[e.entryIndex].date << " " << to_string(trades[e.entryIndex].tradeId) << endl;
					if (e.longShortWinLoss & (1 << WIN_BIT)) {
						outFile << "Profit: " << to_string(trades[e.exitIndex].price) << " " << trades[e.exitIndex].date << " " << to_string(trades[e.exitIndex].tradeId) << endl;
					} else if (e.longShortWinLoss & (1 << LOSS_BIT)) {
						outFile << "Loss: " << to_string(trades[e.exitIndex].price) << " " << trades[e.exitIndex].date << " " << to_string(trades[e.exitIndex].tradeId) << endl;;
					}
				}
			}
#endif
			outFile << endl;
		}
		int maxElementIdx = std::max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(), [](tradeRecord t1, tradeRecord t2) { return t1.capital < t2.capital; }) - tradeRecordsVec.begin();
		outFile << fixed;
		outFile << "Max return:" << endl;
		outFile << "Final capital: " << tradeRecordsVec[maxElementIdx].capital << endl;
		outFile << "Stop loss: " << comboVect[maxElementIdx].stopLoss << endl;
		outFile << "Target: " << comboVect[maxElementIdx].target << endl;
		outFile << "Window: " << comboVect[maxElementIdx].window / ONE_MINUTE_MICROSECONDS << " minutes" << endl;
		outFile << "Buy delta threshold: " << comboVect[maxElementIdx].buyDeltaPercentile << endl;
		outFile << "Sell delta threshold: " << comboVect[maxElementIdx].sellDeltaPercentile << endl;
		outFile << "Total trades: " << tradeRecordsVec[maxElementIdx].shorts + tradeRecordsVec[maxElementIdx].longs << endl;
		outFile << "Wins: " << tradeRecordsVec[maxElementIdx].shortWins + tradeRecordsVec[maxElementIdx].longWins << endl;
		outFile << "Losses: " << tradeRecordsVec[maxElementIdx].shortLosses + tradeRecordsVec[maxElementIdx].longLosses << endl;
		outFile << "Longs: " << tradeRecordsVec[maxElementIdx].longs << endl;
		outFile << "Long wins: " << tradeRecordsVec[maxElementIdx].longWins << endl;
		outFile << "Long losses: " << tradeRecordsVec[maxElementIdx].longLosses << endl;
		outFile << "Shorts: " << tradeRecordsVec[maxElementIdx].shorts << endl;
		outFile << "Short wins: " << tradeRecordsVec[maxElementIdx].shortWins << endl;
		outFile << "Short losses: " << tradeRecordsVec[maxElementIdx].shortLosses << endl;
#ifdef LIST_TRADES
		for (int j = 0; j < MAX_TOTAL_TRADES; j++) {
			entryAndExit e = entriesAndExits[maxElementIdx * MAX_TOTAL_TRADES + j];
			if (e.entryIndex == 0) break;
			else {
				if (e.longShortWinLoss & (1 << LONG_POS_BIT)) {
					outFile << "LONG ";
				} else if (e.longShortWinLoss & (1 << SHORT_POS_BIT)) {
					outFile << "SHORT ";
				}
				outFile << to_string(trades[e.entryIndex].price) << " " << trades[e.entryIndex].date << " " << to_string(trades[e.entryIndex].tradeId) << endl;
				if (e.longShortWinLoss & (1 << WIN_BIT)) {
					outFile << "Profit: " << to_string(trades[e.exitIndex].price) << " " << trades[e.exitIndex].date << " " << to_string(trades[e.exitIndex].tradeId) << endl;
				} else if (e.longShortWinLoss & (1 << LOSS_BIT)) {
					outFile << "Loss: " << to_string(trades[e.exitIndex].price) << " " << trades[e.exitIndex].date << " " << to_string(trades[e.exitIndex].tradeId) << endl;;
				}
			}
		}
#endif
		outFile.close();

		auto outputTime = high_resolution_clock::now();
		duration = duration_cast<microseconds>(outputTime - afterKernelTime);
		cout << "Time taken to write output: " << (double) duration.count() / 1000000 << " seconds" << endl;
	}
#endif

	int maxElementIdx = std::max_element(tradeRecordsVec.begin(), tradeRecordsVec.end(), [](tradeRecord t1, tradeRecord t2) { return t1.capital < t2.capital; }) - tradeRecordsVec.begin();
	cout << fixed;
	cout << "Max return:" << endl;
	cout << "Final capital: " << tradeRecordsVec[maxElementIdx].capital << endl;
	cout << "Stop loss: " << comboVect[maxElementIdx].stopLoss << endl;
	cout << "Target: " << comboVect[maxElementIdx].target << endl;
	cout << "Window: " << comboVect[maxElementIdx].window / ONE_MINUTE_MICROSECONDS << " minutes" << endl;
	cout << "Buy delta threshold: " << comboVect[maxElementIdx].buyDeltaPercentile << endl;
	cout << "Sell delta threshold: " << comboVect[maxElementIdx].sellDeltaPercentile << endl;
	cout << "Total trades: " << tradeRecordsVec[maxElementIdx].shorts + tradeRecordsVec[maxElementIdx].longs << endl;
	cout << "Wins: " << tradeRecordsVec[maxElementIdx].shortWins + tradeRecordsVec[maxElementIdx].longWins << endl;
	cout << "Losses: " << tradeRecordsVec[maxElementIdx].shortLosses + tradeRecordsVec[maxElementIdx].longLosses << endl;
	cout << "Longs: " << tradeRecordsVec[maxElementIdx].longs << endl;
	cout << "Long wins: " << tradeRecordsVec[maxElementIdx].longWins << endl;
	cout << "Long losses: " << tradeRecordsVec[maxElementIdx].longLosses << endl;
	cout << "Shorts: " << tradeRecordsVec[maxElementIdx].shorts << endl;
	cout << "Short wins: " << tradeRecordsVec[maxElementIdx].shortWins << endl;
	cout << "Short losses: " << tradeRecordsVec[maxElementIdx].shortLosses << endl;

	auto endTime = high_resolution_clock::now();
	duration = duration_cast<microseconds>(endTime - startTime);
	cout << "Total time taken: " << (double) duration.count() / 1000000 << " seconds" << endl;

	return 0;
}
