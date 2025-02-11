#include <algorithm>
#include <CL/cl2.hpp>
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
#define LONG_BIT 2
#define SHORT_BIT 3

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
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	if (devices.empty()) {
		cerr << "No devices found!" << endl;
		exit(1);
	}

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
	std::ifstream kernel_file("vol-trend-trader-with-trades.cl");
#else
	std::ifstream kernel_file("vol-trend-trader.cl");
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
				 << "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
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
	cl_int isBuyerMaker;
};

struct __attribute__ ((packed)) combo {
	cl_long window;
	cl_double buyVolPercentile;
	cl_double sellVolPercentile;
	cl_double entryThreshold;
	cl_double stopLoss;
	cl_double target;
};

struct __attribute__ ((packed)) entryAndExit {
	cl_int entryIndex;
	cl_int exitIndex;
	cl_int longShortWinLoss;
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

	initializeDevice();
	_putenv("TZ=/usr/share/zoneinfo/UTC");
	ifstream myFile;
	myFile.open(argv[1]);
	vector<trade> trades;
	vector<tradeWithoutDate> tradesWithoutDates;

	vector<long long> windows(NUM_WINDOWS);
	long long n = 0;
	generate(windows.begin(), windows.end(), [n] () mutable { return n += FIFTEEN_MINUTES_MICROSECONDS; });

	vector<double> stopLosses(6);
	double x = 0;
	generate(stopLosses.begin(), stopLosses.end(), [x] () mutable { return x += 0.5; });

	vector<double> targets(15);
	x = 0.5;
	generate(targets.begin(), targets.end(), [x] () mutable { return x += 0.5; });

	vector<double> entryThresholds(10);
	x = 0.0;
	generate(entryThresholds.begin(), entryThresholds.end(), [x] () mutable { return x += 0.05; });

	if (myFile.is_open()) {
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
			t.timestamp = duration_cast<microseconds>(tpLocal.time_since_epoch()).count() + ONE_HOUR_MICROSECONDS + micros;
			// trades.emplace_back(t);
			tradesWithoutDates.emplace_back(convertTrade(t));
		}
		myFile.close();
	}
	cl_int err;
	cl::Buffer inputTrades(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, tradesWithoutDates.size() * sizeof(tradeWithoutDate), &tradesWithoutDates[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputTrades: " << err << endl;
		return 1;
	}

	vector<double> buyVols(NUM_WINDOWS, 0);
	vector<double> sellVols(NUM_WINDOWS, 0);

	vector< vector<double> > buyVolPercentiles;
	vector< vector<double> > sellVolPercentiles;

	myFile.open("buyPercentiles");
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

	myFile.open("sellPercentiles");
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

	vector< vector<long long> > indWindows(NUM_WINDOWS);
	for (int i = 0; i < NUM_WINDOWS; i++) {
		indWindows[i] = vector<long long>(1, windows[i]);
	}
	auto firstCombo = cartesian_product(indWindows[0], buyVolPercentiles[0], sellVolPercentiles[0], entryThresholds, stopLosses, targets);
	vector< decltype(firstCombo) > combos;
	combos.emplace_back(firstCombo);

	for (int i = 1; i < NUM_WINDOWS; i++) {
		combos.emplace_back(cartesian_product(indWindows[i], buyVolPercentiles[i], sellVolPercentiles[i], entryThresholds, stopLosses, targets));
	}

	vector<combo> comboVect;

	for (int i = 0; i < NUM_WINDOWS; i++) {
		for (unsigned int j = 0; j < combos[i].size(); j++) {
			combo c = {get<0>(combos[i][j]), get<1>(combos[i][j]), get<2>(combos[i][j]), get<3>(combos[i][j]), get<4>(combos[i][j]), get<5>(combos[i][j])};
			// cout << c.window << " " << c.target << " " << c.stopLoss << " " << c.buyVolPercentile << " " << c.sellVolPercentile << endl;
			comboVect.emplace_back(c);
		}
	}
	cl::Buffer inputCombos(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, comboVect.size() * sizeof(combo), &comboVect[0], &err);
	if (err != CL_SUCCESS) {
		cout << "Error for inputCombos: " << err << endl;
		return 1;
	}
	cl::Buffer capitals(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, comboVect.size() * sizeof(cl_double), nullptr, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for capitals: " << err << endl;
		return 1;
	}
	cl::Buffer totalTrades(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, comboVect.size() * sizeof(cl_int), nullptr, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for totalTrades: " << err << endl;
		return 1;
	}
	cl::Buffer wins(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, comboVect.size() * sizeof(cl_int), nullptr, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for wins: " << err << endl;
		return 1;
	}
	cl::Buffer losses(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, comboVect.size() * sizeof(cl_int), nullptr, &err);
	if (err != CL_SUCCESS) {
		cout << "Error for losses: " << err << endl;
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

	cl::Kernel volTrendKernel(program, "volTrendTrader", &err);
	if (err != CL_SUCCESS) {
		cout << "Error for creating volTrendKernel: " << err << endl;
		return 1;
	}
	int s = tradesWithoutDates.size();
	err = volTrendKernel.setArg(0, sizeof(int), &s);
	if (err != CL_SUCCESS) {
		cout << "Error for volTrendKernel setArg 0: " << err << endl;
	}
	err = volTrendKernel.setArg(1, inputTrades);
	if (err != CL_SUCCESS) {
		cout << "Error for volTrendKernel setArg 1: " << err << endl;
	}
	err = volTrendKernel.setArg(2, inputCombos);
	if (err != CL_SUCCESS) {
		cout << "Error for volTrendKernel setArg 2: " << err << endl;
	}
	err = volTrendKernel.setArg(3, capitals);
	if (err != CL_SUCCESS) {
		cout << "Error for volTrendKernel setArg 3: " << err << endl;
	}
	err = volTrendKernel.setArg(4, totalTrades);
	if (err != CL_SUCCESS) {
		cout << "Error for volTrendKernel setArg 4: " << err << endl;
	}
	err = volTrendKernel.setArg(5, wins);
	if (err != CL_SUCCESS) {
		cout << "Error for volTrendKernel setArg 5: " << err << endl;
	}
	err = volTrendKernel.setArg(6, losses);
	if (err != CL_SUCCESS) {
		cout << "Error for volTrendKernel setArg 6: " << err << endl;
	}
#ifdef LIST_TRADES
	err = volTrendKernel.setArg(7, entriesAndExitsBuf);
	if (err != CL_SUCCESS) {
		cout << "Error for volTrendKernel setArg 7: " << err << endl;
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

	err = queue.enqueueNDRangeKernel(volTrendKernel, cl::NullRange, cl::NDRange(comboVect.size()));
	if (err != CL_SUCCESS) {
		cout << "Error for volTrendKernel: " << err << endl;
		return 1;
	}

	vector<cl_double> finalCapitals(comboVect.size());
	vector<cl_int> finalTotalTrades(comboVect.size());
	vector<cl_int> finalWins(comboVect.size());
	vector<cl_int> finalLosses(comboVect.size());

	err = queue.enqueueReadBuffer(totalTrades, CL_TRUE, 0, comboVect.size() * sizeof(cl_int), &finalTotalTrades[0]);
	if (err != CL_SUCCESS) {
		cout << "Error for reading totalTrades: " << err << endl;
		return 1;
	}
	err = queue.enqueueReadBuffer(wins, CL_TRUE, 0, comboVect.size() * sizeof(cl_int), &finalWins[0]);
	if (err != CL_SUCCESS) {
		cout << "Error for reading wins: " << err << endl;
		return 1;
	}
	err = queue.enqueueReadBuffer(losses, CL_TRUE, 0, comboVect.size() * sizeof(cl_int), &finalLosses[0]);
	if (err != CL_SUCCESS) {
		cout << "Error for reading losses: " << err << endl;
		return 1;
	}
	err = queue.enqueueReadBuffer(capitals, CL_TRUE, 0, comboVect.size() * sizeof(cl_double), &finalCapitals[0]);
	if (err != CL_SUCCESS) {
		cout << "Error for reading capitals: " << err << endl;
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

	ofstream outFile;
	outFile.open("resultsCppPar");
	if (outFile.is_open()) {
		for (size_t i = 0; i < comboVect.size(); i++) {
			outFile << "Stop loss: " << to_string(comboVect[i].stopLoss) << endl;
			outFile << "Target: " << to_string(comboVect[i].target) << endl;
			outFile << "Window: " << to_string(comboVect[i].window / ONE_MINUTE_MICROSECONDS) << " minutes" << endl;
			outFile << "Buy volume threshold: " << to_string(comboVect[i].buyVolPercentile) << endl;
			outFile << "Sell volume threshold: " << to_string(comboVect[i].sellVolPercentile) << endl;
			outFile << "Entry threshold: " << to_string(comboVect[i].entryThreshold) << endl;
			outFile << "Total trades: " << to_string(finalTotalTrades[i]) << endl;
			outFile << "Wins: " << to_string(finalWins[i]) << endl;
			outFile << "Losses: " << to_string(finalLosses[i]) << endl;
			outFile << "Final capital: " << to_string(finalCapitals[i]) << endl;
#ifdef LIST_TRADES
			for (int j = 0; j < MAX_TOTAL_TRADES; j++) {
				entryAndExit e = entriesAndExits[i * MAX_TOTAL_TRADES + j];
				if (e.entryIndex == 0) break;
				else {
					if (e.longShortWinLoss & (1 << LONG_BIT)) {
						outFile << "LONG ";
					} else if (e.longShortWinLoss & (1 << SHORT_BIT)) {
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
		int maxElementIdx = std::max_element(finalCapitals.begin(), finalCapitals.end()) - finalCapitals.begin();
		outFile << fixed;
		outFile << "Max return:" << endl;
		outFile << "Final capital: " << finalCapitals[maxElementIdx] << endl;
		outFile << "Stop loss: " << comboVect[maxElementIdx].stopLoss << endl;
		outFile << "Target: " << comboVect[maxElementIdx].target << endl;
		outFile << "Window: " << comboVect[maxElementIdx].window / ONE_MINUTE_MICROSECONDS << " minutes" << endl;
		outFile << "Buy volume threshold: " << comboVect[maxElementIdx].buyVolPercentile << endl;
		outFile << "Sell volume threshold: " << comboVect[maxElementIdx].sellVolPercentile << endl;
		outFile << "Entry threshold: " << comboVect[maxElementIdx].entryThreshold << endl;
		outFile << "Total trades: " << finalTotalTrades[maxElementIdx] << endl;
		outFile << "Wins: " << finalWins[maxElementIdx] << endl;
		outFile << "Losses: " << finalLosses[maxElementIdx] << endl;
#ifdef LIST_TRADES
		for (int j = 0; j < MAX_TOTAL_TRADES; j++) {
			entryAndExit e = entriesAndExits[maxElementIdx * MAX_TOTAL_TRADES + j];
			if (e.entryIndex == 0) break;
			else {
				if (e.longShortWinLoss & (1 << LONG_BIT)) {
					outFile << "LONG ";
				} else if (e.longShortWinLoss & (1 << SHORT_BIT)) {
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
	}

	int maxElementIdx = std::max_element(finalCapitals.begin(), finalCapitals.end()) - finalCapitals.begin();
	cout << fixed;
	cout << "Max return:" << endl;
	cout << "Final capital: " << finalCapitals[maxElementIdx] << endl;
	cout << "Stop loss: " << comboVect[maxElementIdx].stopLoss << endl;
	cout << "Target: " << comboVect[maxElementIdx].target << endl;
	cout << "Window: " << comboVect[maxElementIdx].window / ONE_MINUTE_MICROSECONDS << " minutes" << endl;
	cout << "Buy volume threshold: " << comboVect[maxElementIdx].buyVolPercentile << endl;
	cout << "Sell volume threshold: " << comboVect[maxElementIdx].sellVolPercentile << endl;
	cout << "Entry threshold: " << comboVect[maxElementIdx].entryThreshold << endl;
	cout << "Total trades: " << finalTotalTrades[maxElementIdx] << endl;
	cout << "Wins: " << finalWins[maxElementIdx] << endl;
	cout << "Losses: " << finalLosses[maxElementIdx] << endl;

	return 0;
}
