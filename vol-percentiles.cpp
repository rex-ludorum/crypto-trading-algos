#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <chrono>
#include <tuple>
#include <pthread.h>

using std::ifstream;
using std::istringstream;
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
using std::to_string;

using std::cout;
using std::endl;

#define ONE_HOUR_MICROSECONDS 3600000000
#define FIFTEEN_MINUTES_MICROSECONDS 900000000

#define NUM_WINDOWS 4

#define STARTING_PERCENTILE 25
#define ENDING_PERCENTILE 70

vector<double> qtys;
vector<long long> timestamps;
vector<bool> sides;
vector<int> ids;

vector<long long> windows(NUM_WINDOWS);
vector< vector<double> > allBuyVols(NUM_WINDOWS);
vector< vector<double> > allSellVols(NUM_WINDOWS);

vector<string> split(string const &input) {
	istringstream buffer(input);
	vector<string> ret((istream_iterator<string>(buffer)), istream_iterator<string>());
	return ret;
}

void *performWork(void *arguments){
	int index = *((int*) arguments);
	long long window = windows[index];
	printf("Thread %d: Started. Window: %lld\n", index, window);
	// printf("Thread %d: Ended.\n", index);
	double buyVol = 0, sellVol = 0;
	vector<double> buyVols, sellVols;
	bool startCollectingVols = false;
	tuple<size_t, long long> timeWindow = {0, timestamps[0]};

	for (size_t i = 0; i < qtys.size(); i++) {
		if (sides[i]) sellVol += qtys[i];
		else buyVol += qtys[i];
		if (timestamps[i] - get<1>(timeWindow) > window) {
			startCollectingVols = true;
			size_t j;
			for (j = get<0>(timeWindow); j < i; j++) {
				if (timestamps[i] - timestamps[j] > window) {
					if (sides[j]) sellVol -= qtys[j];
					else buyVol -= qtys[j];
				} else {
					break;
				}
			}
			timeWindow = {j, timestamps[j]};
		}
		if (startCollectingVols) {
			buyVols.push_back(buyVol);
			sellVols.push_back(sellVol);
		}
		if (buyVol < 0 || sellVol < 0) cout << buyVol << " " << sellVol << " " << ids[i] << endl;
	}
	sort(buyVols.begin(), buyVols.end());
	sort(sellVols.begin(), sellVols.end());
	allBuyVols[index] = buyVols;
	allSellVols[index] = sellVols;

	return NULL;
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cout << "No file name specified!" << endl;
		return 1;
	}

#if defined(__WIN64)
	_putenv("TZ=/usr/share/zoneinfo/UTC");
#elif defined(__linux)
	putenv("TZ=/usr/share/zoneinfo/UTC");
#endif

	ifstream myFile;
	for (int i = 1; i < argc; i++) {
		myFile.open(argv[i]);
		if (myFile.is_open()) {
			cout << "Reading " << argv[i] << endl;
			string line;
			while (getline(myFile, line)) {
				vector<string> splits = split(line);
				double qty = stod(splits[4]);
				qtys.emplace_back(qty);
				bool isBuyerMaker;
				istringstream(splits[5]) >> boolalpha >> isBuyerMaker;
				sides.emplace_back(isBuyerMaker);
				string date = splits[1] + "T" + splits[2];
				tm localTimeTm;
				int micros;
				istringstream(date) >> get_time(&localTimeTm, "%Y-%m-%dT%H:%M:%S.") >> micros;
				micros /= 1000;
				localTimeTm.tm_isdst = 0;
				auto tpLocal = system_clock::from_time_t(mktime(&localTimeTm));
				long long timestamp = duration_cast<microseconds>(tpLocal.time_since_epoch()).count() + micros;
				timestamps.emplace_back(timestamp);

				ids.emplace_back(stoi(splits[0]));
			}
			myFile.close();
		}
	}

	long long n = 0;
	generate(windows.begin(), windows.end(), [n] () mutable { return n += FIFTEEN_MINUTES_MICROSECONDS; });

	pthread_t threads[NUM_WINDOWS];
	int thread_args[NUM_WINDOWS];

	for (int i = 0; i < NUM_WINDOWS; i++) {
		printf("In main: Creating thread %d.\n", i);
		thread_args[i] = i;
		int resultCode = pthread_create(&threads[i], NULL, performWork, &thread_args[i]);
		cout << resultCode << endl;
		// assert(!result_code);
	}

	printf("In main: All threads are created.\n");

	//wait for each thread to complete
	for (int i = 0; i < NUM_WINDOWS; i++) {
		int resultCode = pthread_join(threads[i], NULL);
		// assert(!result_code);
		printf("In main: Thread %d has ended.\n", i);
		cout << resultCode << endl;
	}

	std::ofstream outFile;
	outFile.open("buyPercentiles");
	if (outFile.is_open()) {
		for (size_t j = 0; j < allBuyVols.size(); j++) {
			if (j != 0) outFile << endl;
			for (int k = STARTING_PERCENTILE; k <= ENDING_PERCENTILE; k += 5) {
				if (k != STARTING_PERCENTILE) outFile << " ";
				outFile << to_string(allBuyVols[j][(int) (allBuyVols[j].size() * ((double) k / 100))]);
			}
		}
		outFile.close();
	}

	outFile.open("sellPercentiles");
	if (outFile.is_open()) {
		for (size_t j = 0; j < allSellVols.size(); j++) {
			if (j != 0) outFile << endl;
			for (int k = STARTING_PERCENTILE; k <= ENDING_PERCENTILE; k += 5) {
				if (k != STARTING_PERCENTILE) outFile << " ";
				outFile << to_string(allSellVols[j][(int) (allSellVols[j].size() * ((double) k / 100))]);
			}
		}
		outFile.close();
	}

	return 0;
}
