#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <pthread.h>
#include <string>
#include <tuple>
#include <vector>

using std::boolalpha;
using std::generate;
using std::get;
using std::get_time;
using std::getline;
using std::ifstream;
using std::istream_iterator;
using std::istringstream;
using std::mktime;
using std::stod;
using std::stoi;
using std::string;
using std::tm;
using std::to_string;
using std::tolower;
using std::transform;
using std::tuple;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::system_clock;

using std::cout;
using std::endl;

#define FIFTEEN_MINUTES_MICROSECONDS 900000000

#define NUM_WINDOWS 16

#define STARTING_PERCENTILE 5

#define MAX_THREADS 2

vector<double> qtys;
vector<double> prices;
vector<long long> timestamps;
vector<bool> sides;
vector<int> ids;

vector<long long> windows(NUM_WINDOWS);
vector<vector<double>> allBuyVols(NUM_WINDOWS);
vector<vector<double>> allSellVols(NUM_WINDOWS);
vector<vector<double>> allDeltaVols(NUM_WINDOWS);

vector<string> splitBySpace(string const &input) {
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

void *performWork(void *arguments) {
	int index = *((int *)arguments);
	long long window = windows[index];
	printf("Thread %d: Started. Window: %lld\n", index, window);
	// printf("Thread %d: Ended.\n", index);
	double buyVol = 0, sellVol = 0;
	vector<double> buyVols, sellVols, deltaVols;
	bool startCollectingVols = false;
	tuple<size_t, long long> timeWindow = {0, timestamps[0]};

	for (size_t i = 0; i < qtys.size(); i++) {
		if (sides[i])
			sellVol += qtys[i] * prices[i];
		else
			buyVol += qtys[i] * prices[i];
		if (timestamps[i] - get<1>(timeWindow) > window) {
			startCollectingVols = true;
			size_t j;
			for (j = get<0>(timeWindow); j < i; j++) {
				if (timestamps[i] - timestamps[j] > window) {
					if (sides[j])
						sellVol -= qtys[j] * prices[j];
					else
						buyVol -= qtys[j] * prices[j];
				} else {
					break;
				}
			}
			timeWindow = {j, timestamps[j]};
		}
		if (startCollectingVols) {
			buyVols.push_back(buyVol);
			sellVols.push_back(sellVol);
			deltaVols.push_back(buyVol - sellVol);
		}
		// if (buyVol < 0 || sellVol < 0) cout << buyVol << " " << sellVol << " " <<
		// ids[i] << endl;
	}
	sort(buyVols.begin(), buyVols.end());
	sort(sellVols.begin(), sellVols.end());
	sort(deltaVols.begin(), deltaVols.end());

	vector<double> finalBuyPercentiles, finalSellPercentiles,
			finalDeltaPercentiles;
	for (double k = STARTING_PERCENTILE; k <= 96; k += 2.5) {
		finalBuyPercentiles.push_back(
				buyVols[(int)(buyVols.size() * ((double)k / 100))]);
		finalSellPercentiles.push_back(
				sellVols[(int)(sellVols.size() * ((double)k / 100))]);
		finalDeltaPercentiles.push_back(
				deltaVols[(int)(deltaVols.size() * ((double)k / 100))]);
	}

	allBuyVols[index] = finalBuyPercentiles;
	allSellVols[index] = finalSellPercentiles;
	allDeltaVols[index] = finalDeltaPercentiles;

	return NULL;
}

int main(int argc, char *argv[]) {
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
			getline(myFile, line);
			while (getline(myFile, line)) {
				vector<string> splits = splitByComma(line);
				double qty = stod(splits[5]);
				qtys.emplace_back(qty);
				prices.emplace_back(stod(splits[4]));
				bool isBuyerMaker;
				string isBuyerMakerString = splits[6];
				transform(isBuyerMakerString.begin(), isBuyerMakerString.end(),
									isBuyerMakerString.begin(),
									[](unsigned char c) { return tolower(c); });
				istringstream(isBuyerMakerString) >> boolalpha >> isBuyerMaker;
				sides.emplace_back(isBuyerMaker);

				/*
				string date = splits[1] + "T" + splits[2];
				tm localTimeTm;
				int micros;
				istringstream(date) >> get_time(&localTimeTm, "%Y-%m-%dT%H:%M:%S.") >>
				micros; micros /= 1000; localTimeTm.tm_isdst = 0; auto tpLocal =
				system_clock::from_time_t(mktime(&localTimeTm)); long long timestamp =
				duration_cast<microseconds>(tpLocal.time_since_epoch()).count() +
				micros;
				 */

				long long timestamp = stoll(splits[0]);
				timestamps.emplace_back(timestamp);

				// ids.emplace_back(stoi(splits[0]));
			}
			myFile.close();
		}
	}

	long long n = 0;
	generate(windows.begin(), windows.end(),
					 [n]() mutable { return n += FIFTEEN_MINUTES_MICROSECONDS; });

	pthread_t threads[NUM_WINDOWS];
	int thread_args[NUM_WINDOWS];

	int i = 0;

	while (i < NUM_WINDOWS) {
		for (int j = i; j < i + MAX_THREADS && j < NUM_WINDOWS; j++) {
			printf("In main: Creating thread %d.\n", j);
			thread_args[j] = j;
			int resultCode =
					pthread_create(&threads[j], NULL, performWork, &thread_args[j]);
			cout << resultCode << endl;
		}

		int j;
		for (j = i; j < i + MAX_THREADS && j < NUM_WINDOWS; j++) {
			int resultCode = pthread_join(threads[j], NULL);
			// assert(!result_code);
			printf("In main: Thread %d has ended.\n", j);
			cout << resultCode << endl;
		}

		i = j;
	}

	/*
	for (int i = 0; i < NUM_WINDOWS; i++) {
		printf("In main: Creating thread %d.\n", i);
		thread_args[i] = i;
		int resultCode =
				pthread_create(&threads[i], NULL, performWork, &thread_args[i]);
		cout << resultCode << endl;
		// assert(!result_code);

		resultCode = pthread_join(threads[i], NULL);
		// assert(!result_code);
		printf("In main: Thread %d has ended.\n", i);
		cout << resultCode << endl;
	}

	printf("In main: All threads are created.\n");

	//wait for each thread to complete
	for (int i = 0; i < NUM_WINDOWS; i++) {
		int resultCode = pthread_join(threads[i], NULL);
		// assert(!result_code);
		printf("In main: Thread %d has ended.\n", i);
		cout << resultCode << endl;
	}
	*/

	std::ofstream outFile;
	outFile.open("buyPercentiles");
	if (outFile.is_open()) {
		for (size_t j = 0; j < allBuyVols.size(); j++) {
			if (j != 0)
				outFile << endl;
			bool first = true;
			for (auto v : allBuyVols[j]) {
				if (!first)
					outFile << " ";
				else
					first = false;
				outFile << to_string(v);
			}
		}
		outFile.close();
	}

	outFile.open("sellPercentiles");
	if (outFile.is_open()) {
		for (size_t j = 0; j < allSellVols.size(); j++) {
			if (j != 0)
				outFile << endl;
			bool first = true;
			for (auto v : allSellVols[j]) {
				if (!first)
					outFile << " ";
				else
					first = false;
				outFile << to_string(v);
			}
		}
		outFile.close();
	}

	outFile.open("deltaPercentiles");
	if (outFile.is_open()) {
		for (size_t j = 0; j < allSellVols.size(); j++) {
			if (j != 0)
				outFile << endl;
			bool first = true;
			for (auto v : allDeltaVols[j]) {
				if (!first)
					outFile << " ";
				else
					first = false;
				outFile << to_string(v);
			}
		}
		outFile.close();
	}

	return 0;
}
