#include <algorithm>
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

using std::cout;
using std::endl;

#define ONE_HOUR_MICROSECONDS 3600000000
#define FIFTEEN_MINUTES_MICROSECONDS 900000000

#define NUM_WINDOWS 16

vector<string> split(string const &input) {
	istringstream buffer(input);
	vector<string> ret((istream_iterator<string>(buffer)), istream_iterator<string>());
	return ret;
}

int main() {
	_putenv("TZ=/usr/share/zoneinfo/UTC");
	ifstream myFile;
	myFile.open("BTC-USD_2024-07-22-02.00.00.000000000_2024-11-08-02.00.00.000000000");

	vector<double> qtys;
	vector<long long> timestamps;
	vector<bool> sides;

	double buyVol = 0, sellVol = 0;
	vector<double> allBuyVols, allSellVols;
	bool startCollectingVols = false;
	tuple<int, long long> timeWindow = {0, 0};
	int idx = 0;

	if (myFile.is_open()) {
		string line;
		vector<string> l;
		while (getline(myFile, line)) {
			vector<string> splits = split(line);
			double qty = stod(splits[4]);
			qtys.emplace_back(qty);
			bool isBuyerMaker;
			sides.emplace_back(isBuyerMaker);
			istringstream(splits[5]) >> boolalpha >> isBuyerMaker;
			string date = splits[1] + "T" + splits[2];
			tm localTimeTm;
			int micros;
			istringstream(date) >> get_time(&localTimeTm, "%Y-%m-%dT%H:%M:%S.") >> micros;
			micros /= 1000;
			localTimeTm.tm_isdst = 0;
			auto tpLocal = system_clock::from_time_t(mktime(&localTimeTm));
			long long timestamp = duration_cast<microseconds>(tpLocal.time_since_epoch()).count() + micros;
			timestamps.emplace_back(timestamp);
			if (get<1>(timeWindow) == 0) timeWindow = {0, timestamp};

			if (isBuyerMaker) sellVol += qty;
			else buyVol += qty;
			if (timestamp - get<1>(timeWindow) > FIFTEEN_MINUTES_MICROSECONDS) {
				startCollectingVols = true;
				for (int i = get<0>(timeWindow); i < idx; i++) {
					if (timestamp - timestamps[i] > FIFTEEN_MINUTES_MICROSECONDS) {
						if (sides[i]) sellVol -= qtys[i];
						else buyVol -= qtys[i];
					} else {
						timeWindow = {i, timestamps[i]};
						break;
					}
				}
			}
			if (startCollectingVols) {
				allBuyVols.push_back(buyVol);
				allSellVols.push_back(sellVol);
			}

			idx += 1;
		}
	}

	return 0;
}
