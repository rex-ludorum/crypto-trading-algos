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

#define NUM_WINDOWS 2

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

string joinStrings(tuple<bool, double, string, int> t) {
	string s = get<0>(t) ? "LONG " : "SHORT ";
	s += to_string(get<1>(t));
	s += " ";
	s += get<2>(t);
	s += " ";
	s += to_string(get<3>(t));
	return s;
}

int main() {
	_putenv("TZ=/usr/share/zoneinfo/UTC");
	ifstream myFile;
	myFile.open("BTC-USD_2024-07-18-23.00.00.000000000_2024-07-22-02.00.00.000000000");
	vector<trade> trades;

	vector<long long> windows(NUM_WINDOWS);
	long long n = 0;
	generate(windows.begin(), windows.end(), [n] () mutable { return n += FIFTEEN_MINUTES_MICROSECONDS; });

	// vector<double> stopLosses(6);
	vector<double> stopLosses(2);
	double x = 0;
	generate(stopLosses.begin(), stopLosses.end(), [x] () mutable { return x += 0.5; });

	// vector<double> targets(9);
	vector<double> targets(2);
	x = 0.5;
	generate(targets.begin(), targets.end(), [x] () mutable { return x += 0.5; });

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
			trades.emplace_back(t);
		}
		myFile.close();
  }

	vector< tuple<int, long long> > timeWindows(NUM_WINDOWS, {0, trades[0].timestamp});

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

	auto firstCombo = cartesian_product(buyVolPercentiles[0], sellVolPercentiles[0], stopLosses, targets);
	vector< decltype(firstCombo) > combos;
	combos.emplace_back(firstCombo);

	for (int i = 1; i < NUM_WINDOWS; i++) {
		combos.emplace_back(cartesian_product(buyVolPercentiles[i], sellVolPercentiles[i], stopLosses, targets));
	}

	vector< vector< tuple<bool, double, string, int> > > entries(NUM_WINDOWS);
	vector< vector< vector<string> > > tradeLogs(NUM_WINDOWS);
	vector< vector<double> > startingCapitals(NUM_WINDOWS);
	vector< vector<double> > maxProfits(NUM_WINDOWS);
	vector< vector<int> > totalTrades(NUM_WINDOWS);
	vector< vector<int> > wins(NUM_WINDOWS);
	vector< vector<int> > losses(NUM_WINDOWS);

	for (int i = 0; i < NUM_WINDOWS; i++) {
		entries[i] = vector< tuple<bool, double, string, int> >(combos[i].size(), {false, 0.0, "", 0});
		tradeLogs[i] = vector< vector<string> >(combos[i].size());
		startingCapitals[i] = vector<double>(combos[i].size(), 1.0);
		maxProfits[i] = vector<double>(combos[i].size(), 0.0);
		totalTrades[i] = vector<int>(combos[i].size(), 0);
		wins[i] = vector<int>(combos[i].size(), 0);
		losses[i] = vector<int>(combos[i].size(), 0);
	}

	for (int i = 0; i < trades.size(); i++) {
		cout << i << endl;
		double vol = trades[i].qty;
		double price = trades[i].price;
		long long microseconds = trades[i].timestamp;

		for (int j = 0; j < windows.size(); j++) {
			if (microseconds - get<1>(timeWindows[j]) > windows[j]) {
				for (int k = get<0>(timeWindows[j]); k < i; k++) {
					long long newMicroseconds = trades[k].timestamp;
					if (microseconds - newMicroseconds > windows[j]) {
						double newVol = trades[k].qty;
						if (trades[k].isBuyerMaker) sellVols[j] -= newVol;
						else buyVols[j] -= newVol;
					} else {
						timeWindows[j] = {k, newMicroseconds};
						break;
					}
				}
			}
			if (trades[i].isBuyerMaker) sellVols[j] += vol;
			else buyVols[j] += vol;

			for (int k = 0; k < combos[j].size(); k++) {
				double buyVolPercentile = get<0>(combos[j][k]);
				double sellVolPercentile = get<1>(combos[j][k]);
				double stopLoss = get<2>(combos[j][k]);
				double target = get<3>(combos[j][k]);

				if (get<3>(entries[j][k]) == 0) {
					if (buyVols[j] >= buyVolPercentile) {
						maxProfits[j][k] = price;
						entries[j][k] = {true, price, trades[i].date, trades[i].tradeId};
						tradeLogs[j][k].emplace_back(joinStrings(entries[j][k]));
						totalTrades[j][k] += 1;
					} else if (sellVols[j] >= sellVolPercentile) {
						maxProfits[j][k] = price;
						entries[j][k] = {false, price, trades[i].date, trades[i].tradeId};
						tradeLogs[j][k].emplace_back(joinStrings(entries[j][k]));
						totalTrades[j][k] += 1;
					}
				} else {
					if (get<0>(entries[j][k])) {
						maxProfits[j][k] = max(maxProfits[j][k], price);
						double profitMargin = (maxProfits[j][k] - get<1>(entries[j][k])) / get<1>(entries[j][k]);
						if (profitMargin >= target / 100) {
							startingCapitals[j][k] *= 1 + profitMargin;
							entries[j][k] = {false, 0.0, "", 0};
							tradeLogs[j][k].emplace_back("Profit: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
							tradeLogs[j][k].emplace_back("Capital: " + to_string(startingCapitals[j][k]));
							wins[j][k] += 1;
						} else if (price <= (1 - stopLoss / 100) * get<1>(entries[j][k])) {
							startingCapitals[j][k] *= 1 - stopLoss / 100;
							entries[j][k] = {false, 0.0, "", 0};
							tradeLogs[j][k].emplace_back("Loss: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
							tradeLogs[j][k].emplace_back("Capital: " + to_string(startingCapitals[j][k]));
							losses[j][k] += 1;
						}
					} else {
						maxProfits[j][k] = min(maxProfits[j][k], price);
						double profitMargin = (get<1>(entries[j][k]) - maxProfits[j][k]) / get<1>(entries[j][k]);
						if (profitMargin >= target / 100) {
							startingCapitals[j][k] *= 1 + profitMargin;
							entries[j][k] = {false, 0.0, "", 0};
							tradeLogs[j][k].emplace_back("Profit: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
							tradeLogs[j][k].emplace_back("Capital: " + to_string(startingCapitals[j][k]));
							wins[j][k] += 1;
						} else if (price >= (1 + stopLoss / 100) * get<1>(entries[j][k])) {
							startingCapitals[j][k] *= 1 - stopLoss / 100;
							entries[j][k] = {false, 0.0, "", 0};
							tradeLogs[j][k].emplace_back("Loss: " + to_string(price) + " " + trades[i].date + " " + to_string(trades[i].tradeId));
							tradeLogs[j][k].emplace_back("Capital: " + to_string(startingCapitals[j][k]));
							losses[j][k] += 1;
						}
					}
				}
			}
		}
	}
	cout << "7" << endl;

	std::ofstream outFile;
	outFile.open("resultsCpp");
	if (outFile.is_open()) {
		for (int i = 0; i < NUM_WINDOWS; i++) {
			for (int j = 0; j < combos[i].size(); j++) {
				outFile << "Time window: " << to_string(windows[i]) << endl;
				outFile << "Buy vol threshold: " << to_string(get<0>(combos[i][j])) << endl;
				outFile << "Sell vol threshold: " << to_string(get<1>(combos[i][j])) << endl;
				outFile << "Stop loss: " << to_string(get<2>(combos[i][j])) << endl;
				outFile << "Target: " << to_string(get<3>(combos[i][j])) << endl;
				outFile << "Total trades: " << to_string(totalTrades[i][j]) << endl;
				outFile << "Wins: " << to_string(wins[i][j]) << endl;
				outFile << "Losses: " << to_string(losses[i][j]) << endl;
				outFile << "Final capital: " << to_string(startingCapitals[i][j]) << endl;
				for (string s : tradeLogs[i][j]) {
					outFile << s << endl;
				}
				outFile << endl;
			}
		}
		outFile.close();
  }

	return 0;
}
