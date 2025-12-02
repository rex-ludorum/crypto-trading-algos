import json
import requests
import time
import traceback
import os
import dateutil.parser
import datetime
import argparse
from coinbase import jwt_generator
from functools import cmp_to_key
from enum import Enum, auto

COINBASE_API_KEY_NAME = os.environ.get("COINBASE_API_KEY_NAME")
COINBASE_API_PRIVATE_KEY = os.environ.get("COINBASE_API_PRIVATE_KEY")

MAX_REST_API_TRADES = 1000

ONE_SECOND_MAX_TRADES = 750

SLIDING_WINDOW_SIZE = 30
MAX_WINDOW_SIZE = 200
MIN_WINDOW_SIZE = 40
MAX_OFFSET = 120

class RetVal(Enum):
	WAIT = auto()
	SUCCESS = auto()
	GAP_EXCEEDED = auto()
	FAILURE = auto()

def printError(error):
	errorMessage = repr(error) + " encountered for " + symbol + " at " + str(time.strftime("%H:%M:%S", time.localtime()))
	print(errorMessage)

def prepareRecord(response):
	# These two fields are only present in REST API responses
	if 'bid' in response:
		response.pop('bid')
	if 'ask' in response:
		response.pop('ask')

	makerSide = response['side']
	if (makerSide != 'BUY' and makerSide != 'SELL'):
		'''
		print("Unknown maker side for " + json.dumps(response))
		return {}
		'''
		raise ValueError("Unknown maker side for " + json.dumps(response))
	isBuyerMaker = makerSide == 'BUY'

	formattedDate = dateutil.parser.isoparse(response['time'])
	microseconds = round(datetime.datetime.timestamp(formattedDate) * 1000000)

	# try:
		# Coinbase sometimes sends trades with non-integer IDs
		# Skip the trade if that's the case
	tradeId = int(response['trade_id'])
	record = [
		str(microseconds),
		str(tradeId),
		str(response['price']),
		str(response['size']),
		str(isBuyerMaker)
	]
	return record

# Obtain the range of aggregate IDs we are writing
def getTradeIds(records):
	tradeIds = []
	for record in records:
		tradeIds.append(int(record[1]))

	return tradeIds

# Timestream does not allow two records with the same timestamp and dimensions to have different measure values
# Therefore, add one us to the later timestamp
def updateRecordTime(record, lastTrade):
	recordTime = record[0]
	if lastTrade and lastTrade['Time'] != '0' and int(record[0]) <= int(lastTrade['Time']) + int(lastTrade['offset']):
		record[0] = str(int(lastTrade['Time']) + int(lastTrade['offset']) + 1)
		# print("Time %s for %s conflicts with last trade time (%s with offset %s, tradeId %s), updating to %s" % (recordTime, symbol, lastTrade['Time'], lastTrade['offset'], lastTrade['tradeId'], record['Time']))
		lastTrade['offset'] = str(int(lastTrade['offset']) + 1)
	else:
		lastTrade['Time'] = recordTime
		lastTrade['offset'] = str(0)
	lastTrade['tradeId'] = getTradeIds([record])[0]

def adjustWindow(record, windows):
	recordTime = str(int(record[0]) // 1000000)
	if windows['startTime'] == '':
		windows['startTime'] = recordTime
		windows['windows'].append(1)
	else:
		gap = int(recordTime) - int(windows['startTime']) - len(windows['windows'])
		assert(gap >= -1)
		if gap >= 0:
			windows['windows'] += gap * [0]
			windows['windows'].append(1)
			if len(windows['windows']) - SLIDING_WINDOW_SIZE > 0:
				windows['windows'] = windows['windows'][len(windows['windows']) - SLIDING_WINDOW_SIZE:]
			windows['startTime'] = str(int(recordTime) - len(windows['windows']) + 1)
		else:
			windows['windows'][-1] += 1

def computeAverage(windows):
	if len(windows['windows']) == 0:
		return 0
	else:
		return sum(windows['windows']) // len(windows['windows'])

def computeOffset(windows):
	if not windows['windows']:
		return MAX_OFFSET
	elif max(windows["windows"]) < MIN_WINDOW_SIZE:
		windowOffset = max(ONE_SECOND_MAX_TRADES // max(1, max(windows['windows'])), 1)
		windowOffset = min(windowOffset, MAX_OFFSET)
		return windowOffset
	elif max(windows["windows"]) < MAX_WINDOW_SIZE:
		windowOffset = max(ONE_SECOND_MAX_TRADES // max(1, computeAverage(windows)), 1)
		windowOffset = min(windowOffset, MAX_OFFSET)
		return windowOffset
	else:
		return 1

# If we have to reconnect after a websocket exception, get any trades we might have missed
def handleGap(startTime, startId, endTime, endId):
	if int(endId) > int(startId) + 1:
		time.sleep(0.5)
		lastTrade = {'Time': '0', 'offset': '0', 'tradeId': str(startId)}
		startDate = datetime.datetime.fromtimestamp(startTime, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
		endDate = datetime.datetime.fromtimestamp(endTime, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
		with open('%s_%s_%s.csv' % (symbol, startDate.replace("T", "-").replace(":", "."), endDate.replace("T", "-").replace(":", ".")), 'w') as file:
			file.write(",".join(["time", "measure_name", "symbol", "tradeId", "price", "size", "isBuyerMaker"]) + '\n')
			logMsg = "Gap found: %s - %s (%s - %s)" % (startId, endId, startDate, endDate)
			print(logMsg)
			prevLastTradeId = startId
			missedTrades = []
			startingLastTradeId = startId

			# Use a three-second-minimum window since the max observed trades in a one-second window was 260 on Feb 28 2024
			# Actually, see the test commands for a one-second window with more than 1000 trades
			windows = {'startTime': '', 'windows': []}
			windowOffset = computeOffset(windows)

			while True:
				while (retVal := getGap(endId, min(startTime + windowOffset, endTime), startTime, lastTrade, missedTrades, windows, file)) == RetVal.WAIT:
					# Rate limit is 30 requests per second
					time.sleep(1 / 30)

				if retVal == RetVal.FAILURE:
					break
				if retVal == RetVal.SUCCESS and startTime + windowOffset >= endTime or endId == int(lastTrade['tradeId']) + 1:
					break

				if retVal == RetVal.GAP_EXCEEDED:
					windowOffset = 1
				elif str(prevLastTradeId) == lastTrade['tradeId'] or retVal == RetVal.SUCCESS:
					startTime = startTime + windowOffset
					windowOffset = computeOffset(windows)
				else:
					# Might be able to just use the previous endTime as the new startTime like in the above case
					startTime = int(lastTrade['Time']) // 1000000
					windowOffset = computeOffset(windows)
				prevLastTradeId = lastTrade['tradeId']

			if not missedTrades:
				missedTrades = list(range(int(lastTrade['tradeId']) + 1, endId))
			else:
				lastMissedTradeId = max(int(lastTrade['tradeId']) + 1, missedTrades[-1] + 1)
				missedTrades.extend(range(lastMissedTradeId, endId))

			if not all(missedTrades[i] <= missedTrades[i + 1] for i in range(len(missedTrades) - 1)):
				printError(RuntimeError("List of missed trades is not in increasing order: %s" % (" ".join(str(x) for x in missedTrades))))
			if len(missedTrades) > len(set(missedTrades)):
				printError(RuntimeError("List of missed trades has duplicates: %s" % (" ".join(str(x) for x in missedTrades))))
			if endId != int(lastTrade['tradeId']) + 1 or missedTrades:
				errMsg = "Gaps still exist between %s and %s: %s\n" % (startingLastTradeId, endId, ", ".join(getMissedRanges(missedTrades)))
				printError(RuntimeError(errMsg))

def getGap(endId, endTime, startTime, lastTrade, missedTrades, windows, file):
	url = "https://api.coinbase.com/api/v3/brokerage/products/%s/ticker" % (symbol)
	params = {"limit": MAX_REST_API_TRADES, "start": str(startTime), "end": str(endTime)}
	jwt_uri = jwt_generator.format_jwt_uri("GET", "/api/v3/brokerage/products/%s/ticker" % (symbol))
	jwt_token = jwt_generator.build_rest_jwt(jwt_uri, COINBASE_API_KEY_NAME, COINBASE_API_PRIVATE_KEY)
	headers = {"Authorization": "Bearer " + jwt_token}
	startDate = datetime.datetime.fromtimestamp(startTime, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
	endDate = datetime.datetime.fromtimestamp(endTime, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
	try:
		logMsg = "Sending HTTP request for %s trades from %s to %s (lastTradeId: %s)" % (symbol, startDate, endDate, lastTrade['tradeId'])
		print(logMsg)
		response = requests.get(url, params=params, headers=headers)
		response.raise_for_status()
		responseTrades = response.json()['trades']
		length = len(responseTrades)
		if not responseTrades:
			logMsg = "HTTP response contains 0 trades"
			print(logMsg)
			return RetVal.SUCCESS

		cleanTrades(responseTrades)
		if not responseTrades:
			logMsg = "HTTP response contains 0 trades (after cleaning)"
			print(logMsg)
			return RetVal.SUCCESS

		logMsg = "HTTP response contains %d trades (%s) (%s - %s)" % (len(responseTrades), ", ".join(getRanges(responseTrades)), responseTrades[0]['time'], responseTrades[-1]['time'])
		print(logMsg)
		# print(responseTrades)

		tradeId = int(lastTrade['tradeId']) + 1
		idx = next((i for i, x in enumerate(responseTrades) if int(x['trade_id']) >= tradeId), -1)
		if idx != -1:
			if length >= MAX_REST_API_TRADES and endTime - startTime > 1:
				lastTradeTime = datetime.datetime.fromtimestamp(int(lastTrade['Time']) // 1000000, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
				logMsg = "Moving gap back, lastTrade has timestamp %s.%s" % (lastTradeTime, str((int(lastTrade['Time']) % 1000000)).zfill(6))
				print(logMsg)
				return RetVal.GAP_EXCEEDED

		'''
		windows = []
		idx = 0
		currWindow = 0
		while idx < len(responseTrades):
			windows.append(currWindow)
			currWindow = 0
			formattedDate = dateutil.parser.isoparse(responseTrades[idx]['time'])
			print(int(datetime.datetime.timestamp(formattedDate)))
			window = int(datetime.datetime.timestamp(formattedDate)) * 1000000
			microseconds = window
			print(window)
			print(window + 1000000)
			print("follows")
			while (microseconds < window + 1000000):
				print(microseconds)
				currWindow += 1
				idx += 1
				if idx >= len(responseTrades):
					windows.append(currWindow)
					break;
				formattedDate = dateutil.parser.isoparse(responseTrades[idx]['time'])
				microseconds = int(datetime.datetime.timestamp(formattedDate) * 1000000)
		print(max(windows))
		print(windows)
		'''

		# Fringe case for when the lastTrade comes after the trades in the response
		if int(lastTrade['tradeId']) >= int(responseTrades[-1]['trade_id']):
			return RetVal.SUCCESS

		while (tradeId < endId):
			idx = next((i for i, x in enumerate(responseTrades) if int(x['trade_id']) == tradeId), -1)
			if (idx != -1):
				record = prepareRecord(responseTrades[idx])
				updateRecordTime(record, lastTrade)
				adjustWindow(record, windows)
				tempRecord = record.copy()
				tempRecord.insert(1, symbol)
				tempRecord.insert(1, "price")
				file.write(",".join(tempRecord) + '\n')
				# How do we know if we got all the trades in this given window or if there are still missing ones after the last?
				if (idx == len(responseTrades) - 1):
					break;
			else:
				missedTrades.append(tradeId)
				# printError(LookupError("Trade ID " + str(tradeId) + " not found"), "Requests")
			tradeId += 1
		return RetVal.SUCCESS
	except requests.HTTPError as e:
		logMsg = "Encountered HTTPError %s" % (repr(e))
		print(logMsg)
		traceback.print_exc()
		printError(e)
		if response.status_code == 429 or response.status_code == 502 or response.status_code == 500 or response.status_code == 401 or response.status_code == 524 or response.status_code == 503 or response.status_code == 404 or response.status_code == 504:
			return RetVal.WAIT
		else:
			return RetVal.FAILURE
	except requests.ConnectionError as e:
		logMsg = "Encountered ConnectionError %s" % (repr(e))
		print(logMsg)
		traceback.print_exc()
		printError(e)
		return RetVal.WAIT
	except requests.exceptions.ChunkedEncodingError as e:
		logMsg = "Encountered ChunkedEncodingError %s" % (repr(e))
		print(logMsg)
		traceback.print_exc()
		printError(e)
		return RetVal.WAIT
	except Exception as e:
		logMsg = "Encountered other exception %s" % (repr(e))
		print(logMsg)
		traceback.print_exc()
		printError(e)
		return RetVal.FAILURE

def cleanTrades(trades):
	for idx, _ in enumerate(trades):
		MoreWeirdTradeIds = True
		while MoreWeirdTradeIds:
			try:
				if (idx >= len(trades)):
					break
				int(trades[idx]['trade_id'])
				MoreWeirdTradeIds = False
			except ValueError:
				trades.pop(idx)
	trades.sort(key=cmp_to_key(lambda item1, item2: int(item1['trade_id']) - int(item2['trade_id'])))

	# For earlier data, there are sometimes duplicate trades
	for idx, trade in enumerate(trades):
		if idx + 1 >= len(trades):
			break
		while trades[idx + 1]["trade_id"] == trade["trade_id"]:
			trades.pop(idx + 1)
			if idx + 1 >= len(trades):
				break

def getRanges(trades):
	if len(trades) == 1:
		return [trades[0]["trade_id"]]
	if not trades:
		return []

	ranges = []
	lastContiguousId = trades[0]["trade_id"]
	for idx, trade in enumerate(trades):
		if idx == 0:
			continue
		prevId = trades[idx - 1]["trade_id"]
		if int(trade["trade_id"]) - 1 != int(prevId):
			if lastContiguousId == prevId:
				ranges.append(lastContiguousId)
			else:
				ranges.append(lastContiguousId + "-" + prevId)
			lastContiguousId = trade["trade_id"]
		if idx == len(trades) - 1:
			if lastContiguousId == trade["trade_id"]:
				ranges.append(lastContiguousId)
			else:
				ranges.append(lastContiguousId + "-" + trade["trade_id"])
	return ranges

def getMissedRanges(ids):
	if len(ids) == 1:
		return [str(ids[0])]
	if not ids:
		return []

	ranges = []
	lastContiguousId = ids[0]
	for idx, trade in enumerate(ids):
		if idx == 0:
			continue
		prevId = ids[idx - 1]
		if trade - 1 != prevId:
			if lastContiguousId == prevId:
				ranges.append(str(lastContiguousId))
			else:
				ranges.append(str(lastContiguousId) + "-" + str(prevId))
			lastContiguousId = trade
		if idx == len(ids) - 1:
			if lastContiguousId == trade:
				ranges.append(str(lastContiguousId))
			else:
				ranges.append(str(lastContiguousId) + "-" + str(trade))
	return ranges

parser = argparse.ArgumentParser(description='Collect trading data from Coinbase and save it to a file.')
parser.add_argument('symbol', help='the trading pair to collect data from', choices=['BTC-USD', 'ETH-USD'])
parser.add_argument('startId', help='the trade ID of the trade preceding the gap', type=int)
parser.add_argument('startTimestamp', help='the timestamp of the trade preceding the gap in seconds, rounded down', type=int)
parser.add_argument('endId', help='the trade ID of the trade following the gap', type=int)
parser.add_argument('endTimestamp', help='the timestamp of the trade following the gap in seconds, rounded up', type=int)
args = parser.parse_args()
symbol = vars(args)['symbol']
startId = vars(args)['startId']
startTimestamp = vars(args)['startTimestamp']
endId = vars(args)['endId']
endTimestamp = vars(args)['endTimestamp']

handleGap(startTimestamp, startId, endTimestamp, endId)

# Test commands for handling gaps
# handleGap({'trade_id': '999999999', 'time': '2024-06-02T00:00:00.000000Z', 'product_id': 'BTC-USD'}, [], {'Time': '1617216665966502', 'offset': '0', 'tradeId': '151436694'}, writeClient, commonAttributes, mysns)
# handleGap({'trade_id': '151436698', 'time': '2024-06-02T00:00:00.000000Z', 'product_id': 'BTC-USD'}, [], {'Time': '1617216665966502', 'offset': '0', 'tradeId': '151436694'}, {}, {}, {})

# Used to find the max window size (260)
# getGap(symbol, 999999999, 1709144520, [], 1705311931, {}, [], [], [])

# Shows that there are unavoidable gaps in the trade data
# getGap(symbol, 655395509, 1719187536, [], 1719187535, {'tradeId': '655395410'}, [], [], [])

# Shows that you should keep the window as small as possible otherwise trades might be missed
# getGap(symbol, 655395509, 1719187537, [], 1719187535, {'tradeId': '655395410'}, [], [], [])
# getGap(symbol, 655395509, 1719187537, [], 1719187536, {'tradeId': '655395410'}, [], [], [])
