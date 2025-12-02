import argparse
import requests
import os
import traceback
import time
import datetime
import dateutil.parser
from functools import cmp_to_key
from coinbase import jwt_generator
from enum import Enum, auto

COINBASE_API_KEY_NAME = os.environ.get("COINBASE_API_KEY_NAME")
COINBASE_API_PRIVATE_KEY = os.environ.get("COINBASE_API_PRIVATE_KEY")

MAX_REST_API_TRADES = 1000

ONE_SECOND_MAX_TRADES = 750

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

def getMicrosecondsFromDate(trade):
	date = trade[1] + "T" + trade[2] + "Z"
	parsedDate = dateutil.parser.isoparse(date)
	return round(datetime.datetime.timestamp(parsedDate) * 1000000)

def checkGaps():
	with open(inputFile, 'r') as f:
		ids = []
		timestamps = []
		for line in f:
			if '.csv' in inputFile and 'time' in line:
				continue

			if '.csv' in inputFile:
				data = line.split(",")
				ids.append(int(data[3]))
				timestamps.append(int(data[0]))
			else:
				data = line.split(" ")
				ids.append(int(data[0]))
				timestamps.append(getMicrosecondsFromDate(data))

		missedRanges = getMissedRanges(ids, timestamps)
		print(missedRanges[0])
		print(missedRanges[1])

def getIntRanges(ids):
	if len(ids) == 1:
		return [ids[0]]
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

def getMissedRanges(ids, timestamps):
	if len(ids) == 1:
		return []
	if not ids:
		return []

	ranges = []
	diffs = []
	lastResponse = []
	with open('fill.csv', 'w') as file:
		file.write(",".join(["time", "measure_name", "symbol", "tradeId", "price", "size", "isBuyerMaker"]) + '\n')

		for idx, trade in enumerate(ids):
			if idx == 0:
				continue
			prevId = ids[idx - 1]
			if trade - 1 != prevId:
				startMicros = str(timestamps[idx - 1] % 1000000)
				endMicros = str(timestamps[idx] % 1000000)
				startDate = datetime.datetime.fromtimestamp(timestamps[idx - 1] // 1000000, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + '.' + startMicros.zfill(6)
				endDate = datetime.datetime.fromtimestamp(timestamps[idx] // 1000000, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + '.' + endMicros.zfill(6)
				print("Gap found: %d - %d (%s - %s)" % (prevId, trade, startDate, endDate))
				if trade - prevId > 2:
					ranges.append(str(prevId + 1) + "-" + str(trade - 1))
				else:
					ranges.append(str(prevId + 1))
				diffs.append(trade - prevId - 1)

				handleGap(timestamps[idx - 1] // 1000000, prevId, timestamps[idx] // 1000000 + 1, trade, file, lastResponse)

	return (ranges, diffs)

def handleGap(startTime, startId, endTime, endId, file, lastResponse):
	windowOffset = endTime - startTime
	filledTrades = []

	while True:
		while (retVal := getGap(endId, min(startTime + windowOffset, endTime), startId, startTime, filledTrades, file, lastResponse)) == RetVal.WAIT:
			# Rate limit is 30 requests per second
			time.sleep(1 / 30)

		if retVal == RetVal.FAILURE:
			break
		if retVal == RetVal.SUCCESS and startTime + windowOffset >= endTime:
			break

		if retVal == RetVal.GAP_EXCEEDED:
			windowOffset = 1
		elif retVal == RetVal.SUCCESS:
			startTime = startTime + windowOffset
			windowOffset = endTime - startTime
		else:
			# Might be able to just use the previous endTime as the new startTime like in the above case
			# startTime = int(lastTrade['Time']) // 1000000
			windowOffset = endTime - startTime

	if not filledTrades:
		print("Did not find any trades in the gap")
	else:
		print("Found trades " + str(getIntRanges(filledTrades)))
		print("")

def getGap(endId, endTime, startId, startTime, filledTrades, file, lastResponse):
	url = "https://api.coinbase.com/api/v3/brokerage/products/%s/ticker" % (symbol)
	params = {"limit": MAX_REST_API_TRADES, "start": str(startTime), "end": str(endTime)}
	jwt_uri = jwt_generator.format_jwt_uri("GET", "/api/v3/brokerage/products/%s/ticker" % (symbol))
	jwt_token = jwt_generator.build_rest_jwt(jwt_uri, COINBASE_API_KEY_NAME, COINBASE_API_PRIVATE_KEY)
	headers = {"Authorization": "Bearer " + jwt_token}
	startDate = datetime.datetime.fromtimestamp(startTime, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
	endDate = datetime.datetime.fromtimestamp(endTime, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
	try:
		if not lastResponse or lastResponse[0] != startTime or lastResponse[1] != endTime:
			logMsg = "Sending HTTP request for %s trades from %s to %s" % (symbol, startDate, endDate)
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

			lastResponse.clear()
			lastResponse.append(startTime)
			lastResponse.append(endTime)
			lastResponse.append(responseTrades)
			lastResponse.append(length)
		else:
			logMsg = "Reusing last response"
			print(logMsg)
			responseTrades = lastResponse[2]
			length = lastResponse[3]

		tradeId = startId + 1
		idx = next((i for i, x in enumerate(responseTrades) if int(x['trade_id']) >= tradeId), -1)
		if idx != -1:
			if length >= MAX_REST_API_TRADES and endTime - startTime > 1:
				logMsg = "Moving gap back"
				print(logMsg)
				return RetVal.GAP_EXCEEDED

		while (tradeId < endId):
			idx = next((i for i, x in enumerate(responseTrades) if int(x['trade_id']) == tradeId), -1)
			if (idx != -1):
				# print("Found trade %d" % (tradeId))
				# How do we know if we got all the trades in this given window or if there are still missing ones after the last?
				record = prepareRecord(responseTrades[idx])
				record.insert(1, symbol)
				record.insert(1, "price")
				file.write(",".join(record) + '\n')
				filledTrades.append(tradeId)
				if (idx == len(responseTrades) - 1):
					break
			# else:
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

	for idx, trade in enumerate(trades):
		if idx + 1 >= len(trades):
			break
		while trades[idx + 1]["trade_id"] == trade["trade_id"]:
			trades.pop(idx + 1)
			if idx + 1 >= len(trades):
				break

parser = argparse.ArgumentParser(description='Show the gaps in trading data collected from Coinbase and see if they can be filled.')
parser.add_argument('file', help='the file to analyze for gaps')
args = parser.parse_args()
inputFile = vars(args)['file']
if "ETH-USD" in inputFile:
	symbol = "ETH-USD"
else:
	symbol = "BTC-USD"

checkGaps()
