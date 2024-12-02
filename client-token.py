import asyncio
import websockets
import json
import requests
import boto3
import time
import traceback
import os
import dateutil.parser
import datetime
import argparse
from functools import cmp_to_key
from urllib.error import HTTPError

DATABASE_NAME = "coinbase-websocket-data"

COINBASE_WEBSOCKET_ARN = "arn:aws:sns:us-east-2:471112880949:coinbase-websocket-notifications"
ACCESS_KEY = "AKIAW3MEECM242BBX6NJ"
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

COINBASE_API_KEY_NAME = os.environ.get("COINBASE_API_KEY_NAME")
COINBASE_API_PRIVATE_KEY = os.environ.get("COINBASE_API_PRIVATE_KEY")

REGION_NAME = "us-east-2"

# Empirically determined number of records over which the write size will exceed 1 kB
NUM_RECORDS = 30

MAX_REST_API_TRADES = 1000

SLIDING_WINDOW_SIZE = 5
MAX_WINDOW_SIZE = 200

def pullData(symbol):
	queryClient = boto3.client('timestream-query', region_name=REGION_NAME, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
	try:
		queryString = 'SELECT * FROM "coinbase-websocket-data"."%s" WHERE time between date(TIMESTAMP \'2024-07-18 23:00:00.000000000\') and date(TIMESTAMP \'2024-07-22 02:00:00.000000000\') ORDER BY time asc' % (symbol)
		clientToken = 'stringstringstringstringstringstring'
		print("Querying %s with query %s and ClientToken %s at %s" % (symbol, queryString, clientToken, str(datetime.datetime.now())))
		response = queryClient.query(
			QueryString=queryString,
			ClientToken=clientToken
		)
		# status = result['ResponseMetadata']['HTTPStatusCode']
		print(response)
	except Exception as e:
		# publishAndPrintError(mysns, e, "Other WriteClient", symbol)
		errorMessage = repr(e) + " encountered for " + symbol + " at " + str(time.strftime("%H:%M:%S", time.localtime()))
		print(errorMessage)

def publishAndPrintError(mysns, error, subject, symbol):
	errorMessage = repr(error) + " encountered for " + symbol + " at " + str(time.strftime("%H:%M:%S", time.localtime()))
	print(errorMessage)
	try:
		mysns.publish(
			TopicArn = COINBASE_WEBSOCKET_ARN,
			Message = errorMessage,
			Subject = subject + " Exception",
		)
	except Exception as e:
		print(repr(e), "encountered at", str(time.strftime("%H:%M:%S", time.localtime())), "during publishing")

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
	record = {
		'Time': str(microseconds),
		'MeasureValues': [
			prepareMeasure('tradeId', tradeId, 'BIGINT'),
			prepareMeasure('price', response['price'], 'DOUBLE'),
			prepareMeasure('size', response['size'], 'DOUBLE'),
			prepareMeasure('isBuyerMaker', isBuyerMaker, 'BOOLEAN')
		]
	}
	return record
	'''
	except ValueError:
		print("Unknown trade_id for " + json.dumps(response))
		return {}
	'''

def prepareMeasure(name, value, measureType):
	measure = {
		'Name': name,
		'Value': str(value),
		'Type': measureType
	}
	return measure

# Obtain the range of aggregate IDs we are writing
def getTradeIds(records):
	tradeIds = []
	for record in records:
		measureValues = record['MeasureValues']
		for measure in measureValues:
			if measure['Name'] == 'tradeId':
				tradeIds.append(int(measure['Value']))
				break

	'''
	tradeIds = []
	firstTradeMeasureValues = records[0]['MeasureValues']
	for measure in firstTradeMeasureValues:
		if measure['Name'] == 'tradeId':
			tradeIds.append(measure['Value'])
			break
	lastTradeMeasureValues = records[-1]['MeasureValues']
	for measure in lastTradeMeasureValues:
		if measure['Name'] == 'tradeId':
			tradeIds.append(measure['Value'])
			break
	'''
	return tradeIds

def writeRecords(symbol, writeClient, records, commonAttributes, mysns):
	try:
		tradeIds = getTradeIds(records)
		print("Writing %d %s records (%s) at %s" % (len(records), symbol, ", ".join(getMissedRanges(tradeIds)), str(datetime.datetime.now())))
		'''
		gap = int(tradeIds[1]) - int(tradeIds[0])
		if gap != 29:
			print("Writing %d records instead of 30" % (gap + 1))
		'''
		result = writeClient.write_records(DatabaseName=DATABASE_NAME, TableName=symbol, CommonAttributes=commonAttributes, Records=records)
		status = result['ResponseMetadata']['HTTPStatusCode']
		print("Processed %d %s records (%s). WriteRecords HTTPStatusCode: %s" % (len(records), commonAttributes['Dimensions'][0]['Value'], ", ".join(getMissedRanges(tradeIds)), status))
	except writeClient.exceptions.RejectedRecordsException as e:
		# print("RejectedRecords at", str(time.strftime("%H:%M:%S", time.localtime())), ":", e)
		publishAndPrintError(mysns, e, "RejectedRecords", symbol)
		for rr in e.response["RejectedRecords"]:
			print("Rejected Index " + str(rr["RecordIndex"]) + ": " + rr["Reason"])
			print(json.dumps(records[rr['RecordIndex']], indent=2))
			if "ExistingVersion" in rr:
				print("Rejected record existing version: ", rr["ExistingVersion"])
	except Exception as e:
		publishAndPrintError(mysns, e, "Other WriteClient", symbol)

# Timestream does not allow two records with the same timestamp and dimensions to have different measure values
# Therefore, add one us to the later timestamp
def updateRecordTime(record, lastTrade, recordList, symbol):
	recordTime = record['Time']
	if lastTrade and int(record['Time']) <= int(lastTrade['Time']) + int(lastTrade['offset']):
		record['Time'] = str(int(lastTrade['Time']) + int(lastTrade['offset']) + 1)
		# print("Time %s for %s conflicts with last trade time (%s with offset %s, tradeId %s), updating to %s" % (recordTime, symbol, lastTrade['Time'], lastTrade['offset'], lastTrade['tradeId'], record['Time']))
		lastTrade['offset'] = str(int(lastTrade['offset']) + 1)
	else:
		lastTrade['Time'] = recordTime
		lastTrade['offset'] = str(0)
	lastTrade['tradeId'] = getTradeIds([record])[0]
	recordList.append(record)

# Check if we have reached the 1 kB write size and write the records
def checkWriteThreshold(symbol, writeClient, trades, commonAttributes, mysns):
	if len(trades) == NUM_RECORDS:
		# print(json.dumps(trades, indent=2))
		writeRecords(symbol, writeClient, trades, commonAttributes, mysns)
		trades.clear()

async def collectData(symbol):
	url = "wss://advanced-trade-ws.coinbase.com"
	headers = {"Sec-WebSocket-Extensions": "permessage-deflate"}
	trades = []
	lastTrade = {'Time': '0', 'offset': '0', 'tradeId': '0'}

	# Rolling window of the number of trades in each of the last 5 seconds
	windows = {"startTime": "", "windows": []}
	handleFirstGap = False

	commonAttributes = {
		'Dimensions': [
			{'Name': 'symbol', 'Value': symbol}
		],
		'MeasureName': 'price',
		'MeasureValueType': 'MULTI',
		'TimeUnit': 'MICROSECONDS'
	}

	writeClient = boto3.client('timestream-write', region_name=REGION_NAME, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

	mysns = boto3.client("sns", region_name=REGION_NAME, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

	async for websocket in websockets.connect(url, extra_headers=headers):
		jwtToken = jwt_generator.build_ws_jwt(COINBASE_API_KEY_NAME, COINBASE_API_PRIVATE_KEY)
		tradesRequest = {
			"type": "subscribe",
			"product_ids": [
				symbol
			],
			"channel": "market_trades",
			"jwt": jwtToken
		}
		heartbeatsRequest = {
			"type": "subscribe",
			"product_ids": [
				symbol
			],
			"channel": "heartbeats",
			"jwt": jwtToken
		}

		try:
			await websocket.send(json.dumps(heartbeatsRequest))
			await websocket.send(json.dumps(tradesRequest))
			while True:
				response = json.loads(await websocket.recv())
				if response['channel'] == 'market_trades' and response['events'][0]['type'] == 'update':
					responseTrades = response['events'][0]['trades']
					cleanTrades(responseTrades)
					for trade in responseTrades:
						if lastTrade['tradeId'] == '0' or int(trade['trade_id']) > int(lastTrade['tradeId']):
							record = prepareRecord(trade)
							if handleFirstGap or lastTrade['tradeId'] != '0':
								handleGap(trade, trades, lastTrade, windows, writeClient, commonAttributes, mysns)
							updateRecordTime(record, lastTrade, trades, trade['product_id'])
							adjustWindow(record, windows)
							checkWriteThreshold(trade['product_id'], writeClient, trades, commonAttributes, mysns)
		except websockets.ConnectionClosedOK as e:
			traceback.print_exc()
			publishAndPrintError(mysns, e, "Websocket ConnectionClosedOK", symbol)
		except websockets.ConnectionClosedError as e:
			traceback.print_exc()
			publishAndPrintError(mysns, e, "Websocket ConnectionClosedError", symbol)
		except Exception as e:
			traceback.print_exc()
			print(trade)
			print(lastTrade)
			print(response)
			publishAndPrintError(mysns, e, "Other Websocket", symbol)
			break

def adjustWindow(record, windows):
	recordTime = str(int(record['Time']) // 1000000)
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

# If we have to reconnect after a websocket exception, get any trades we might have missed
def handleGap(response, trades, lastTrade, windows, writeClient, commonAttributes, mysns):
	if int(response['trade_id']) > int(lastTrade['tradeId']) + 1:
		time.sleep(0.5)
		endId = int(response['trade_id'])
		endDate = dateutil.parser.isoparse(response['time'])
		endTime = int(datetime.datetime.timestamp(endDate)) + 1
		startTime = int(lastTrade['Time']) // 1000000
		startMicros = str(int(lastTrade['Time']) % 1000000)
		startDate = datetime.datetime.fromtimestamp(startTime).strftime('%Y-%m-%dT%H:%M:%S') + '.' + startMicros.zfill(6)
		logMsg = "Gap found: %s - %s (%s - %s)" % (lastTrade['tradeId'], response['trade_id'], startDate, response['time'])
		print(logMsg)
		log = [logMsg]
		prevLastTradeId = lastTrade['tradeId']
		missedTrades = []
		startingLastTradeId = lastTrade['tradeId']
		while True:
			# Check old last trade time and increase the gap if it's still the same
			# Use a three-second-minimum window since the max observed trades in a one-second window was 260 on Feb 28 2024
			# Actually, see the test commands for a one-second window with more than 1000 trades
			if max(windows["windows"]) < MAX_WINDOW_SIZE:
				windowOffset = max(MAX_REST_API_TRADES // max(1, computeAverage(windows)), 1)
			else:
				windowOffset = 1
			while not getGap(response['product_id'], endId, min(startTime + windowOffset, endTime), trades, startTime, lastTrade, missedTrades, log, False, windows, writeClient, commonAttributes, mysns):
				# Rate limit is 30 requests per second
				time.sleep(1 / 30)
			if startTime + windowOffset >= endTime or int(response['trade_id']) == int(lastTrade['tradeId']) + 1:
				break
			if prevLastTradeId == lastTrade['tradeId']:
				startTime = startTime + windowOffset
				# endTimeOffset += 10
			else:
				startTime = int(lastTrade['Time']) // 1000000
				# endTimeOffset = 0
			prevLastTradeId = lastTrade['tradeId']

		if not missedTrades:
			missedTrades = list(range(int(lastTrade['tradeId']) + 1, endId))
		else:
			lastMissedTradeId = max(int(lastTrade['tradeId']) + 1, missedTrades[-1] + 1)
			missedTrades.extend(range(lastMissedTradeId, endId))

		if not all(missedTrades[i] <= missedTrades[i + 1] for i in range(len(missedTrades) - 1)):
			publishAndPrintError(mysns, RuntimeError("List of missed trades is not in increasing order: %s" % (" ".join(str(x) for x in missedTrades))), "Requests", symbol)
		if len(missedTrades) > len(set(missedTrades)):
			publishAndPrintError(mysns, RuntimeError("List of missed trades has duplicates: %s" % (" ".join(str(x) for x in missedTrades))), "Requests", symbol)
		if int(response['trade_id']) != int(lastTrade['tradeId']) + 1 or missedTrades:
			errMsg = "Gaps still exist between %s and %s: %s\n" % (startingLastTradeId, response['trade_id'], ", ".join(getMissedRanges(missedTrades)))
			errMsg += "\n".join(log)
			publishAndPrintError(mysns, RuntimeError(errMsg), "Requests", symbol)

def getGap(symbol, endId, endTime, trades, startTime, lastTrade, missedTrades, log, retried, windows, writeClient, commonAttributes, mysns):
	url = "https://api.coinbase.com/api/v3/brokerage/products/%s/ticker" % (symbol)
	params = {"limit": MAX_REST_API_TRADES, "start": str(startTime), "end": str(endTime)}
	jwt_uri = jwt_generator.format_jwt_uri("GET", "/api/v3/brokerage/products/%s/ticker" % (symbol))
	jwt_token = jwt_generator.build_rest_jwt(jwt_uri, COINBASE_API_KEY_NAME, COINBASE_API_PRIVATE_KEY)
	headers = {"Authorization": "Bearer " + jwt_token}
	startDate = datetime.datetime.fromtimestamp(startTime).strftime('%Y-%m-%dT%H:%M:%S')
	endDate = datetime.datetime.fromtimestamp(endTime).strftime('%Y-%m-%dT%H:%M:%S')
	try:
		logMsg = "Sending HTTP request for %s trades from %s to %s (lastTradeId: %s)" % (symbol, startDate, endDate, lastTrade['tradeId'])
		print(logMsg)
		log.append(logMsg)
		response = requests.get(url, params=params, headers=headers)
		response.raise_for_status()
		responseTrades = response.json()['trades']
		if not responseTrades:
			logMsg = "HTTP response contains 0 trades"
			print(logMsg)
			log.append(logMsg)
			return True

		cleanTrades(responseTrades)
		logMsg = "HTTP response contains %d trades (%s) (%s - %s)" % (len(responseTrades), ", ".join(getRanges(responseTrades)), responseTrades[0]['time'], responseTrades[-1]['time'])
		print(logMsg)
		log.append(logMsg)
		# print(responseTrades)

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
		tradeId = int(lastTrade['tradeId'])
		if tradeId >= int(responseTrades[-1]['trade_id']):
			if not retried:
				time.sleep(0.5)
				logMsg = "Retrying - tradeId %s is geq %s" % (lastTrade['tradeId'], responseTrades[-1]['trade_id'])
				print(logMsg)
				log.append(logMsg)
				return getGap(symbol, endId, endTime, trades, startTime, lastTrade, missedTrades, log, True, windows, writeClient, commonAttributes, mysns)
			else:
				return True

		'''
		# Fringe case for when the lastTrade is the very last trade in the response
		idx = next((i for i, x in enumerate(responseTrades) if int(x['trade_id']) == tradeId), -1)
		if (idx == len(responseTrades) - 1):
			return True
		'''

		tradeId = int(lastTrade['tradeId']) + 1
		# idx = next((i for i, x in enumerate(responseTrades) if int(x['trade_id']) == tradeId), -1)
		# if (idx == -1):
			# raise LookupError("Last trade not found for ID " + str(tradeId))
		# tradeId += 1
		while (tradeId < endId):
			# if (idx == len(responseTrades) - 1):
				# break;
			idx = next((i for i, x in enumerate(responseTrades) if int(x['trade_id']) == tradeId), -1)
			if (idx != -1):
				record = prepareRecord(responseTrades[idx])
				updateRecordTime(record, lastTrade, trades, symbol)
				adjustWindow(record, windows)
				checkWriteThreshold(symbol, writeClient, trades, commonAttributes, mysns)
				# How do we know if we got all the trades in this given window or if there are still missing ones after the last?
				if (idx == len(responseTrades) - 1):
					break;
			else:
				if not retried:
					time.sleep(0.5)
					logMsg = "Retrying - tradeId %s was not found" % (lastTrade['tradeId'])
					print(logMsg)
					log.append(logMsg)
					return getGap(symbol, endId, endTime, trades, startTime, lastTrade, missedTrades, log, True, windows, writeClient, commonAttributes, mysns)
				else:
					missedTrades.append(tradeId)
				# publishAndPrintError(mysns, LookupError("Trade ID " + str(tradeId) + " not found"), "Requests", symbol)
			tradeId += 1
		return True
	except HTTPError as e:
		publishAndPrintError(mysns, e, "Requests", symbol)
		return not e.code == 429
	except Exception as e:
		publishAndPrintError(mysns, e, "Requests", symbol)
		return True

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
	'''
	first = True
	for i, e in reversed(list(enumerate(trades))):
		if first:
			first = False
			continue
		if int(e['trade_id']) != int(trades[i + 1]['trade_id']) - 1:
			del(trades[:i+1])
			break
	'''

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

parser = argparse.ArgumentParser(description='Collect trading data from Coinbase and send it to AWS Timestream.')
parser.add_argument('symbol', help='the trading pair to collect data from', choices=['BTC-USD', 'ETH-USD'])
args = parser.parse_args()
symbol = vars(args)['symbol']

pullData(symbol)

# Test commands for handling gaps
'''
commonAttributes = {
	'Dimensions': [
		{'Name': 'symbol', 'Value': symbol}
	],
	'MeasureName': 'price',
	'MeasureValueType': 'MULTI',
	'TimeUnit': 'MICROSECONDS'
}
'''

# writeClient = boto3.client('timestream-write', region_name=REGION_NAME, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# mysns = boto3.client("sns", region_name=REGION_NAME, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
# handleGap({'trade_id': '999999999', 'time': '2024-06-02T00:00:00.000000Z', 'product_id': 'BTC-USD'}, [], {'Time': '1617216665966502', 'offset': '0', 'tradeId': '151436694'}, writeClient, commonAttributes, mysns)
# handleGap({'trade_id': '151436698', 'time': '2024-06-02T00:00:00.000000Z', 'product_id': 'BTC-USD'}, [], {'Time': '1617216665966502', 'offset': '0', 'tradeId': '151436694'}, {}, {}, {})

# Used to find the max window size (260)
# getGap(symbol, 999999999, 1709144520, [], 1705311931, {}, [], [], [])

# Shows that there are unavoidable gaps in the trade data
# getGap(symbol, 655395509, 1719187536, [], 1719187535, {'tradeId': '655395410'}, [], [], [])

# Shows that you should keep the window as small as possible otherwise trades might be missed
# getGap(symbol, 655395509, 1719187537, [], 1719187535, {'tradeId': '655395410'}, [], [], [])
# getGap(symbol, 655395509, 1719187537, [], 1719187536, {'tradeId': '655395410'}, [], [], [])
