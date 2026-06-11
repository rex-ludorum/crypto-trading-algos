import json
import requests
import boto3
import time
import traceback
import os
import dateutil.parser
import datetime
import argparse
import sys
from coinbase import jwt_generator
from functools import cmp_to_key
from enum import Enum, auto

DATABASE_NAME = "coinbase-data-really-fixed"

ACCESS_KEY = "AKIAW3MEECM242BBX6NJ"
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

REGION_NAME = "us-east-2"

MAX_WRITE_BATCH_SIZE = 100

def printError(error):
	errorMessage = repr(error) + " encountered for " + symbol + " at " + str(time.strftime("%H:%M:%S", time.localtime()))
	print(errorMessage)

def prepareRecord(data, version=None):
	isBuyerMaker = data[6].lower() == 'true'

	microseconds = data[0]
	tradeId = data[3]
	price = data[4]
	size = data[5]

	record = {
		'Time': str(microseconds),
		'MeasureValues': [
			prepareMeasure('tradeId', tradeId, 'BIGINT'),
			prepareMeasure('price', price, 'DOUBLE'),
			prepareMeasure('size', size, 'DOUBLE'),
			prepareMeasure('isBuyerMaker', isBuyerMaker, 'BOOLEAN')
		],
	}

	if version:
		record['Version'] = version

	return record

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

	return tradeIds

def writeRecords(records):
	try:
		tradeIds = getTradeIds(records)
		print("Writing %d %s records (%s) at %s" % (len(records), symbol, ", ".join(getMissedRanges(tradeIds)), str(datetime.datetime.now(datetime.timezone.utc))))
		result = writeClient.write_records(DatabaseName=DATABASE_NAME, TableName=symbol, CommonAttributes=commonAttributes, Records=records)
		status = result['ResponseMetadata']['HTTPStatusCode']
		print("Processed %d %s records (%s). WriteRecords HTTPStatusCode: %s" % (len(records), commonAttributes['Dimensions'][0]['Value'], ", ".join(getMissedRanges(tradeIds)), status))
	except writeClient.exceptions.RejectedRecordsException as e:
		printError(e)
		for rr in e.response["RejectedRecords"]:
			print("Rejected Index " + str(rr["RecordIndex"]) + ": " + rr["Reason"])
			print(json.dumps(records[rr['RecordIndex']], indent=2))
			if "ExistingVersion" in rr:
				print("Rejected record existing version: ", rr["ExistingVersion"])
	except Exception as e:
		printError(e)

def checkWriteThreshold(trades, forceWrite):
	if not forceWrite:
		if len(trades) == MAX_WRITE_BATCH_SIZE:
			writeRecords(trades)
			trades.clear()
	elif trades:
		writeRecords(trades)
		trades.clear()

def uploadTrades(file, startId, endId, version):
	recordsToWrite = []
	with open(file, "r") as f:
		f.readline()
		for line in f:
			data = line.split(",")
			if int(data[3]) >= startId and int(data[3]) <= endId:
				recordsToWrite.append(prepareRecord(data, version))
				checkWriteThreshold(recordsToWrite, False)

	# print(recordsToWrite)
	checkWriteThreshold(recordsToWrite, True)

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

parser = argparse.ArgumentParser(description='Collect trading data from Coinbase and send it to AWS Timestream.')
parser.add_argument('file', help='the file containing trades to upload', type=str)
parser.add_argument('startId', help='the trade ID of the first trade to upload', type=int)
parser.add_argument('endId', help='the trade ID of the last trade to upload', type=int)
parser.add_argument('-v', '--version', type=int, help='the version number to use for the Timestream records')
args = parser.parse_args()
file = vars(args)['file']
startId = vars(args)['startId']
endId = vars(args)['endId']
version = vars(args)['version']

symbol = None
if "BTC-USD" in file:
	symbol = "BTC-USD"
elif "ETH-USD" in file:
	symbol = "ETH-USD"
else:
	print("Symbol not found")
	sys.exit(1)

commonAttributes = {
	'Dimensions': [
		{'Name': 'symbol', 'Value': symbol}
	],
	'MeasureName': 'price',
	'MeasureValueType': 'MULTI',
	'TimeUnit': 'MICROSECONDS'
}

writeClient = boto3.client('timestream-write', region_name=REGION_NAME, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

uploadTrades(file, startId, endId, version)
