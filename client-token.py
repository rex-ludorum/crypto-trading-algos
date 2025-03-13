import boto3
import json
import time
import traceback
import os
import datetime
import argparse

DATABASE_NAME = "coinbase-websocket-data"

ACCESS_KEY = "AKIAW3MEECM242BBX6NJ"
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

REGION_NAME = "us-east-2"

START_DATE = "2024-07-17 22:28:47"
END_DATE = "2024-07-18 00:12:55"
CLIENT_TOKEN = "clientTokenClientTokenClientToken1"

def getFirstNextToken():
	queryClient = boto3.client('timestream-query', region_name=REGION_NAME, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
	try:
		queryString = 'SELECT * FROM "coinbase-websocket-data"."%s" WHERE time between TIMESTAMP \'%s\' and TIMESTAMP \'%s\' ORDER BY time asc' % (symbol, START_DATE, END_DATE)
		print("Querying %s with query %s and ClientToken %s at %s" % (symbol, queryString, CLIENT_TOKEN, str(datetime.datetime.now())))
		response = queryClient.query(
			QueryString=queryString,
			ClientToken=CLIENT_TOKEN
		)
		# print(response)
		return response
	except Exception as e:
		traceback.print_exc()
		printError(e)
		return {}

def pullData(nextToken):
	try:
		queryString = 'SELECT * FROM "coinbase-websocket-data"."%s" WHERE time between date(TIMESTAMP \'%s\') and date(TIMESTAMP \'%s\') ORDER BY time asc' % (symbol, START_DATE, END_DATE)
		print("Querying %s at %s" % (symbol, str(datetime.datetime.now())))
		response = queryClient.query(
			QueryString=queryString,
			ClientToken=CLIENT_TOKEN,
			NextToken=nextToken
		)
		# print(response)
		return response
	except Exception as e:
		traceback.print_exc()
		printError(e)
		return {}

def checkResponse(response, queryId):
	if not response or response['ResponseMetadata']['HTTPStatusCode'] != 200:
		print("Error with retrieval:")
		print(response)
		try:
			print("Cancelling query:")
			response = queryClient.cancel_query(
				QueryId=queryId
			)
			if response['ResponseMetadata']['HTTPStatusCode'] != 200:
				print("Error cancelling query:")
				print(response)
			else:
				print("Successfully cancelled query:")
				print(response)
			return False
		except Exception as e:
			traceback.print_exc()
			printError(e)
			return False
	else:
		return True

def printProgress(response):
	try:
		print("Current progress: " + str(response['QueryStatus']['ProgressPercentage']))
		print("Cumulative bytes scanned: " + str(response['QueryStatus']['CumulativeBytesScanned']))
		print("Cumulative bytes metered: " + str(response['QueryStatus']['CumulativeBytesMetered']))
		print("Response has %d rows" % (len(response['Rows'])))
	except Exception as e:
		traceback.print_exc()
		printError(e)

def saveToFile():
	response = getFirstNextToken()
	if not response or response['ResponseMetadata']['HTTPStatusCode'] != 200:
		print("Retrieval failed!")
		return
	queryId = response["QueryId"]
	print("QueryId: " + queryId)
	if "NextToken" not in response:
		print("No NextToken:")
		print(json.dumps(response, indent=2))
		return
	else:
		nextToken = response["NextToken"]
	with open('%s_%s_%s' % (symbol, START_DATE.replace(" ", "-").replace(":", "."), END_DATE.replace(" ", "-").replace(":", ".")), 'w') as file:
		response = pullData(nextToken)
		if not checkResponse(response, queryId):
			return
		printProgress(response)
		while response:
			for row in response['Rows']:
				rowData = row['Data']
				date = rowData[2]['ScalarValue']
				qty = rowData[3]['ScalarValue']
				price = rowData[4]['ScalarValue']
				isBuyerMaker = rowData[5]['ScalarValue']
				tradeId = rowData[6]['ScalarValue']
				file.write(" ".join([tradeId, date, price, qty, isBuyerMaker]) + '\n')
			if "NextToken" in response:
				nextToken = response['NextToken']
				response = pullData(nextToken)
				if not checkResponse(response, queryId):
					break
				printProgress(response)
			else:
				print("Query finished:")
				print(response)
				break

def printError(error):
	errorMessage = repr(error) + " encountered for " + symbol + " at " + str(time.strftime("%H:%M:%S", time.localtime()))
	print(errorMessage)

parser = argparse.ArgumentParser(description='Retrieve trading data from AWS Timestream.')
parser.add_argument('symbol', help='the trading pair to collect data from', choices=['BTC-USD', 'ETH-USD'])
args = parser.parse_args()
symbol = vars(args)['symbol']

queryClient = boto3.client('timestream-query', region_name=REGION_NAME, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

saveToFile()
