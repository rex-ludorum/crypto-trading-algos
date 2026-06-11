import argparse
import boto3
import os
import time

ACCESS_KEY = "AKIAW3MEECM242BBX6NJ"
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

REGION_NAME = "us-east-2"

def printError(error):
	errorMessage = repr(error) + " encountered at " + str(time.strftime("%H:%M:%S", time.localtime()))
	print(errorMessage)

def uploadFiles(bucket, inputFiles):
	for file in inputFiles:
		print("Uploading " + file)
		try:
			s3.upload_file(file, bucket, file)
		except Exception as e:
				printError(e)

parser = argparse.ArgumentParser(description='Upload files to an S3 bucket.')
parser.add_argument('bucket', help='the name of the bucket')
parser.add_argument('files', nargs="+", help="the files to upload")
args = parser.parse_args()
bucket = vars(args)['bucket']
inputFiles = vars(args)['files']

s3 = boto3.client("s3", region_name=REGION_NAME, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

uploadFiles(bucket, inputFiles)
