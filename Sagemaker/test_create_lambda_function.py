
import boto3
import sagemaker

#########################################################################################
ACCESS_KEY_ID = 'AKIAUHC5UZEKGAES5SEE'
ACCESS_KEY_SECRET = '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'
REGION = 'us-east-2'

role = 'arn:aws:iam::290106689812:role/KrlyMgr'


#########################################################################################
boto3_session = boto3.session.Session(
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_KEY_SECRET,
    region_name=REGION
)

sagemaker_session = sagemaker.Session(
    boto_session=boto3_session
)

bucket = sagemaker_session.default_bucket()

lambda_client = boto3_session.client('lambda')

import base64

file_content = None
with open('aws_lambda_template.zip', 'rb') as file:
    file_content = file.read()

zip_base64_encoded = base64.b64encode(file_content)

response = lambda_client.create_function(
    FunctionName='OIL-PAINTING-V8_handler',
    Runtime='python3.9',
    Handler='aws_lambda_template.lambda_handler',
    Role=role,
    PackageType='Zip',
    Code={
        'ZipFile': file_content
    },
    Publish=True,
    TracingConfig={
        'Mode': 'Active',
    },
)

