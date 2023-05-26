
import boto3
import json

#########################################################################################
ACCESS_KEY_ID = 'AKIAUHC5UZEKGAES5SEE'
ACCESS_KEY_SECRET = '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'
REGION = 'us-east-2'

ENDPOINT = 'WATER-COLOR-V3-7c562771-f399-41f5-b1c6-88fc92a4f560'

#########################################################################################
boto3_session = boto3.session.Session(
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_KEY_SECRET,
    region_name=REGION
)

lambda_client = boto3_session.client('lambda')
sagemaker_client = boto3_session.client('sagemaker-runtime')


#########################################################################################
def make_result_json(status_code, body, cls=None):
    if cls is not None:
        body = json.dumps(body, cls=cls)
    else:
        body = json.dumps(body)
    return {
        'statusCode': status_code,
        'isBase64Encoded': False,
        'headers': {
            'Content-Type': 'application/json',
            'access-control-allow-origin': '*',
            'access-control-allow-methods': '*',
            'access-control-allow-headers': '*'

        },
        'body': body
    }


def lambda_handler(event, context):
    body = event.get("body", "")
    body = json.loads(body)
    if body == "":
        return make_result_json(400, {"msg": "need prompt"})

    input_s3_path = body['input_s3_path']

    response = sagemaker_client.invoke_endpoint_async(
        EndpointName=ENDPOINT,
        InputLocation=input_s3_path
    )
    return response

